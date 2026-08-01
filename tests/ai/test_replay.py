import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ai.replay import SCHEMA_VERSION, ReplayBuffer, ReplayCompatibilityError
from ai.self_play import GameResult, TrainingSample
from xiangqi.domain import Color


def _game(marker: int, samples: int = 2) -> GameResult:
    entries = tuple(
        TrainingSample(
            state=np.full((15, 10, 9), marker + index, dtype=np.float32),
            policy_indices=np.asarray([marker, marker + 10], dtype=np.int64),
            policy_probabilities=np.asarray([0.25, 0.75], dtype=np.float32),
            side=Color.RED if index % 2 == 0 else Color.BLACK,
            value=float((-1) ** index),
        )
        for index in range(samples)
    )
    return GameResult(entries, Color.RED, samples, "checkmate")


def _downgrade_to_v1(path: Path) -> bytes:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 1
    manifest.pop("total_games")
    manifest.pop("sample_counts")
    for game_id in manifest["games"]:
        game_path = path / "games" / f"{game_id:012d}.npz"
        with np.load(game_path, allow_pickle=False) as stored:
            payload = {key: stored[key] for key in stored.files}
        payload["schema_version"] = np.asarray([1], dtype=np.int64)
        with game_path.open("wb") as stream:
            np.savez_compressed(stream, **payload)
    legacy = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()
    manifest_path.write_bytes(legacy)
    return legacy


def test_replay_commits_complete_games_and_evicts_oldest(tmp_path: Path) -> None:
    replay = ReplayBuffer(tmp_path, capacity_games=2)

    assert replay.append_game(_game(1)) == 1
    assert replay.append_game(_game(2)) == 2
    assert replay.append_game(_game(3)) == 3

    reopened = ReplayBuffer(tmp_path, capacity_games=2)
    assert reopened.game_ids == (2, 3)
    assert not (tmp_path / "games" / "000000000001.npz").exists()


def test_manifest_persists_total_games_and_each_game_sample_count(
    tmp_path: Path,
) -> None:
    replay = ReplayBuffer(tmp_path, capacity_games=2)
    replay.append_game(_game(1, samples=1))
    replay.append_game(_game(2, samples=2))
    replay.append_game(_game(3, samples=3))

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["total_games"] == 3
    assert manifest["next_game_id"] == 4
    assert manifest["sample_counts"] == {"2": 2, "3": 3}
    assert replay.total_games == 3
    assert replay.sample_count == 5


@pytest.mark.parametrize(("games", "capacity"), [(0, 2), (3, 1)])
def test_v1_manifest_is_atomically_migrated_with_counts_and_cursor(
    tmp_path: Path, games: int, capacity: int
) -> None:
    replay = ReplayBuffer(tmp_path, capacity_games=capacity)
    for marker in range(1, games + 1):
        replay.append_game(_game(marker, samples=marker))
    legacy = _downgrade_to_v1(tmp_path)

    migrated = ReplayBuffer(tmp_path, capacity_games=capacity)
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert migrated.legacy_manifest_hash == hashlib.sha256(legacy).hexdigest()
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["total_games"] == games
    assert migrated.total_games == games
    assert migrated.game_ids == (() if games == 0 else (games,))
    assert manifest["sample_counts"] == ({} if games == 0 else {str(games): games})
    if games:
        assert migrated.sample(1, np.random.default_rng(1)).states.shape[0] == 1


def test_migration_sidecar_survives_crash_before_manifest_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ReplayBuffer(tmp_path, capacity_games=2).append_game(_game(1))
    legacy = _downgrade_to_v1(tmp_path)
    original = ReplayBuffer._write_manifest

    def crash_before_v2(self: ReplayBuffer, manifest: dict[str, object]) -> None:
        if manifest.get("schema_version") == SCHEMA_VERSION:
            raise OSError("crash before manifest")
        original(self, manifest)

    monkeypatch.setattr(ReplayBuffer, "_write_manifest", crash_before_v2)
    with pytest.raises(OSError, match="crash before manifest"):
        ReplayBuffer(tmp_path, capacity_games=2)

    assert (tmp_path / "manifest.json").read_bytes() == legacy
    assert (tmp_path / "migration-v1-to-v2.json").is_file()

    monkeypatch.setattr(ReplayBuffer, "_write_manifest", original)
    recovered = ReplayBuffer(tmp_path, capacity_games=2)
    assert recovered.total_games == 1
    assert recovered.legacy_manifest_hash == hashlib.sha256(legacy).hexdigest()


def test_v2_with_valid_sidecar_restores_predecessor_after_restart(
    tmp_path: Path,
) -> None:
    ReplayBuffer(tmp_path, capacity_games=2).append_game(_game(1))
    legacy = _downgrade_to_v1(tmp_path)
    migrated = ReplayBuffer(tmp_path, capacity_games=2)
    assert migrated.migration_path.is_file()

    reopened = ReplayBuffer(tmp_path, capacity_games=2)

    assert reopened.legacy_manifest_hash == hashlib.sha256(legacy).hexdigest()


@pytest.mark.parametrize(
    "corruption",
    ["missing_field", "bad_hex", "wrong_new_hash", "wrong_total", "bad_type"],
)
def test_corrupt_migration_sidecar_is_rejected(tmp_path: Path, corruption: str) -> None:
    ReplayBuffer(tmp_path, capacity_games=2).append_game(_game(1))
    _downgrade_to_v1(tmp_path)
    migrated = ReplayBuffer(tmp_path, capacity_games=2)
    sidecar = json.loads(migrated.migration_path.read_text())
    if corruption == "missing_field":
        sidecar.pop("legacy_manifest_hash")
    elif corruption == "bad_hex":
        sidecar["legacy_manifest_hash"] = "xyz"
    elif corruption == "wrong_new_hash":
        sidecar["new_manifest_hash"] = "f" * 64
    elif corruption == "bad_type":
        sidecar["schema_version"] = True
    else:
        sidecar["total_games"] = 999
    migrated.migration_path.write_text(json.dumps(sidecar))

    with pytest.raises(ReplayCompatibilityError, match="migration|迁移"):
        ReplayBuffer(tmp_path, capacity_games=2)


def test_normal_v2_without_sidecar_has_no_predecessor(tmp_path: Path) -> None:
    replay = ReplayBuffer(tmp_path, capacity_games=2)

    assert replay.legacy_manifest_hash is None
    assert not replay.migration_path.exists()


def test_sidecar_replace_failure_keeps_v1_manifest_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ReplayBuffer(tmp_path, capacity_games=2).append_game(_game(1))
    legacy = _downgrade_to_v1(tmp_path)
    real_replace = os.replace

    def fail_sidecar_replace(source: object, destination: object) -> None:
        if Path(destination) == tmp_path / "migration-v1-to-v2.json":
            raise OSError("sidecar replace failed")
        real_replace(source, destination)

    monkeypatch.setattr("ai.replay.os.replace", fail_sidecar_replace)
    with pytest.raises(OSError, match="sidecar replace failed"):
        ReplayBuffer(tmp_path, capacity_games=2)

    assert (tmp_path / "manifest.json").read_bytes() == legacy
    assert not (tmp_path / "migration-v1-to-v2.json").exists()
    monkeypatch.setattr("ai.replay.os.replace", real_replace)
    assert ReplayBuffer(tmp_path, capacity_games=2).total_games == 1


def test_sidecar_delete_failure_retains_persisted_retry_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ReplayBuffer(tmp_path, capacity_games=2).append_game(_game(1))
    _downgrade_to_v1(tmp_path)
    migrated = ReplayBuffer(tmp_path, capacity_games=2)
    real_unlink = Path.unlink

    def fail_sidecar_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path == migrated.migration_path:
            raise OSError("sidecar delete failed")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_sidecar_unlink)
    with pytest.raises(OSError, match="sidecar delete failed"):
        migrated.clear_migration()

    assert migrated.migration_path.is_file()
    monkeypatch.setattr(Path, "unlink", real_unlink)
    retry = ReplayBuffer(tmp_path, capacity_games=2)
    retry.clear_migration()
    assert not retry.migration_path.exists()


@pytest.mark.parametrize(
    "corruption", ["missing_game", "noncontiguous", "payload", "extra_field"]
)
def test_corrupt_v1_is_rejected_without_rewriting_manifest(
    tmp_path: Path, corruption: str
) -> None:
    replay = ReplayBuffer(tmp_path, capacity_games=2)
    replay.append_game(_game(1))
    legacy = _downgrade_to_v1(tmp_path)
    if corruption == "missing_game":
        (tmp_path / "games" / "000000000001.npz").unlink()
    elif corruption == "noncontiguous":
        manifest = json.loads(legacy)
        manifest["games"] = [2]
        legacy = json.dumps(manifest, sort_keys=True).encode()
        (tmp_path / "manifest.json").write_bytes(legacy)
    elif corruption == "extra_field":
        manifest = json.loads(legacy)
        manifest["sample_counts"] = {"1": 2}
        legacy = json.dumps(manifest, sort_keys=True).encode()
        (tmp_path / "manifest.json").write_bytes(legacy)
    else:
        game_path = tmp_path / "games" / "000000000001.npz"
        game_path.write_bytes(b"not npz")

    before = (tmp_path / "manifest.json").read_bytes()
    with pytest.raises(ReplayCompatibilityError):
        ReplayBuffer(tmp_path, capacity_games=2)

    assert (tmp_path / "manifest.json").read_bytes() == before


def test_sample_loads_only_games_selected_for_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay = ReplayBuffer(tmp_path, capacity_games=2_000)
    replay.append_game(_game(1, samples=1))
    source = tmp_path / "games" / "000000000001.npz"
    for game_id in range(2, 2_001):
        os.link(source, tmp_path / "games" / f"{game_id:012d}.npz")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest.update(
        next_game_id=2_001,
        total_games=2_000,
        games=list(range(1, 2_001)),
        sample_counts={str(game_id): 1 for game_id in range(1, 2_001)},
    )
    replay._write_manifest(manifest)
    replay = ReplayBuffer(tmp_path, capacity_games=2_000)
    loaded: list[int] = []
    original = replay._read_game

    def track(game_id: int) -> dict[str, np.ndarray]:
        loaded.append(game_id)
        return original(game_id)

    monkeypatch.setattr(replay, "_read_game", track)

    replay.sample(8, np.random.default_rng(7))

    assert len(loaded) == len(set(loaded))
    assert len(loaded) <= 8


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest.pop("total_games"),
        lambda manifest: manifest.pop("sample_counts"),
        lambda manifest: manifest["sample_counts"].update({"1": 999}),
        lambda manifest: manifest.update(total_games=99),
    ],
)
def test_manifest_rejects_invalid_persisted_counts(
    tmp_path: Path, mutation: Callable[[dict[str, object]], object]
) -> None:
    ReplayBuffer(tmp_path, capacity_games=2).append_game(_game(1))
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    mutation(manifest)
    path.write_text(json.dumps(manifest))

    with pytest.raises(ReplayCompatibilityError):
        ReplayBuffer(tmp_path, capacity_games=2)


def test_replay_sample_restores_dense_states_sparse_policies_and_values(
    tmp_path: Path,
) -> None:
    replay = ReplayBuffer(tmp_path, capacity_games=3)
    replay.append_game(_game(4))

    batch = replay.sample(2, np.random.default_rng(2))

    assert batch.states.shape == (2, 15, 10, 9)
    assert batch.states.dtype == np.float32
    assert tuple(tuple(array.tolist()) for array in batch.policy_indices) == (
        (4, 14),
        (4, 14),
    )
    assert tuple(tuple(array.tolist()) for array in batch.policy_probabilities) == (
        (0.25, 0.75),
        (0.25, 0.75),
    )
    assert sorted(batch.values.tolist()) == [-1.0, 1.0]


def test_replay_rejects_incompatible_manifest(tmp_path: Path) -> None:
    ReplayBuffer(tmp_path, capacity_games=1)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        manifest.read_text().replace(
            f'"schema_version": {SCHEMA_VERSION}', '"schema_version": 99'
        )
    )

    with pytest.raises(ReplayCompatibilityError, match="schema_version"):
        ReplayBuffer(tmp_path, capacity_games=1)


def test_failed_game_write_does_not_publish_manifest_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay = ReplayBuffer(tmp_path, capacity_games=2)

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("ai.replay.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        replay.append_game(_game(5))

    reopened = ReplayBuffer(tmp_path, capacity_games=2)
    assert reopened.game_ids == ()


def test_game_payload_version_is_checked_when_sampled(tmp_path: Path) -> None:
    replay = ReplayBuffer(tmp_path, capacity_games=1)
    replay.append_game(_game(6))
    game_path = tmp_path / "games" / "000000000001.npz"
    with np.load(game_path, allow_pickle=False) as stored:
        payload = {key: stored[key] for key in stored.files}
    payload["action_version"] = np.asarray([99], dtype=np.int64)
    with game_path.open("wb") as stream:
        np.savez_compressed(stream, **payload)

    with pytest.raises(ReplayCompatibilityError, match="action_version"):
        replay.sample(1, np.random.default_rng(0))


def test_fractional_game_version_is_not_truncated_to_current_version(
    tmp_path: Path,
) -> None:
    replay = ReplayBuffer(tmp_path, capacity_games=1)
    replay.append_game(_game(6))
    game_path = tmp_path / "games" / "000000000001.npz"
    with np.load(game_path, allow_pickle=False) as stored:
        payload = {key: stored[key] for key in stored.files}
    payload["action_version"] = np.asarray([1.5], dtype=np.float64)
    with game_path.open("wb") as stream:
        np.savez_compressed(stream, **payload)

    with pytest.raises(ReplayCompatibilityError, match="action_version"):
        replay.sample(1, np.random.default_rng(0))


def test_manifest_fsync_error_never_leaves_published_manifest_without_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay = ReplayBuffer(tmp_path, capacity_games=1)
    real_fsync = replay._fsync_directory

    def fail_after_manifest_replace(directory: Path) -> None:
        if directory == tmp_path:
            raise OSError("manifest directory fsync failed")
        real_fsync(directory)

    monkeypatch.setattr(replay, "_fsync_directory", fail_after_manifest_replace)
    with pytest.raises(OSError, match="manifest directory fsync failed"):
        replay.append_game(_game(7))

    reopened = ReplayBuffer(tmp_path, capacity_games=1)
    assert reopened.game_ids == (1,)


def test_corrupt_sparse_policy_offsets_are_explicitly_rejected(tmp_path: Path) -> None:
    replay = ReplayBuffer(tmp_path, capacity_games=1)
    replay.append_game(_game(8))
    game_path = tmp_path / "games" / "000000000001.npz"
    with np.load(game_path, allow_pickle=False) as stored:
        payload = {key: stored[key] for key in stored.files}
    payload["policy_offsets"] = np.asarray([0, 999], dtype=np.int64)
    with game_path.open("wb") as stream:
        np.savez_compressed(stream, **payload)

    with pytest.raises(ReplayCompatibilityError, match="policy_offsets"):
        replay.sample(1, np.random.default_rng(0))


def test_manifest_rejects_nonadvancing_game_cursor(tmp_path: Path) -> None:
    replay = ReplayBuffer(tmp_path, capacity_games=2)
    replay.append_game(_game(9))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        manifest.read_text().replace('"next_game_id": 2', '"next_game_id": 1')
    )

    with pytest.raises(ReplayCompatibilityError, match="next_game_id"):
        ReplayBuffer(tmp_path, capacity_games=2)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("states", lambda value: np.full_like(value, np.nan)),
        ("sides", lambda value: np.full_like(value, 2)),
        ("plies", lambda value: np.asarray([-1], dtype=np.int64)),
        ("plies", lambda value: np.asarray([1], dtype=np.int64)),
        ("plies", lambda value: np.asarray([[2]], dtype=np.int64)),
        ("policy_indices", lambda value: np.full_like(value, int(value[0]))),
        ("policy_indices", lambda value: value.astype(np.float32)),
        ("policy_offsets", lambda value: value.astype(np.float32) + 0.25),
    ],
)
def test_poisoned_game_payload_is_explicitly_rejected(
    tmp_path: Path, field: str, replacement: Callable[[np.ndarray], np.ndarray]
) -> None:
    replay = ReplayBuffer(tmp_path, capacity_games=1)
    replay.append_game(_game(10))
    game_path = tmp_path / "games" / "000000000001.npz"
    with np.load(game_path, allow_pickle=False) as stored:
        payload = {key: stored[key] for key in stored.files}
    payload[field] = replacement(payload[field])
    with game_path.open("wb") as stream:
        np.savez_compressed(stream, **payload)

    with pytest.raises(ReplayCompatibilityError):
        replay.sample(1, np.random.default_rng(0))


@pytest.mark.parametrize(
    "corruption", ["state", "duplicate_action", "negative_plies", "plies_mismatch"]
)
def test_append_rejects_invalid_complete_game_before_publishing(
    tmp_path: Path, corruption: str
) -> None:
    game = _game(11)
    if corruption == "state":
        sample = replace(
            game.samples[0], state=np.full((15, 10, 9), np.nan, dtype=np.float32)
        )
        game = replace(game, samples=(sample, *game.samples[1:]))
    elif corruption == "duplicate_action":
        sample = replace(
            game.samples[0], policy_indices=np.asarray([11, 11], dtype=np.int64)
        )
        game = replace(game, samples=(sample, *game.samples[1:]))
    elif corruption == "negative_plies":
        game = replace(game, plies=-1)
    else:
        game = replace(game, plies=len(game.samples) + 1)
    replay = ReplayBuffer(tmp_path, capacity_games=2)

    with pytest.raises(ValueError):
        replay.append_game(game)

    assert replay.game_ids == ()
    assert tuple((tmp_path / "games").glob("*.npz")) == ()
