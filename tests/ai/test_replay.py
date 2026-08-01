from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from ai.replay import ReplayBuffer, ReplayCompatibilityError
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


def test_replay_commits_complete_games_and_evicts_oldest(tmp_path: Path) -> None:
    replay = ReplayBuffer(tmp_path, capacity_games=2)

    assert replay.append_game(_game(1)) == 1
    assert replay.append_game(_game(2)) == 2
    assert replay.append_game(_game(3)) == 3

    reopened = ReplayBuffer(tmp_path, capacity_games=2)
    assert reopened.game_ids == (2, 3)
    assert not (tmp_path / "games" / "000000000001.npz").exists()


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
        manifest.read_text().replace('"schema_version": 1', '"schema_version": 99')
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
