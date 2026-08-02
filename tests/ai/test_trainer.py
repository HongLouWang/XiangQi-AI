from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from ai.checkpoint import CheckpointManager, TrainingProgress
from ai.config import TrainingConfig
from ai.control import RunControl
from ai.encoding import ACTION_SIZE, INPUT_CHANNELS
from ai.mcts import SearchState
from ai.network import PolicyValueNetwork
from ai.replay import SCHEMA_VERSION, ReplayBuffer
from ai.self_play import GameResult, TrainingSample
from ai.trainer import (
    TorchEvaluator,
    Trainer,
    _devices_match,
    _parallel_game_candidates,
    train_batch,
)
from xiangqi.board import Board
from xiangqi.domain import Color
from xiangqi.rules import all_legal_moves


def _sample(value: float = 0.0) -> TrainingSample:
    return TrainingSample(
        state=np.zeros((INPUT_CHANNELS, 10, 9), dtype=np.float32),
        policy_indices=np.asarray([0, 1], dtype=np.int64),
        policy_probabilities=np.asarray([0.75, 0.25], dtype=np.float32),
        side=Color.RED,
        value=value,
    )


def one_sample_game(seed: int) -> GameResult:
    return GameResult((_sample(),), None, 1, f"seed-{seed}")


def pid_game(seed: int) -> GameResult:
    time.sleep(0.15)
    return GameResult((_sample(),), None, 1, str(os.getpid()))


def failing_game(seed: int) -> GameResult:
    raise RuntimeError(f"boom-{seed}")


def filesystem_flaky_game(seed: int) -> GameResult:
    marker = Path(os.environ["XIANGQI_RETRY_MARKERS"]) / str(seed)
    try:
        marker.touch(exist_ok=False)
    except FileExistsError:
        return one_sample_game(seed)
    raise RuntimeError(f"transient-{seed}")


def _legacy_v1_replay(run_dir: Path, game: GameResult) -> str:
    replay = ReplayBuffer(run_dir / "replay", capacity_games=10)
    replay.append_game(game)
    game_path = run_dir / "replay" / "games" / "000000000001.npz"
    with np.load(game_path, allow_pickle=False) as stored:
        payload = {key: stored[key] for key in stored.files}
    payload["schema_version"] = np.asarray([1], dtype=np.int64)
    with game_path.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    manifest_path = run_dir / "replay" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 1
    manifest.pop("total_games")
    manifest.pop("sample_counts")
    legacy = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()
    manifest_path.write_bytes(legacy)
    return hashlib.sha256(legacy).hexdigest()


def _config(tmp_path: Path, **changes: object) -> TrainingConfig:
    values: dict[str, object] = {
        "target_games": 2,
        "max_full_moves": 1,
        "device": "cpu",
        "torch_threads": 1,
        "self_play_workers": 1,
        "simulations_per_move": 1,
        "residual_blocks": 1,
        "channels": 2,
        "batch_size": 1,
        "replay_capacity_games": 10,
        "checkpoint_interval_games": 1,
        "seed": 23,
        "run_dir": tmp_path,
    }
    values.update(changes)
    return TrainingConfig(**values)  # type: ignore[arg-type]


def test_train_batch_expands_sparse_policy_and_really_updates_weights() -> None:
    torch.manual_seed(1)
    model = PolicyValueNetwork(channels=2, residual_blocks=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }

    policy_loss, value_loss = train_batch(
        model,
        optimizer,
        np.zeros((1, INPUT_CHANNELS, 10, 9), dtype=np.float32),
        (np.asarray([5, 11], dtype=np.int64),),
        (np.asarray([0.25, 0.75], dtype=np.float32),),
        np.asarray([1.0], dtype=np.float32),
        torch.device("cpu"),
    )

    assert np.isfinite(policy_loss)
    assert np.isfinite(value_loss)
    assert any(
        not torch.equal(before[name], tensor)
        for name, tensor in model.state_dict().items()
    )


def test_torch_evaluator_uses_eval_no_grad_and_returns_numpy() -> None:
    model = PolicyValueNetwork(channels=2, residual_blocks=1)
    evaluator = TorchEvaluator(model, torch.device("cpu"))

    logits, value = evaluator.evaluate(SearchState(Board.standard(), Color.RED))

    assert logits.shape == (ACTION_SIZE,)
    assert logits.dtype == np.float32
    assert isinstance(value, float)
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_torch_evaluator_batch_matches_individual_evaluations() -> None:
    model = PolicyValueNetwork(channels=2, residual_blocks=1)
    evaluator = TorchEvaluator(model, torch.device("cpu"))
    first = SearchState(Board.standard(), Color.RED)
    move = all_legal_moves(first.board, first.side)[0]
    states = (first, first.play(move))

    batch_logits, batch_values = evaluator.evaluate_many(states)
    individual = [evaluator.evaluate(state) for state in states]

    np.testing.assert_allclose(
        batch_logits,
        np.stack([item[0] for item in individual]),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        batch_values, [item[1] for item in individual], rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("was_training", [True, False])
def test_torch_evaluator_restores_original_model_mode(was_training: bool) -> None:
    model = PolicyValueNetwork(channels=2, residual_blocks=1)
    model.train(was_training)

    TorchEvaluator(model, torch.device("cpu")).evaluate(
        SearchState(Board.standard(), Color.RED)
    )

    assert model.training is was_training


def test_torch_evaluator_restores_mode_when_forward_fails() -> None:
    class BrokenModel(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            raise RuntimeError("inference failed")

    model = BrokenModel()
    model.train()

    with pytest.raises(RuntimeError, match="inference failed"):
        TorchEvaluator(model, torch.device("cpu")).evaluate(
            SearchState(Board.standard(), Color.RED)
        )

    assert model.training


def test_torch_evaluator_rejects_model_on_a_different_device() -> None:
    model = PolicyValueNetwork(channels=2, residual_blocks=1).to("meta")

    with pytest.raises(ValueError, match="device"):
        TorchEvaluator(model, torch.device("cpu"))


@pytest.mark.parametrize(
    ("requested", "actual", "expected"),
    [
        ("cuda", "cuda:0", True),
        ("cuda", "cuda:7", True),
        ("cuda:0", "cuda:0", True),
        ("cuda:1", "cuda:0", False),
        ("cuda:0", "cuda", False),
        ("cpu", "cpu", True),
        ("cpu", "meta", False),
        ("cuda", "cpu", False),
    ],
)
def test_device_matching_distinguishes_default_and_explicit_cuda_index(
    requested: str, actual: str, expected: bool
) -> None:
    assert _devices_match(torch.device(requested), torch.device(actual)) is expected


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("indices", np.asarray([], dtype=np.int64)),
        ("indices", np.asarray([[1]], dtype=np.int64)),
        ("indices", np.asarray([1.0], dtype=np.float32)),
        ("indices", np.asarray([-1], dtype=np.int64)),
        ("indices", np.asarray([ACTION_SIZE], dtype=np.int64)),
        ("indices", np.asarray([1, 1], dtype=np.int64)),
        ("probabilities", np.asarray([np.nan], dtype=np.float32)),
        ("probabilities", np.asarray([-0.1], dtype=np.float32)),
        ("probabilities", np.asarray([0.8], dtype=np.float32)),
        ("states", np.full((1, INPUT_CHANNELS, 10, 9), np.nan, dtype=np.float32)),
        ("values", np.asarray([np.nan], dtype=np.float32)),
        ("values", np.asarray([1.1], dtype=np.float32)),
    ],
)
def test_train_batch_rejects_invalid_targets_before_forward_or_update(
    field: str, replacement: np.ndarray
) -> None:
    class CountingModel(PolicyValueNetwork):
        calls = 0

        def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.calls += 1
            return super().forward(inputs)

    model = CountingModel(channels=2, residual_blocks=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    inputs: dict[str, object] = {
        "states": np.zeros((1, INPUT_CHANNELS, 10, 9), dtype=np.float32),
        "indices": np.asarray([1], dtype=np.int64),
        "probabilities": np.asarray([1.0], dtype=np.float32),
        "values": np.asarray([0.0], dtype=np.float32),
    }
    inputs[field] = replacement

    with pytest.raises(ValueError):
        train_batch(
            model,
            optimizer,
            inputs["states"],  # type: ignore[arg-type]
            (inputs["indices"],),  # type: ignore[arg-type]
            (inputs["probabilities"],),  # type: ignore[arg-type]
            inputs["values"],  # type: ignore[arg-type]
            torch.device("cpu"),
        )

    assert model.calls == 0
    assert all(
        torch.equal(before[name], tensor) for name, tensor in model.state_dict().items()
    )


def test_trainer_runs_target_and_persists_checkpoint_and_final_model(
    tmp_path: Path,
) -> None:
    trainer = Trainer(_config(tmp_path), game_factory=one_sample_game)

    trainer.run()

    status = RunControl(tmp_path).read_status()
    assert status.phase == "completed"
    assert status.completed_games == 2
    assert status.training_steps == 2
    assert CheckpointManager(tmp_path).has_checkpoint()
    final_path = tmp_path / "final_model.pt"
    assert final_path.is_file()
    assert torch.load(final_path, map_location="cpu", weights_only=True)


def test_resume_continues_progress_and_optimizer_instead_of_restarting(
    tmp_path: Path,
) -> None:
    Trainer(_config(tmp_path, target_games=1), game_factory=one_sample_game).run()
    first = RunControl(tmp_path).read_status()
    RunControl(tmp_path).extend(2)

    resumed = Trainer(_config(tmp_path, target_games=3), game_factory=one_sample_game)
    resumed.run(resume=True)

    status = RunControl(tmp_path).read_status()
    assert first.phase == "completed"
    assert first.completed_games == 1
    assert first.target_games == 1
    assert first.training_steps == 1
    assert first.device == "cpu"
    assert status.completed_games == 3
    assert status.training_steps == 3
    assert resumed.progress.completed_games == 3


@pytest.mark.parametrize("forward_games", [1, 3])
def test_restore_moves_forward_over_games_committed_after_checkpoint(
    tmp_path: Path, forward_games: int
) -> None:
    config = _config(tmp_path, target_games=1, replay_capacity_games=10)
    original = Trainer(config, game_factory=one_sample_game)
    original.run()
    checkpoint_model = {
        name: tensor.detach().clone()
        for name, tensor in original.model.state_dict().items()
    }
    for offset in range(forward_games):
        original.replay.append_game(one_sample_game(100 + offset))

    resumed = Trainer(
        _config(tmp_path, target_games=forward_games + 2, replay_capacity_games=10),
        game_factory=one_sample_game,
    )
    resumed.restore()

    assert resumed.progress.completed_games == forward_games + 1
    assert all(
        torch.equal(checkpoint_model[name], tensor)
        for name, tensor in resumed.model.state_dict().items()
    )
    CheckpointManager(tmp_path).load_latest(
        resumed.model,
        resumed.optimizer,
        expected_config=resumed.config,
        expected_replay_manifest_hash=resumed.replay.manifest_hash,
        expected_replay_manifest_version=resumed.replay.manifest_version,
        numpy_generator=resumed.rng,
    )


def test_restore_uses_total_game_cursor_after_replay_capacity_eviction(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, target_games=1, replay_capacity_games=1)
    original = Trainer(config, game_factory=one_sample_game)
    original.run()
    for seed in (101, 102, 103):
        original.replay.append_game(one_sample_game(seed))

    resumed = Trainer(
        _config(tmp_path, target_games=5, replay_capacity_games=1),
        game_factory=one_sample_game,
    )
    resumed.restore()

    assert resumed.replay.game_ids == (4,)
    assert resumed.progress.completed_games == 4


def test_restore_rejects_replay_behind_checkpoint(tmp_path: Path) -> None:
    trainer = Trainer(_config(tmp_path, target_games=2), game_factory=one_sample_game)
    trainer.run()
    manifest_path = tmp_path / "replay" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(games=[1], next_game_id=2, total_games=1, sample_counts={"1": 1})
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "replay" / "games" / "000000000002.npz").unlink()

    with pytest.raises(RuntimeError, match="落后|behind"):
        Trainer(
            _config(tmp_path, target_games=3), game_factory=one_sample_game
        ).restore()


def test_forward_restore_generates_the_next_seed_without_repeating_games(
    tmp_path: Path,
) -> None:
    first = Trainer(_config(tmp_path, target_games=1), game_factory=one_sample_game)
    first.run()
    first.replay.append_game(one_sample_game(999))
    seeds: list[int] = []

    def capture(seed: int) -> GameResult:
        seeds.append(seed)
        return one_sample_game(seed)

    Trainer(_config(tmp_path, target_games=3), game_factory=capture).run(resume=True)

    assert seeds == [26]


def test_restore_once_accepts_exact_v1_predecessor_and_rewrites_checkpoint(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, target_games=2)
    legacy_hash = _legacy_v1_replay(tmp_path, one_sample_game(24))
    model = PolicyValueNetwork(channels=2, residual_blocks=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    expected_model = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    CheckpointManager(tmp_path).save(
        model,
        optimizer,
        TrainingProgress(1, 2, 0),
        config,
        replay_manifest_hash=legacy_hash,
        replay_manifest_version=1,
        numpy_generator=np.random.default_rng(config.seed),
    )

    resumed = Trainer(config, game_factory=one_sample_game)
    assert resumed.replay.legacy_manifest_hash == legacy_hash
    resumed.restore()

    assert resumed.progress.completed_games == 1
    assert all(
        torch.equal(expected_model[name], tensor)
        for name, tensor in resumed.model.state_dict().items()
    )
    loaded = CheckpointManager(tmp_path).load_latest(
        resumed.model,
        resumed.optimizer,
        expected_replay_manifest_hash=resumed.replay.manifest_hash,
        expected_replay_manifest_version=resumed.replay.manifest_version,
        numpy_generator=resumed.rng,
    )
    assert loaded.replay_manifest_hash == resumed.replay.manifest_hash
    assert loaded.replay_manifest_version == SCHEMA_VERSION
    assert resumed.replay.legacy_manifest_hash is None
    assert not resumed.replay.migration_path.exists()


def test_v1_predecessor_does_not_allow_an_arbitrary_checkpoint_hash(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, target_games=2)
    _legacy_v1_replay(tmp_path, one_sample_game(24))
    model = PolicyValueNetwork(channels=2, residual_blocks=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    CheckpointManager(tmp_path).save(
        model,
        optimizer,
        TrainingProgress(1, 2, 0),
        config,
        replay_manifest_hash="f" * 64,
        replay_manifest_version=1,
        numpy_generator=np.random.default_rng(config.seed),
    )

    with pytest.raises(RuntimeError, match="hash"):
        Trainer(config, game_factory=one_sample_game).restore()


def test_strict_v2_checkpoint_cleans_leftover_migration_sidecar(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, target_games=2)
    _legacy_v1_replay(tmp_path, one_sample_game(24))
    replay = ReplayBuffer(tmp_path / "replay", capacity_games=10)
    model = PolicyValueNetwork(channels=2, residual_blocks=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    CheckpointManager(tmp_path).save(
        model,
        optimizer,
        TrainingProgress(1, 2, 0),
        config,
        replay_manifest_hash=replay.manifest_hash,
        replay_manifest_version=replay.manifest_version,
        numpy_generator=np.random.default_rng(config.seed),
    )

    resumed = Trainer(config, game_factory=one_sample_game)
    resumed.restore()

    assert not resumed.replay.migration_path.exists()


def test_sidecar_delete_failure_keeps_retryable_state_after_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, target_games=2)
    legacy_hash = _legacy_v1_replay(tmp_path, one_sample_game(24))
    model = PolicyValueNetwork(channels=2, residual_blocks=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    CheckpointManager(tmp_path).save(
        model,
        optimizer,
        TrainingProgress(1, 2, 0),
        config,
        replay_manifest_hash=legacy_hash,
        replay_manifest_version=1,
        numpy_generator=np.random.default_rng(config.seed),
    )
    resumed = Trainer(config, game_factory=one_sample_game)
    monkeypatch.setattr(
        resumed.replay,
        "clear_migration",
        lambda: (_ for _ in ()).throw(OSError("delete failed")),
    )

    with pytest.raises(OSError, match="delete failed"):
        resumed.restore()

    assert resumed.replay.migration_path.is_file()
    CheckpointManager(tmp_path).load_latest(
        resumed.model,
        resumed.optimizer,
        expected_replay_manifest_hash=resumed.replay.manifest_hash,
        expected_replay_manifest_version=SCHEMA_VERSION,
        numpy_generator=resumed.rng,
    )
    retry = Trainer(config, game_factory=one_sample_game)
    retry.restore()
    assert not retry.replay.migration_path.exists()


def test_removed_sidecar_never_reauthorizes_legacy_checkpoint(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, target_games=2)
    legacy_hash = _legacy_v1_replay(tmp_path, one_sample_game(24))
    model = PolicyValueNetwork(channels=2, residual_blocks=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    manager = CheckpointManager(tmp_path)
    manager.save(
        model,
        optimizer,
        TrainingProgress(1, 2, 0),
        config,
        replay_manifest_hash=legacy_hash,
        replay_manifest_version=1,
        numpy_generator=np.random.default_rng(config.seed),
    )
    Trainer(config, game_factory=one_sample_game).restore()
    manager.save(
        model,
        optimizer,
        TrainingProgress(1, 2, 0),
        config,
        replay_manifest_hash=legacy_hash,
        replay_manifest_version=1,
        numpy_generator=np.random.default_rng(config.seed),
    )

    with pytest.raises(RuntimeError, match="hash"):
        Trainer(config, game_factory=one_sample_game).restore()


def test_pause_is_safe_and_checkpoint_can_be_loaded(tmp_path: Path) -> None:
    def pause_after_complete_game(seed: int) -> GameResult:
        result = one_sample_game(seed)
        RunControl(tmp_path).request_pause()
        return result

    trainer = Trainer(
        _config(tmp_path, target_games=5), game_factory=pause_after_complete_game
    )
    trainer.run()

    status = RunControl(tmp_path).read_status()
    assert status.phase == "paused"
    assert status.completed_games == 1
    restored = Trainer(_config(tmp_path, target_games=5), game_factory=one_sample_game)
    restored.restore()
    assert restored.progress == trainer.progress


def test_extend_during_run_changes_the_live_cumulative_target(tmp_path: Path) -> None:
    calls = 0

    def extend_once(seed: int) -> GameResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            RunControl(tmp_path).extend(2)
        return one_sample_game(seed)

    Trainer(_config(tmp_path, target_games=1), game_factory=extend_once).run()

    status = RunControl(tmp_path).read_status()
    assert status.phase == "completed"
    assert status.completed_games == 3
    assert status.target_games == 3


def test_two_cpu_workers_are_real_spawned_processes(tmp_path: Path) -> None:
    trainer = Trainer(
        _config(tmp_path, target_games=4, self_play_workers=2),
        game_factory=pid_game,
    )

    trainer.run()

    assert trainer.worker_count == 2
    assert trainer.worker_pids.isdisjoint({os.getpid()})
    assert len(trainer.worker_pids) == 2


def test_cuda_mode_never_starts_cpu_self_play_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ai.trainer.configure_device", lambda *_: torch.device("cpu"))
    trainer = Trainer(
        _config(tmp_path, device="cuda", self_play_workers=4, target_games=1),
        game_factory=one_sample_game,
    )

    trainer.run()

    assert trainer.worker_count == 1
    assert trainer.worker_pids == {os.getpid()}
    assert "self_play_workers_effective=1" in RunControl(tmp_path).read_status().message


def test_cuda_mode_uses_requested_parallel_game_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: list[tuple[int, int]] = []

    class FakeBatchedSelfPlay:
        last_batch_size = 0

        def __init__(self, *args, **kwargs) -> None:
            pass

        def generate(self, *, count: int, parallel_games: int):
            requested.append((count, parallel_games))
            self.last_batch_size = count
            return [one_sample_game(index) for index in range(count)]

    monkeypatch.setattr("ai.trainer.configure_device", lambda *_: torch.device("cpu"))
    monkeypatch.setattr("ai.trainer.BatchedSelfPlay", FakeBatchedSelfPlay)
    trainer = Trainer(
        _config(tmp_path, device="cuda", parallel_games=4, target_games=3)
    )

    trainer.run()

    assert requested == [(3, 4)]
    status = RunControl(tmp_path).read_status()
    assert "parallel_games_requested=4" in status.message
    assert "parallel_games_effective=4" in status.message


def test_parallel_game_candidates_halve_to_one() -> None:
    assert _parallel_game_candidates(16) == (16, 8, 4, 2, 1)
    assert _parallel_game_candidates(10) == (10, 5, 2, 1)


def test_cuda_oom_retries_batch_at_lower_parallelism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[int] = []

    class OomOnceBatchedSelfPlay:
        last_batch_size = 1

        def __init__(self, *args, **kwargs) -> None:
            pass

        def generate(self, *, count: int, parallel_games: int):
            attempts.append(parallel_games)
            if len(attempts) == 1:
                raise torch.OutOfMemoryError("CUDA out of memory")
            return [one_sample_game(index) for index in range(count)]

    monkeypatch.setattr("ai.trainer.configure_device", lambda *_: torch.device("cpu"))
    monkeypatch.setattr("ai.trainer.BatchedSelfPlay", OomOnceBatchedSelfPlay)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    trainer = Trainer(
        _config(tmp_path, device="cuda", parallel_games=4, target_games=2)
    )

    trainer.run()

    assert attempts == [4, 2]
    assert trainer.parallel_games_effective == 2
    assert trainer.oom_downgrades == 1
    assert trainer.replay.total_games == 2


def test_small_replay_commits_progress_before_training_is_possible(
    tmp_path: Path,
) -> None:
    trainer = Trainer(
        _config(tmp_path, target_games=1, batch_size=2),
        game_factory=one_sample_game,
    )

    trainer.run()

    status = RunControl(tmp_path).read_status()
    assert status.completed_games == 1
    assert status.training_steps == 0


def test_extend_during_finalization_keeps_training(tmp_path: Path) -> None:
    trainer = Trainer(_config(tmp_path, target_games=1), game_factory=one_sample_game)
    original_export = trainer.checkpoints.export_model
    extended = False

    def export_and_extend(model: torch.nn.Module, destination: Path) -> Path:
        nonlocal extended
        result = original_export(model, destination)
        if not extended:
            extended = True
            RunControl(tmp_path).extend(1)
        return result

    trainer.checkpoints.export_model = export_and_extend  # type: ignore[method-assign]

    trainer.run()

    status = RunControl(tmp_path).read_status()
    assert status.phase == "completed"
    assert status.completed_games == 2
    assert status.target_games == 2


def test_extend_immediately_before_completion_handshake_keeps_training(
    tmp_path: Path,
) -> None:
    trainer = Trainer(_config(tmp_path, target_games=1), game_factory=one_sample_game)
    original_handshake = trainer.control.try_mark_completed
    extended = False

    def extend_then_handshake(*args: object, **kwargs: object) -> bool:
        nonlocal extended
        if not extended:
            extended = True
            RunControl(tmp_path).extend(1)
        return original_handshake(*args, **kwargs)

    trainer.control.try_mark_completed = extend_then_handshake  # type: ignore[method-assign]

    trainer.run()

    status = RunControl(tmp_path).read_status()
    assert status.phase == "completed"
    assert status.completed_games == 2


def test_checkpoint_failure_does_not_prevent_failed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = Trainer(_config(tmp_path), game_factory=failing_game)
    monkeypatch.setattr(
        trainer, "_save_checkpoint", lambda: (_ for _ in ()).throw(OSError("disk"))
    )

    with pytest.raises(RuntimeError, match="boom"):
        trainer.run()

    assert RunControl(tmp_path).read_status().phase == "failed"


def test_game_failure_saves_failed_checkpoint_without_committing_half_game(
    tmp_path: Path,
) -> None:
    trainer = Trainer(_config(tmp_path), game_factory=failing_game)

    with pytest.raises(RuntimeError, match="boom"):
        trainer.run()

    status = RunControl(tmp_path).read_status()
    assert status.phase == "failed"
    assert status.completed_games == 0
    assert CheckpointManager(tmp_path).has_checkpoint()
    assert not list((tmp_path / "replay" / "games").glob("*.npz"))


def test_sequential_worker_retries_with_the_same_seed_then_succeeds(
    tmp_path: Path,
) -> None:
    attempts: list[int] = []

    def flaky(seed: int) -> GameResult:
        attempts.append(seed)
        if len(attempts) < 3:
            raise RuntimeError("temporary")
        return one_sample_game(seed)

    trainer = Trainer(
        _config(tmp_path, target_games=1, game_retry_limit=2), game_factory=flaky
    )
    trainer.run()

    assert attempts == [24, 24, 24]
    assert trainer.progress.completed_games == 1
    assert trainer.replay.total_games == 1


def test_spawn_workers_retry_transient_failures_and_preserve_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    markers = tmp_path / "markers"
    markers.mkdir()
    monkeypatch.setenv("XIANGQI_RETRY_MARKERS", str(markers))
    trainer = Trainer(
        _config(
            tmp_path / "run",
            target_games=2,
            self_play_workers=2,
            game_retry_limit=1,
        ),
        game_factory=filesystem_flaky_game,
    )

    trainer.run()

    assert {path.name for path in markers.iterdir()} == {"24", "25"}
    assert trainer.replay.total_games == 2


def test_retry_exhaustion_records_game_traceback_and_no_partial_replay(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = Trainer(
        _config(tmp_path, target_games=1, game_retry_limit=1),
        game_factory=failing_game,
    )

    with pytest.raises(RuntimeError, match=r"game 1.*seed 24.*attempt 2"):
        trainer.run()

    status = RunControl(tmp_path).read_status()
    assert status.phase == "failed"
    assert "game 1" in status.message
    assert "Traceback" in status.message
    assert "game 1" in caplog.text
    assert "Traceback" in caplog.text
    assert CheckpointManager(tmp_path).has_checkpoint()
    assert trainer.replay.total_games == 0
