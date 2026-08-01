from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from ai.checkpoint import CheckpointManager
from ai.config import TrainingConfig
from ai.control import RunControl
from ai.encoding import ACTION_SIZE, INPUT_CHANNELS
from ai.mcts import SearchState
from ai.network import PolicyValueNetwork
from ai.self_play import GameResult, TrainingSample
from ai.trainer import TorchEvaluator, Trainer, train_batch
from xiangqi.board import Board
from xiangqi.domain import Color


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
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

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
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
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
    Trainer(
        _config(tmp_path, target_games=1), game_factory=one_sample_game
    ).run()
    first = RunControl(tmp_path).read_status()
    RunControl(tmp_path).extend(2)

    resumed = Trainer(
        _config(tmp_path, target_games=3), game_factory=one_sample_game
    )
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
    restored = Trainer(
        _config(tmp_path, target_games=5), game_factory=one_sample_game
    )
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
    trainer = Trainer(
        _config(tmp_path, target_games=1), game_factory=one_sample_game
    )
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
    trainer = Trainer(
        _config(tmp_path, target_games=1), game_factory=one_sample_game
    )
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
