from __future__ import annotations

import os
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from ai.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointManager,
    TrainingProgress,
)
from ai.config import TrainingConfig
from ai.network import PolicyValueNetwork

_REPLAY_HASH = "a" * 64


def _save(manager: CheckpointManager, *args: object, **kwargs: object) -> Path:
    kwargs.setdefault("replay_manifest_hash", _REPLAY_HASH)
    kwargs.setdefault("replay_manifest_version", 1)
    return CheckpointManager.save(manager, *args, **kwargs)


def _objects() -> tuple[PolicyValueNetwork, torch.optim.Adam]:
    model = PolicyValueNetwork(channels=4, residual_blocks=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    inputs = torch.randn(2, 15, 10, 9)
    policy, value = model(inputs)
    (policy.mean() + value.mean()).backward()
    optimizer.step()
    return model, optimizer


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def test_checkpoint_persists_model_optimizer_progress_config_and_rng(
    tmp_path: Path,
) -> None:
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    model, optimizer = _objects()
    expected_model = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    expected_optimizer = optimizer.state_dict()
    config = TrainingConfig(
        target_games=10, channels=4, residual_blocks=1, run_dir=tmp_path
    )
    progress = TrainingProgress(7, 10, 3)
    manager = CheckpointManager(tmp_path)
    _save(
        manager,
        model,
        optimizer,
        progress,
        config,
        replay_manifest_hash="abc123",
        replay_manifest_version=1,
    )
    expected_random = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1)

    del model, optimizer
    random.seed(90)
    np.random.seed(91)
    torch.manual_seed(92)
    restored_model, restored_optimizer = _objects()
    restored = manager.load_latest(
        restored_model, restored_optimizer, map_location="cpu"
    )

    assert restored.progress == progress
    assert restored.config == config
    assert restored.replay_manifest_hash == "abc123"
    assert restored.replay_manifest_version == 1
    _assert_nested_equal(expected_model, restored_model.state_dict())
    _assert_nested_equal(expected_optimizer, restored_optimizer.state_dict())
    assert random.random() == expected_random
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)


def test_checkpoint_restores_the_actual_numpy_generator(tmp_path: Path) -> None:
    model, optimizer = _objects()
    config = TrainingConfig(channels=4, residual_blocks=1, run_dir=tmp_path)
    generator = np.random.default_rng(1234)
    generator.random(3)
    manager = CheckpointManager(tmp_path)
    _save(
        manager,
        model,
        optimizer,
        TrainingProgress(1, 10_000, 1),
        config,
        numpy_generator=generator,
    )
    expected = generator.random(4)
    generator.random(20)

    fresh_model, fresh_optimizer = _objects()
    manager.load_latest(
        fresh_model,
        fresh_optimizer,
        numpy_generator=generator,
        map_location="cpu",
    )
    assert np.array_equal(generator.random(4), expected)


def test_corrupt_latest_checkpoint_falls_back_to_previous_slot(tmp_path: Path) -> None:
    model, optimizer = _objects()
    config = TrainingConfig(channels=4, residual_blocks=1, run_dir=tmp_path)
    manager = CheckpointManager(tmp_path)
    _save(manager, model, optimizer, TrainingProgress(1, 10_000, 1), config)
    _save(manager, model, optimizer, TrainingProgress(2, 10_000, 2), config)

    manager.latest_path.write_bytes(b"broken")

    fresh_model, fresh_optimizer = _objects()
    restored = manager.load_latest(fresh_model, fresh_optimizer, map_location="cpu")
    assert restored.progress.completed_games == 1


def test_corrupt_latest_pointer_recovers_highest_valid_generation(
    tmp_path: Path,
) -> None:
    model, optimizer = _objects()
    config = TrainingConfig(channels=4, residual_blocks=1, run_dir=tmp_path)
    manager = CheckpointManager(tmp_path)
    _save(manager, model, optimizer, TrainingProgress(1, 10_000, 1), config)
    _save(manager, model, optimizer, TrainingProgress(2, 10_000, 2), config)
    manager.pointer_path.write_bytes(b"broken")

    fresh_model, fresh_optimizer = _objects()
    restored = manager.load_latest(fresh_model, fresh_optimizer, map_location="cpu")
    assert restored.progress.completed_games == 2


def test_pointer_generation_mismatch_recovers_highest_valid_slot(
    tmp_path: Path,
) -> None:
    model, optimizer = _objects()
    config = TrainingConfig(channels=4, residual_blocks=1, run_dir=tmp_path)
    manager = CheckpointManager(tmp_path)
    _save(manager, model, optimizer, TrainingProgress(1, 10_000, 1), config)
    _save(manager, model, optimizer, TrainingProgress(2, 10_000, 2), config)
    manager.pointer_path.write_text('{"slot": "a", "generation": 999}')

    fresh_model, fresh_optimizer = _objects()
    restored = manager.load_latest(fresh_model, fresh_optimizer, map_location="cpu")
    assert restored.progress.completed_games == 2


def test_missing_required_checkpoint_field_is_explicitly_rejected(
    tmp_path: Path,
) -> None:
    model, optimizer = _objects()
    config = TrainingConfig(channels=4, residual_blocks=1, run_dir=tmp_path)
    manager = CheckpointManager(tmp_path)
    path = _save(manager, model, optimizer, TrainingProgress(1, 10_000, 1), config)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["replay_manifest_version"]
    torch.save(payload, path)
    fresh_model, fresh_optimizer = _objects()

    with pytest.raises(CheckpointCompatibilityError, match="replay_manifest_version"):
        manager.load_latest(fresh_model, fresh_optimizer, map_location="cpu")


def test_incompatible_latest_slot_is_explicitly_rejected(
    tmp_path: Path,
) -> None:
    model, optimizer = _objects()
    config = TrainingConfig(channels=4, residual_blocks=1, run_dir=tmp_path)
    manager = CheckpointManager(tmp_path)
    _save(manager, model, optimizer, TrainingProgress(1, 10_000, 1), config)
    newest = _save(manager, model, optimizer, TrainingProgress(2, 10_000, 2), config)
    payload = torch.load(newest, map_location="cpu", weights_only=False)
    payload["schema_version"] = 99
    torch.save(payload, newest)

    fresh_model, fresh_optimizer = _objects()
    with pytest.raises(CheckpointCompatibilityError, match="schema_version"):
        manager.load_latest(fresh_model, fresh_optimizer, map_location="cpu")


def test_replay_manifest_mismatch_is_explicitly_rejected(tmp_path: Path) -> None:
    model, optimizer = _objects()
    config = TrainingConfig(channels=4, residual_blocks=1, run_dir=tmp_path)
    manager = CheckpointManager(tmp_path)
    _save(manager, model, optimizer, TrainingProgress(1, 10_000, 1), config)
    fresh_model, fresh_optimizer = _objects()

    with pytest.raises(CheckpointCompatibilityError, match="Replay manifest hash"):
        manager.load_latest(
            fresh_model,
            fresh_optimizer,
            expected_replay_manifest_hash="b" * 64,
            expected_replay_manifest_version=1,
        )


def test_scheduler_state_round_trips_from_disk(tmp_path: Path) -> None:
    model, optimizer = _objects()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    optimizer.step()
    scheduler.step()
    config = TrainingConfig(channels=4, residual_blocks=1, run_dir=tmp_path)
    manager = CheckpointManager(tmp_path)
    _save(
        manager,
        model,
        optimizer,
        TrainingProgress(1, 10_000, 1),
        config,
        scheduler=scheduler,
    )

    fresh_model, fresh_optimizer = _objects()
    fresh_scheduler = torch.optim.lr_scheduler.StepLR(
        fresh_optimizer, step_size=1, gamma=0.5
    )
    manager.load_latest(
        fresh_model,
        fresh_optimizer,
        scheduler=fresh_scheduler,
        map_location="cpu",
    )
    assert fresh_scheduler.state_dict() == scheduler.state_dict()


def test_failed_new_slot_write_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, optimizer = _objects()
    config = TrainingConfig(channels=4, residual_blocks=1, run_dir=tmp_path)
    manager = CheckpointManager(tmp_path)
    _save(manager, model, optimizer, TrainingProgress(1, 10_000, 1), config)
    real_replace = os.replace

    def fail_checkpoint_replace(source: object, destination: object) -> None:
        if Path(destination).name == "checkpoint-b.pt":
            raise OSError("checkpoint replace failed")
        real_replace(source, destination)

    monkeypatch.setattr("ai.checkpoint.os.replace", fail_checkpoint_replace)
    with pytest.raises(OSError, match="checkpoint replace failed"):
        _save(manager, model, optimizer, TrainingProgress(2, 10_000, 2), config)
    fresh_model, fresh_optimizer = _objects()
    restored = manager.load_latest(fresh_model, fresh_optimizer, map_location="cpu")
    assert restored.progress.completed_games == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime unavailable")
def test_cuda_checkpoint_can_load_into_fresh_cpu_model(tmp_path: Path) -> None:
    model, optimizer = _objects()
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    config = TrainingConfig(channels=4, residual_blocks=1, run_dir=tmp_path)
    manager = CheckpointManager(tmp_path)
    _save(manager, model, optimizer, TrainingProgress(1, 10_000, 1), config)

    cpu_model, cpu_optimizer = _objects()
    manager.load_latest(cpu_model, cpu_optimizer, map_location="cpu")
    assert all(parameter.device.type == "cpu" for parameter in cpu_model.parameters())


def test_incompatible_network_configuration_is_explicitly_rejected(
    tmp_path: Path,
) -> None:
    model, optimizer = _objects()
    config = TrainingConfig(channels=4, residual_blocks=1, run_dir=tmp_path)
    manager = CheckpointManager(tmp_path)
    _save(manager, model, optimizer, TrainingProgress(1, 10_000, 1), config)
    incompatible = replace(config, channels=8)
    fresh_model, fresh_optimizer = _objects()

    with pytest.raises(CheckpointCompatibilityError, match="channels"):
        manager.load_latest(
            fresh_model,
            fresh_optimizer,
            expected_config=incompatible,
            map_location="cpu",
        )


def test_model_shape_mismatch_is_explicitly_rejected(tmp_path: Path) -> None:
    model, optimizer = _objects()
    config = TrainingConfig(channels=4, residual_blocks=1, run_dir=tmp_path)
    manager = CheckpointManager(tmp_path)
    _save(manager, model, optimizer, TrainingProgress(1, 10_000, 1), config)
    wrong_model = PolicyValueNetwork(channels=8, residual_blocks=1)
    wrong_optimizer = torch.optim.Adam(wrong_model.parameters())

    with pytest.raises(CheckpointCompatibilityError, match="model_state"):
        manager.load_latest(wrong_model, wrong_optimizer, map_location="cpu")


def test_exported_model_weights_load_into_a_fresh_cpu_model(tmp_path: Path) -> None:
    model, _optimizer = _objects()
    expected = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    manager = CheckpointManager(tmp_path)
    output = manager.export_model(model, tmp_path / "xiangqi-model.pt")

    fresh = PolicyValueNetwork(channels=4, residual_blocks=1)
    fresh.load_state_dict(torch.load(output, map_location="cpu", weights_only=True))
    _assert_nested_equal(expected, fresh.state_dict())
