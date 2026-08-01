from __future__ import annotations

from pathlib import Path

import torch

from ai.checkpoint import CheckpointManager
from ai.config import TrainingConfig
from ai.network import PolicyValueNetwork
from ai.replay import ReplayBuffer
from ai.trainer import Trainer


def test_real_cpu_training_produces_reloadable_persistent_state(
    tmp_path: Path,
) -> None:
    config = TrainingConfig(
        target_games=1,
        max_full_moves=1,
        simulations_per_move=1,
        channels=8,
        residual_blocks=1,
        batch_size=1,
        checkpoint_interval_games=1,
        run_dir=tmp_path,
        device="cpu",
        torch_threads=1,
        self_play_workers=1,
        seed=101,
    )

    trainer = Trainer(config)
    trainer.run()

    assert trainer.progress.completed_games == 1
    assert trainer.progress.training_steps == 1
    assert trainer.replay.game_count == 1
    assert trainer.replay.sample_count >= 1

    manager = CheckpointManager(tmp_path)
    assert manager.has_checkpoint()
    assert manager.latest_path.is_file()
    assert (tmp_path / "replay" / "manifest.json").is_file()

    restored = Trainer(config)
    restored.restore()
    assert restored.progress == trainer.progress
    assert restored.replay.game_count == trainer.replay.game_count
    for name, parameter in trainer.model.state_dict().items():
        assert torch.equal(restored.model.state_dict()[name], parameter)

    exported_state = torch.load(
        tmp_path / "final_model.pt", map_location="cpu", weights_only=True
    )
    standalone_model = PolicyValueNetwork(channels=8, residual_blocks=1)
    standalone_model.load_state_dict(exported_state, strict=True)

    reopened_replay = ReplayBuffer(
        tmp_path / "replay", capacity_games=config.replay_capacity_games
    )
    assert reopened_replay.game_count == 1
    assert reopened_replay.sample_count == trainer.replay.sample_count
