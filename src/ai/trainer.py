from __future__ import annotations

import multiprocessing
import os
import random
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from ai.checkpoint import CheckpointManager, TrainingProgress
from ai.config import TrainingConfig
from ai.control import RunControl, RunStatus
from ai.encoding import ACTION_SIZE, encode_board
from ai.mcts import MCTS, SearchState
from ai.network import PolicyValueNetwork, configure_device
from ai.replay import ReplayBatch, ReplayBuffer
from ai.self_play import GameResult, play_game

GameFactory = Callable[[int], GameResult]


class TorchEvaluator:
    """在指定设备执行只读 Policy/Value 推理。"""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    def evaluate(
        self, state: SearchState
    ) -> tuple[NDArray[np.float32], float]:
        self.model.eval()
        inputs = torch.from_numpy(encode_board(state.board, state.side)).unsqueeze(0)
        with torch.no_grad():
            logits, values = self.model(inputs.to(self.device))
        policy = logits[0].detach().to("cpu", dtype=torch.float32).numpy().copy()
        return policy, float(values.reshape(-1)[0].detach().cpu())


def train_batch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    states: NDArray[np.float32] | torch.Tensor,
    policy_indices: tuple[NDArray[np.int64], ...],
    policy_probabilities: tuple[NDArray[np.float32], ...],
    value_targets: NDArray[np.float32] | torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """展开 Replay 的稀疏策略目标并完成一次真实梯度更新。"""
    state_tensor = torch.as_tensor(states, dtype=torch.float32)
    value_tensor = torch.as_tensor(value_targets, dtype=torch.float32)
    if state_tensor.ndim != 4:
        raise ValueError("states 必须是四维 batch")
    batch_size = int(state_tensor.shape[0])
    if len(policy_indices) != batch_size or len(policy_probabilities) != batch_size:
        raise ValueError("稀疏策略数量必须与 batch 大小一致")
    dense_policy = torch.zeros((batch_size, ACTION_SIZE), dtype=torch.float32)
    for row, (indices, probabilities) in enumerate(
        zip(policy_indices, policy_probabilities, strict=True)
    ):
        index_tensor = torch.as_tensor(indices, dtype=torch.int64)
        probability_tensor = torch.as_tensor(probabilities, dtype=torch.float32)
        if index_tensor.ndim != 1 or probability_tensor.shape != index_tensor.shape:
            raise ValueError("稀疏策略索引与概率必须是一维等长数组")
        dense_policy[row, index_tensor] = probability_tensor

    model.train()
    logits, values = model(state_tensor.to(device))
    policy_targets = dense_policy.to(device)
    values_target = value_tensor.to(device)
    policy_loss = -(
        policy_targets * torch.log_softmax(logits, dim=1)
    ).sum(dim=1).mean()
    value_loss = torch.nn.functional.mse_loss(
        values.reshape(-1), values_target.reshape(-1)
    )
    loss = policy_loss + value_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(policy_loss.detach().cpu()), float(value_loss.detach().cpu())


@dataclass(frozen=True, slots=True)
class _WorkerJob:
    config: TrainingConfig
    state_dict: dict[str, torch.Tensor]
    seed: int
    game_factory: GameFactory | None


@dataclass(frozen=True, slots=True)
class _WorkerResult:
    pid: int
    game: GameResult


def _production_game(
    config: TrainingConfig,
    state_dict: dict[str, torch.Tensor],
    seed: int,
) -> GameResult:
    torch.set_num_threads(config.torch_threads)
    model = PolicyValueNetwork(
        channels=config.channels, residual_blocks=config.residual_blocks
    )
    model.load_state_dict(state_dict)
    evaluator = TorchEvaluator(model, torch.device("cpu"))
    search = MCTS(
        evaluator,
        simulations=config.simulations_per_move,
        c_puct=1.5,
        seed=seed,
    )
    return play_game(search, max_plies=config.max_plies, seed=seed)


def _worker_entry(job: _WorkerJob) -> _WorkerResult:
    random.seed(job.seed)
    np.random.seed(job.seed % (2**32))
    torch.manual_seed(job.seed)
    game = (
        job.game_factory(job.seed)
        if job.game_factory is not None
        else _production_game(job.config, job.state_dict, job.seed)
    )
    return _WorkerResult(os.getpid(), game)


class Trainer:
    """自我对弈、Replay、优化与持久化的唯一主进程协调器。"""

    def __init__(
        self,
        config: TrainingConfig,
        *,
        game_factory: GameFactory | None = None,
        model: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> None:
        self.config = config
        self.device = configure_device(config.device, config.torch_threads)
        self.model = model or PolicyValueNetwork(
            channels=config.channels, residual_blocks=config.residual_blocks
        )
        self.model.to(self.device)
        self.optimizer = optimizer or torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.scheduler = scheduler
        self.game_factory = game_factory
        self.replay = ReplayBuffer(
            config.run_dir / "replay", capacity_games=config.replay_capacity_games
        )
        self.checkpoints = CheckpointManager(config.run_dir)
        self.control = RunControl(config.run_dir)
        self.rng = np.random.default_rng(config.seed)
        self.progress = TrainingProgress(0, config.target_games, 0)
        cuda_mode = self.device.type == "cuda" or config.device.startswith("cuda")
        self.worker_count = 1 if cuda_mode else config.self_play_workers
        self.worker_pids: set[int] = set()

    def restore(self) -> None:
        loaded = self.checkpoints.load_latest(
            self.model,
            self.optimizer,
            expected_config=self.config,
            scheduler=self.scheduler,
            map_location=self.device,
            numpy_generator=self.rng,
            expected_replay_manifest_hash=self.replay.manifest_hash,
            expected_replay_manifest_version=self.replay.manifest_version,
        )
        target = max(loaded.progress.target_games, self.config.target_games)
        if self.control.status_path.exists():
            target = max(target, self.control.read_status().target_games)
        self.progress = TrainingProgress(
            loaded.progress.completed_games,
            target,
            loaded.progress.training_steps,
        )

    def run(self, *, resume: bool = False) -> None:
        pool: multiprocessing.pool.Pool | None = None
        try:
            if resume:
                self.restore()
                self.control.clear_pause()
            self._write_status("running")
            if self.worker_count > 1:
                pool = multiprocessing.get_context("spawn").Pool(self.worker_count)

            while True:
                while self.progress.completed_games < self._live_target():
                    remaining = self._live_target() - self.progress.completed_games
                    count = min(self.worker_count, remaining)
                    results = self._generate_games(count, pool)
                    for result in results:
                        self.worker_pids.add(result.pid)
                        self._commit_game(result.game)
                        pause = self.control.pause_requested()
                        periodic = (
                            self.progress.completed_games
                            % self.config.checkpoint_interval_games
                            == 0
                        )
                        if pause or periodic:
                            self._save_checkpoint()
                        if pause:
                            self.control.mark_paused(self.progress)
                            return
                        if self.progress.completed_games >= self._live_target():
                            break

                self._save_checkpoint()
                self.checkpoints.export_model(
                    self.model, Path(self.config.run_dir) / "final_model.pt"
                )
                self._write_status("completed")
                if self.control.read_status().phase == "completed":
                    return
                self._live_target()
                self._write_status("running")
        except Exception as error:
            with suppress(Exception):
                self._write_status("failed", message=str(error))
            with suppress(Exception):
                self._save_checkpoint()
            raise
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    def _generate_games(
        self, count: int, pool: multiprocessing.pool.Pool | None
    ) -> list[_WorkerResult]:
        seeds = [self.config.seed + self.progress.completed_games + i + 1 for i in range(count)]
        if pool is None:
            if self.game_factory is None:
                evaluator = TorchEvaluator(self.model, self.device)
                games = []
                for seed in seeds:
                    search = MCTS(
                        evaluator,
                        simulations=self.config.simulations_per_move,
                        c_puct=1.5,
                        seed=seed,
                    )
                    games.append(
                        _WorkerResult(
                            os.getpid(),
                            play_game(
                                search, max_plies=self.config.max_plies, seed=seed
                            ),
                        )
                    )
                return games
            return [
                _WorkerResult(os.getpid(), self.game_factory(seed)) for seed in seeds
            ]

        state_dict = {
            name: tensor.detach().to("cpu").clone()
            for name, tensor in self.model.state_dict().items()
        }
        jobs = [
            _WorkerJob(self.config, state_dict, seed, self.game_factory)
            for seed in seeds
        ]
        return pool.map(_worker_entry, jobs, chunksize=1)

    def _commit_game(self, game: GameResult) -> None:
        self.replay.append_game(game)
        target = self._live_target()
        completed = self.progress.completed_games + 1
        self.progress = TrainingProgress(
            completed, target, self.progress.training_steps
        )
        self._write_status("running")
        try:
            batch = self.replay.sample(self.config.batch_size, self.rng)
        except ValueError as error:
            if not str(error).startswith("样本不足："):
                raise
            return
        self._train(batch)
        self.progress = TrainingProgress(
            completed, target, self.progress.training_steps + 1
        )
        self._write_status("running")

    def _train(self, batch: ReplayBatch) -> tuple[float, float]:
        losses = train_batch(
            self.model,
            self.optimizer,
            batch.states,
            batch.policy_indices,
            batch.policy_probabilities,
            batch.values,
            self.device,
        )
        if self.scheduler is not None:
            self.scheduler.step()
        return losses

    def _live_target(self) -> int:
        if not self.control.status_path.exists():
            return self.progress.target_games
        target = max(self.progress.target_games, self.control.read_status().target_games)
        if target != self.progress.target_games:
            self.progress = TrainingProgress(
                self.progress.completed_games, target, self.progress.training_steps
            )
        return target

    def _save_checkpoint(self) -> None:
        self.checkpoints.save(
            self.model,
            self.optimizer,
            self.progress,
            self.config,
            replay_manifest_hash=self.replay.manifest_hash,
            replay_manifest_version=self.replay.manifest_version,
            scheduler=self.scheduler,
            numpy_generator=self.rng,
        )

    def _write_status(self, phase: str, *, message: str = "") -> None:
        worker_message = f"self_play_workers_effective={self.worker_count}"
        status_message = (
            worker_message if not message else f"{message}; {worker_message}"
        )
        self.control.write_status(
            RunStatus(
                phase=phase,  # type: ignore[arg-type]
                completed_games=self.progress.completed_games,
                target_games=self.progress.target_games,
                training_steps=self.progress.training_steps,
                device=str(self.device),
                message=status_message,
            )
        )
