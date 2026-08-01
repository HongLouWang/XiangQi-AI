from __future__ import annotations

import logging
import multiprocessing
import os
import random
import traceback
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
LOGGER = logging.getLogger(__name__)


def _devices_match(requested: torch.device, actual: torch.device) -> bool:
    """裸 CUDA 接受默认 GPU；显式 CUDA 索引和其他设备严格匹配。"""
    if requested.type != actual.type:
        return False
    if requested.type == "cuda" and requested.index is None:
        return True
    return requested == actual


class TorchEvaluator:
    """在指定设备执行只读 Policy/Value 推理。"""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        if not isinstance(device, torch.device):
            raise TypeError("device 必须是 torch.device")
        tensors = (*model.parameters(), *model.buffers())
        if any(not _devices_match(device, tensor.device) for tensor in tensors):
            raise ValueError("model 的全部参数和缓冲区必须位于指定 device")
        self.model = model
        self.device = device

    def evaluate(self, state: SearchState) -> tuple[NDArray[np.float32], float]:
        was_training = self.model.training
        self.model.eval()
        try:
            inputs = torch.from_numpy(encode_board(state.board, state.side)).unsqueeze(
                0
            )
            with torch.no_grad():
                logits, values = self.model(inputs.to(self.device))
            policy = logits[0].detach().to("cpu", dtype=torch.float32).numpy().copy()
            return policy, float(values.reshape(-1)[0].detach().cpu())
        finally:
            self.model.train(was_training)


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
    state_tensor = torch.as_tensor(states)
    value_tensor = torch.as_tensor(value_targets)
    if state_tensor.ndim != 4:
        raise ValueError("states 必须是四维 batch")
    if not bool(torch.isfinite(state_tensor).all()):
        raise ValueError("states 必须全部为有限数")
    batch_size = int(state_tensor.shape[0])
    if value_tensor.shape != (batch_size,):
        raise ValueError("value_targets 必须是一维且与 batch 大小一致")
    if not bool(torch.isfinite(value_tensor).all()) or bool(
        ((value_tensor < -1) | (value_tensor > 1)).any()
    ):
        raise ValueError("value_targets 必须是 [-1, 1] 内的有限数")
    if len(policy_indices) != batch_size or len(policy_probabilities) != batch_size:
        raise ValueError("稀疏策略数量必须与 batch 大小一致")
    dense_policy = torch.zeros((batch_size, ACTION_SIZE), dtype=torch.float32)
    for row, (indices, probabilities) in enumerate(
        zip(policy_indices, policy_probabilities, strict=True)
    ):
        index_tensor = torch.as_tensor(indices)
        probability_tensor = torch.as_tensor(probabilities)
        if (
            index_tensor.ndim != 1
            or index_tensor.numel() == 0
            or probability_tensor.shape != index_tensor.shape
        ):
            raise ValueError("稀疏策略索引与概率必须是一维、非空且等长")
        if index_tensor.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("策略索引必须是整数")
        if bool(((index_tensor < 0) | (index_tensor >= ACTION_SIZE)).any()):
            raise ValueError("策略索引越界")
        indices64 = index_tensor.to(dtype=torch.int64)
        if torch.unique(indices64).numel() != indices64.numel():
            raise ValueError("策略索引不能重复")
        if not bool(torch.isfinite(probability_tensor).all()) or bool(
            (probability_tensor < 0).any()
        ):
            raise ValueError("策略概率必须有限且非负")
        if not bool(torch.isclose(probability_tensor.sum(), torch.tensor(1.0))):
            raise ValueError("策略概率之和必须为 1")
        dense_policy[row, indices64] = probability_tensor.to(dtype=torch.float32)

    model.train()
    logits, values = model(state_tensor.to(device=device, dtype=torch.float32))
    policy_targets = dense_policy.to(device)
    values_target = value_tensor.to(device=device, dtype=torch.float32)
    policy_loss = -(policy_targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
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
    game_number: int
    attempt: int


@dataclass(frozen=True, slots=True)
class _WorkerResult:
    pid: int
    game: GameResult | None
    seed: int
    game_number: int
    attempt: int
    error: str = ""
    traceback: str = ""


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
    try:
        random.seed(job.seed)
        np.random.seed(job.seed % (2**32))
        torch.manual_seed(job.seed)
        game = (
            job.game_factory(job.seed)
            if job.game_factory is not None
            else _production_game(job.config, job.state_dict, job.seed)
        )
        return _WorkerResult(os.getpid(), game, job.seed, job.game_number, job.attempt)
    except Exception as error:  # noqa: BLE001 - worker 边界必须序列化失败
        return _WorkerResult(
            os.getpid(),
            None,
            job.seed,
            job.game_number,
            job.attempt,
            f"{type(error).__name__}: {error}",
            traceback.format_exc(),
        )


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
            allow_replay_forward=True,
            expected_replay_total_games=self.replay.total_games,
        )
        target = max(loaded.progress.target_games, self.config.target_games)
        if self.control.status_path.exists():
            target = max(target, self.control.read_status().target_games)
        completed = self.replay.total_games
        target = max(target, completed)
        self.progress = TrainingProgress(
            completed,
            target,
            loaded.progress.training_steps,
        )
        # Replay 的完整棋局可能在上次 checkpoint 后已经原子提交。立即同步新
        # checkpoint，使下一次恢复重新回到严格 hash 一致状态。
        self._save_checkpoint()

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
                        if result.game is None:  # pragma: no cover - 生成器保证成功
                            raise AssertionError("成功结果缺少棋局")
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
                if self.control.try_mark_completed(
                    self.progress,
                    device=str(self.device),
                    message=self._worker_message,
                ):
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
        seeds = [
            self.config.seed + self.progress.completed_games + i + 1
            for i in range(count)
        ]
        state_dict = {
            name: tensor.detach().to("cpu").clone()
            for name, tensor in self.model.state_dict().items()
        }
        pending = {
            self.progress.completed_games + offset + 1: (seed, 1)
            for offset, seed in enumerate(seeds)
        }
        completed: dict[int, _WorkerResult] = {}
        while pending:
            jobs = [
                _WorkerJob(
                    self.config,
                    state_dict,
                    seed,
                    self.game_factory,
                    game_number,
                    attempt,
                )
                for game_number, (seed, attempt) in pending.items()
            ]
            if pool is not None:
                round_results = pool.map(_worker_entry, jobs, chunksize=1)
            else:
                round_results = [self._local_worker_entry(job) for job in jobs]
            next_pending: dict[int, tuple[int, int]] = {}
            for result in round_results:
                self.worker_pids.add(result.pid)
                if result.game is not None:
                    completed[result.game_number] = result
                    continue
                detail = (
                    f"self-play game {result.game_number} seed {result.seed} "
                    f"attempt {result.attempt} failed: {result.error}\n"
                    f"{result.traceback}"
                )
                LOGGER.error(detail)
                if result.attempt > self.config.game_retry_limit:
                    raise RuntimeError(detail)
                next_pending[result.game_number] = (result.seed, result.attempt + 1)
            pending = next_pending
        return [completed[number] for number in sorted(completed)]

    def _local_worker_entry(self, job: _WorkerJob) -> _WorkerResult:
        if self.game_factory is not None:
            return _worker_entry(job)
        try:
            random.seed(job.seed)
            np.random.seed(job.seed % (2**32))
            torch.manual_seed(job.seed)
            evaluator = TorchEvaluator(self.model, self.device)
            search = MCTS(
                evaluator,
                simulations=self.config.simulations_per_move,
                c_puct=1.5,
                seed=job.seed,
            )
            game = play_game(search, max_plies=self.config.max_plies, seed=job.seed)
            return _WorkerResult(
                os.getpid(), game, job.seed, job.game_number, job.attempt
            )
        except Exception as error:  # noqa: BLE001 - 与 spawn worker 相同协议
            return _WorkerResult(
                os.getpid(),
                None,
                job.seed,
                job.game_number,
                job.attempt,
                f"{type(error).__name__}: {error}",
                traceback.format_exc(),
            )

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
        target = max(
            self.progress.target_games, self.control.read_status().target_games
        )
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
        worker_message = self._worker_message
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

    @property
    def _worker_message(self) -> str:
        return f"self_play_workers_effective={self.worker_count}"
