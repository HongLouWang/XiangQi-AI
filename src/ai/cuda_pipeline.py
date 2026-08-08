from __future__ import annotations

import inspect
import multiprocessing
import os
import queue
import random
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, Self, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray

from ai.encoding import ACTION_SIZE
from ai.mcts import MCTS, SearchState
from ai.self_play import GameResult, play_game
from xiangqi.board import Board
from xiangqi.domain import Color

T = TypeVar("T")


class PutQueue(Protocol[T]):
    def put(self, item: T) -> None: ...


class GetQueue(Protocol[T]):
    def get(self, block: bool = True, timeout: float | None = None) -> T: ...

    def get_nowait(self) -> T: ...


class BatchEvaluator(Protocol):
    def evaluate_many(
        self, states: tuple[SearchState, ...]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]: ...


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    worker_id: int
    request_id: int
    state: SearchState

    def __reduce__(self) -> tuple[object, tuple[int, int, str, str]]:
        return (
            _restore_inference_request,
            (
                self.worker_id,
                self.request_id,
                self.state.board.to_fen(),
                self.state.side.value,
            ),
        )


def _restore_inference_request(
    worker_id: int, request_id: int, fen: str, side: str
) -> InferenceRequest:
    return InferenceRequest(
        worker_id, request_id, SearchState(Board.from_fen(fen), Color(side))
    )


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    worker_id: int
    request_id: int
    policy: NDArray[np.float32]
    value: float


class RemoteEvaluator:
    def __init__(
        self,
        worker_id: int,
        request_queue: PutQueue[InferenceRequest],
        response_queue: GetQueue[InferenceResponse],
    ) -> None:
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.next_request_id = 1

    def evaluate(self, state: SearchState) -> tuple[NDArray[np.float32], float]:
        request_id = self.next_request_id
        self.next_request_id += 1
        self.request_queue.put(InferenceRequest(self.worker_id, request_id, state))
        response = self.response_queue.get()
        if (response.worker_id, response.request_id) != (self.worker_id, request_id):
            raise RuntimeError("推理响应与请求不匹配")
        return response.policy, response.value


class CudaInferenceBroker:
    def __init__(
        self,
        evaluator: BatchEvaluator,
        request_queue: GetQueue[InferenceRequest],
        response_queues: Mapping[int, PutQueue[InferenceResponse]],
        *,
        max_batch_size: int,
        batch_collect_timeout: float = 0.002,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise ValueError("max_batch_size 必须是正整数")
        self.evaluator = evaluator
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.max_batch_size = max_batch_size
        if batch_collect_timeout < 0:
            raise ValueError("batch_collect_timeout 必须是非负数")
        self.batch_collect_timeout = float(batch_collect_timeout)
        self.last_inference_batch_size = 0
        self.max_inference_batch_size = 0
        self.inference_requests = 0

    def serve_one_batch(self, *, first_request_timeout: float) -> int:
        try:
            first = self.request_queue.get(timeout=first_request_timeout)
        except queue.Empty:
            return 0

        requests = [first]
        deadline = time.monotonic() + self.batch_collect_timeout
        while len(requests) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                requests.append(self.request_queue.get(timeout=remaining))
            except queue.Empty:
                break

        try:
            policies, values = self.evaluator.evaluate_many(
                tuple(request.state for request in requests)
            )
        except torch.OutOfMemoryError:
            for request in requests:
                self.request_queue.put(request)  # type: ignore[attr-defined]
            raise
        policy_array = np.asarray(policies)
        value_array = np.asarray(values)
        if policy_array.shape != (len(requests), ACTION_SIZE) or value_array.shape != (
            len(requests),
        ):
            raise ValueError("批量评估器输出形状与请求数量不匹配")

        for index, request in enumerate(requests):
            response_queue = self.response_queues.get(request.worker_id)
            if response_queue is None:
                raise RuntimeError(f"未知 worker 响应队列: {request.worker_id}")
            response_queue.put(
                InferenceResponse(
                    request.worker_id,
                    request.request_id,
                    np.asarray(policy_array[index], dtype=np.float32),
                    float(value_array[index]),
                )
            )

        count = len(requests)
        self.last_inference_batch_size = count
        self.max_inference_batch_size = max(self.max_inference_batch_size, count)
        self.inference_requests += count
        return count


@dataclass(frozen=True, slots=True)
class SelfPlayTask:
    game_number: int
    seed: int


@dataclass(frozen=True, slots=True)
class GameCompleted:
    worker_id: int
    game_number: int
    pid: int
    game: GameResult


@dataclass(frozen=True, slots=True)
class WorkerFailed:
    worker_id: int
    game_number: int
    pid: int
    error: str
    traceback: str


SpawnGameFactory = Callable[..., GameResult]
HeartbeatCallback = Callable[["CudaSelfPlayPipeline"], None]


def _call_game_factory(
    game_factory: SpawnGameFactory,
    evaluator: RemoteEvaluator,
    seed: int,
    game_number: int,
) -> GameResult:
    signature = inspect.signature(game_factory)
    try:
        signature.bind(evaluator, seed, game_number)
    except TypeError:
        signature.bind(seed)
        return game_factory(seed)
    return game_factory(evaluator, seed, game_number)


def _cuda_worker_entry(
    worker_id: int,
    task_queue: Any,
    request_queue: Any,
    response_queue: Any,
    result_queue: Any,
    simulations_per_move: int,
    max_plies: int,
    c_puct: float,
    game_factory: SpawnGameFactory | None,
) -> None:
    evaluator = RemoteEvaluator(worker_id, request_queue, response_queue)
    while True:
        task = task_queue.get()
        if task is None:
            return
        try:
            random.seed(task.seed)
            np.random.seed(task.seed % (2**32))
            if game_factory is None:
                search = MCTS(
                    evaluator,
                    simulations=simulations_per_move,
                    c_puct=c_puct,
                    seed=task.seed,
                )
                game = play_game(search, max_plies=max_plies, seed=task.seed)
            else:
                game = _call_game_factory(
                    game_factory, evaluator, task.seed, task.game_number
                )
            result_queue.put(
                GameCompleted(worker_id, task.game_number, os.getpid(), game)
            )
        except Exception as error:  # noqa: BLE001 - worker 边界必须传回失败
            result_queue.put(
                WorkerFailed(
                    worker_id,
                    task.game_number,
                    os.getpid(),
                    f"{type(error).__name__}: {error}",
                    traceback.format_exc(),
                )
            )


class CudaSelfPlayPipeline:
    def __init__(
        self,
        evaluator: BatchEvaluator,
        *,
        worker_count: int,
        max_active_games: int,
        simulations_per_move: int,
        max_plies: int,
        seed: int,
        max_batch_size: int | None = None,
        game_retry_limit: int = 0,
        heartbeat_interval: float = 5.0,
        on_heartbeat: HeartbeatCallback | None = None,
        c_puct: float = 1.5,
        game_factory: SpawnGameFactory | None = None,
    ) -> None:
        for name, value in {
            "worker_count": worker_count,
            "max_active_games": max_active_games,
            "simulations_per_move": simulations_per_move,
            "max_plies": max_plies,
        }.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        self.evaluator = evaluator
        self.worker_count = worker_count
        self.max_active_games = min(max_active_games, worker_count)
        self.simulations_per_move = simulations_per_move
        self.max_plies = max_plies
        self.seed = seed
        self.max_batch_size = max_batch_size or self.max_active_games
        if type(game_retry_limit) is not int or game_retry_limit < 0:
            raise ValueError("game_retry_limit 必须是非负整数")
        self.game_retry_limit = game_retry_limit
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval 必须是正数")
        self.heartbeat_interval = float(heartbeat_interval)
        self.on_heartbeat = on_heartbeat
        self.c_puct = c_puct
        self.game_factory = game_factory
        self.active_games = 0
        self.processes: list[multiprocessing.Process] = []
        self._process_by_worker: dict[int, multiprocessing.Process] = {}
        self._task_queues: dict[int, Any] = {}
        self._response_queues: dict[int, Any] = {}
        self._request_queue: Any | None = None
        self._result_queue: Any | None = None
        self._broker: CudaInferenceBroker | None = None
        self._refill_enabled = True
        self.oom_downgrades = 0
        self._context: Any | None = None
        self._next_worker_id = worker_count + 1

    @property
    def last_inference_batch_size(self) -> int:
        return 0 if self._broker is None else self._broker.last_inference_batch_size

    @property
    def max_inference_batch_size(self) -> int:
        return 0 if self._broker is None else self._broker.max_inference_batch_size

    @property
    def inference_requests(self) -> int:
        return 0 if self._broker is None else self._broker.inference_requests

    def __enter__(self) -> Self:
        if self.processes:
            raise RuntimeError("CUDA 自我对弈流水线已经启动")
        context = multiprocessing.get_context("spawn")
        self._context = context
        self._request_queue = context.Queue()
        self._result_queue = context.Queue()
        for worker_id in range(1, self.worker_count + 1):
            self._start_worker(worker_id)
        self._broker = CudaInferenceBroker(
            self.evaluator,
            self._request_queue,
            self._response_queues,
            max_batch_size=self.max_batch_size,
        )
        return self

    def _start_worker(self, worker_id: int) -> None:
        if self._context is None or self._request_queue is None:
            raise RuntimeError("CUDA 自我对弈流水线上下文尚未初始化")
        task_queue = self._context.Queue()
        response_queue = self._context.Queue()
        process = self._context.Process(
            target=_cuda_worker_entry,
            args=(
                worker_id,
                task_queue,
                self._request_queue,
                response_queue,
                self._result_queue,
                self.simulations_per_move,
                self.max_plies,
                self.c_puct,
                self.game_factory,
            ),
            name=f"cuda-self-play-{worker_id}",
        )
        process.start()
        self._task_queues[worker_id] = task_queue
        self._response_queues[worker_id] = response_queue
        self.processes.append(process)
        self._process_by_worker[worker_id] = process

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def generate(self, game_numbers: Iterable[int]) -> Iterator[GameCompleted]:
        if self._broker is None or self._result_queue is None:
            raise RuntimeError("CUDA 自我对弈流水线尚未启动")
        pending = iter(game_numbers)
        busy: set[int] = set()
        task_by_worker: dict[int, int] = {}
        attempts: dict[int, int] = {}
        self._refill_enabled = True
        last_heartbeat = time.monotonic()

        def assign_number(worker_id: int, game_number: int) -> None:
            self._task_queues[worker_id].put(
                SelfPlayTask(game_number, self.seed + game_number)
            )
            busy.add(worker_id)
            task_by_worker[worker_id] = game_number
            self.active_games += 1

        def assign(worker_id: int) -> bool:
            if not self._refill_enabled or len(busy) >= self.max_active_games:
                return False
            try:
                game_number = next(pending)
            except StopIteration:
                return False
            assign_number(worker_id, game_number)
            return True

        for worker_id in range(1, self.max_active_games + 1):
            assign(worker_id)

        while busy:
            try:
                self._broker.serve_one_batch(first_request_timeout=0.005)
            except torch.OutOfMemoryError:
                if self._broker.max_batch_size <= 1:
                    raise
                self._broker.max_batch_size = max(1, self._broker.max_batch_size // 2)
                self.max_active_games = max(1, self.max_active_games // 2)
                self.oom_downgrades += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            now = time.monotonic()
            if (
                self.on_heartbeat is not None
                and now - last_heartbeat >= self.heartbeat_interval
            ):
                self.on_heartbeat(self)
                last_heartbeat = now
            try:
                event = self._result_queue.get_nowait()
            except queue.Empty:
                for worker_id in tuple(busy):
                    process = self._process_by_worker[worker_id]
                    if process.is_alive():
                        continue
                    process.join(timeout=0.1)
                    game_number = task_by_worker.pop(worker_id)
                    busy.remove(worker_id)
                    self.active_games -= 1
                    attempts[game_number] = attempts.get(game_number, 0) + 1
                    if attempts[game_number] > self.game_retry_limit:
                        raise RuntimeError(
                            f"CUDA worker {worker_id} 在处理第 {game_number} 局时"
                            f"意外退出，exitcode={process.exitcode}"
                        )
                    replacement_id = self._next_worker_id
                    self._next_worker_id += 1
                    self._start_worker(replacement_id)
                    assign_number(replacement_id, game_number)
                continue
            busy.remove(event.worker_id)
            task_by_worker.pop(event.worker_id, None)
            self.active_games -= 1
            if type(event) is WorkerFailed:
                attempts[event.game_number] = attempts.get(event.game_number, 0) + 1
                if attempts[event.game_number] <= self.game_retry_limit:
                    assign_number(event.worker_id, event.game_number)
                    continue
                raise RuntimeError(
                    f"CUDA worker {event.worker_id} 生成第 {event.game_number} 局失败: "
                    f"{event.error}\n{event.traceback}"
                )
            if not isinstance(event, GameCompleted):
                raise TypeError("收到未知 CUDA worker 事件")
            yield event
            assign(event.worker_id)

    def stop_refilling(self) -> None:
        self._refill_enabled = False

    def close(self) -> None:
        for task_queue in self._task_queues.values():
            task_queue.put(None)
        for process in self.processes:
            process.join(timeout=2.0)
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        self.active_games = 0
