from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from ai.cuda_pipeline import (
    CudaInferenceBroker,
    CudaSelfPlayPipeline,
    InferenceRequest,
    InferenceResponse,
    RemoteEvaluator,
)
from ai.encoding import ACTION_SIZE
from ai.mcts import SearchState
from ai.self_play import GameResult
from xiangqi.board import Board
from xiangqi.domain import Color

STATE_RED = SearchState(Board.standard(), Color.RED)
STATE_BLACK = SearchState(Board.standard(), Color.BLACK)


class RecordingBatchEvaluator:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def evaluate_many(self, states):
        self.batch_sizes.append(len(states))
        return (
            np.zeros((len(states), ACTION_SIZE), dtype=np.float32),
            np.arange(len(states), dtype=np.float32) / 4,
        )


def _tiny_spawn_game(evaluator, seed: int, game_number: int) -> GameResult:
    evaluator.evaluate(STATE_RED)
    return GameResult((), None, game_number, f"seed-{seed}")


def _second_game_is_slow(evaluator, seed: int, game_number: int) -> GameResult:
    evaluator.evaluate(STATE_RED)
    if game_number == 2:
        time.sleep(0.3)
    return GameResult((), None, game_number, f"seed-{seed}")


def _flaky_spawn_game(evaluator, seed: int, game_number: int) -> GameResult:
    marker = Path(os.environ["XIANGQI_CUDA_RETRY_MARKERS"]) / str(game_number)
    try:
        marker.touch(exist_ok=False)
    except FileExistsError:
        return GameResult((), None, 1, f"seed-{seed}")
    raise RuntimeError(f"transient-{seed}")


def _crash_first_worker(evaluator, seed: int, game_number: int) -> GameResult:
    if game_number != 1:
        time.sleep(0.3)
        return GameResult((), None, 1, f"seed-{seed}")
    marker = Path(os.environ["XIANGQI_CUDA_CRASH_MARKER"])
    try:
        marker.touch(exist_ok=False)
    except FileExistsError:
        return GameResult((), None, 1, f"recovered-{seed}")
    os._exit(7)


def _seed_only_game(seed: int) -> GameResult:
    return GameResult((), None, 1, f"seed-only-{seed}")


def test_remote_evaluator_routes_response_by_worker_and_request_id() -> None:
    requests: queue.Queue[object] = queue.Queue()
    responses: queue.Queue[object] = queue.Queue()
    evaluator = RemoteEvaluator(3, requests, responses)
    state = SearchState(Board.standard(), Color.RED)
    expected = np.zeros(ACTION_SIZE, dtype=np.float32)

    responses.put(InferenceResponse(3, 1, expected, 0.25))
    policy, value = evaluator.evaluate(state)

    request = requests.get_nowait()
    assert (request.worker_id, request.request_id, request.state) == (3, 1, state)
    assert np.array_equal(policy, expected)
    assert value == 0.25


def test_broker_batches_available_requests_and_routes_each_response() -> None:
    evaluator = RecordingBatchEvaluator()
    requests: queue.Queue[InferenceRequest] = queue.Queue()
    responses: dict[int, queue.Queue[InferenceResponse]] = {
        1: queue.Queue(),
        2: queue.Queue(),
    }
    requests.put(InferenceRequest(1, 7, STATE_RED))
    requests.put(InferenceRequest(2, 4, STATE_BLACK))
    broker = CudaInferenceBroker(evaluator, requests, responses, max_batch_size=8)

    assert broker.serve_one_batch(first_request_timeout=0.01) == 2

    assert evaluator.batch_sizes == [2]
    assert responses[1].get_nowait().request_id == 7
    assert responses[2].get_nowait().request_id == 4
    assert broker.inference_requests == 2
    assert broker.last_inference_batch_size == 2
    assert broker.max_inference_batch_size == 2


def test_broker_returns_zero_when_no_request_is_available() -> None:
    broker = CudaInferenceBroker(
        RecordingBatchEvaluator(), queue.Queue(), {1: queue.Queue()}, max_batch_size=8
    )

    assert broker.serve_one_batch(first_request_timeout=0.001) == 0


def test_broker_does_not_take_more_than_batch_limit() -> None:
    requests: queue.Queue[InferenceRequest] = queue.Queue()
    responses: dict[int, queue.Queue[InferenceResponse]] = {1: queue.Queue()}
    requests.put(InferenceRequest(1, 1, STATE_RED))
    requests.put(InferenceRequest(1, 2, STATE_RED))
    broker = CudaInferenceBroker(
        RecordingBatchEvaluator(), requests, responses, max_batch_size=1
    )

    assert broker.serve_one_batch(first_request_timeout=0.01) == 1
    assert requests.qsize() == 1


def test_broker_waits_a_short_window_for_nearby_worker_request() -> None:
    requests: queue.Queue[InferenceRequest] = queue.Queue()
    responses: dict[int, queue.Queue[InferenceResponse]] = {
        1: queue.Queue(),
        2: queue.Queue(),
    }
    requests.put(InferenceRequest(1, 1, STATE_RED))
    delayed = threading.Thread(
        target=lambda: (
            time.sleep(0.01),
            requests.put(InferenceRequest(2, 1, STATE_BLACK)),
        )
    )
    delayed.start()
    evaluator = RecordingBatchEvaluator()
    broker = CudaInferenceBroker(
        evaluator,
        requests,
        responses,
        max_batch_size=2,
        batch_collect_timeout=0.05,
    )

    assert broker.serve_one_batch(first_request_timeout=0.01) == 2
    delayed.join()
    assert evaluator.batch_sizes == [2]


def test_broker_rejects_wrong_evaluator_output_shape() -> None:
    class InvalidEvaluator:
        def evaluate_many(self, states):
            return (
                np.zeros((len(states) + 1, ACTION_SIZE), dtype=np.float32),
                np.zeros(len(states), dtype=np.float32),
            )

    requests: queue.Queue[InferenceRequest] = queue.Queue()
    requests.put(InferenceRequest(1, 1, STATE_RED))
    broker = CudaInferenceBroker(
        InvalidEvaluator(), requests, {1: queue.Queue()}, max_batch_size=8
    )

    with pytest.raises(ValueError, match="输出形状"):
        broker.serve_one_batch(first_request_timeout=0.01)


def test_broker_restores_requests_when_cuda_oom_occurs() -> None:
    class OomOnceEvaluator(RecordingBatchEvaluator):
        def evaluate_many(self, states):
            if not self.batch_sizes:
                self.batch_sizes.append(len(states))
                raise torch.OutOfMemoryError("CUDA out of memory")
            return super().evaluate_many(states)

    requests: queue.Queue[InferenceRequest] = queue.Queue()
    responses: dict[int, queue.Queue[InferenceResponse]] = {1: queue.Queue()}
    requests.put(InferenceRequest(1, 1, STATE_RED))
    broker = CudaInferenceBroker(
        OomOnceEvaluator(), requests, responses, max_batch_size=2
    )

    with pytest.raises(torch.OutOfMemoryError):
        broker.serve_one_batch(first_request_timeout=0.01)

    assert requests.qsize() == 1
    broker.max_batch_size = 1
    assert broker.serve_one_batch(first_request_timeout=0.01) == 1


def test_pipeline_uses_real_spawn_workers_and_cleans_them_up() -> None:
    pipeline = CudaSelfPlayPipeline(
        RecordingBatchEvaluator(),
        worker_count=2,
        max_active_games=2,
        simulations_per_move=1,
        max_plies=1,
        seed=10,
        game_factory=_tiny_spawn_game,
    )

    with pipeline:
        completed = list(pipeline.generate(game_numbers=(1, 2)))

    assert {item.game_number for item in completed} == {1, 2}
    assert len({item.pid for item in completed}) == 2
    assert all(item.pid != os.getpid() for item in completed)
    assert pipeline.active_games == 0
    assert not any(process.is_alive() for process in pipeline.processes)


def test_pipeline_yields_each_game_before_the_whole_group_finishes() -> None:
    with CudaSelfPlayPipeline(
        RecordingBatchEvaluator(),
        worker_count=2,
        max_active_games=2,
        simulations_per_move=1,
        max_plies=1,
        seed=20,
        game_factory=_second_game_is_slow,
    ) as pipeline:
        generated = pipeline.generate(game_numbers=(1, 2))
        first = next(generated)

        assert first.game_number == 1
        assert pipeline.active_games == 1
        assert [item.game_number for item in generated] == [2]


def test_pipeline_retries_failed_game_with_the_same_seed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    markers = tmp_path / "markers"
    markers.mkdir()
    monkeypatch.setenv("XIANGQI_CUDA_RETRY_MARKERS", str(markers))
    with CudaSelfPlayPipeline(
        RecordingBatchEvaluator(),
        worker_count=1,
        max_active_games=1,
        simulations_per_move=1,
        max_plies=1,
        seed=30,
        game_retry_limit=1,
        game_factory=_flaky_spawn_game,
    ) as pipeline:
        completed = list(pipeline.generate(game_numbers=(1,)))

    assert completed[0].game.termination == "seed-31"


def test_pipeline_accepts_existing_seed_only_game_factory() -> None:
    with CudaSelfPlayPipeline(
        RecordingBatchEvaluator(),
        worker_count=1,
        max_active_games=1,
        simulations_per_move=1,
        max_plies=1,
        seed=60,
        game_factory=_seed_only_game,
    ) as pipeline:
        completed = list(pipeline.generate(game_numbers=(1,)))

    assert completed[0].game.termination == "seed-only-61"


def test_pipeline_replaces_a_crashed_worker_and_retries_the_same_seed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "crashed-once"
    monkeypatch.setenv("XIANGQI_CUDA_CRASH_MARKER", str(marker))
    with (
        CudaSelfPlayPipeline(
            RecordingBatchEvaluator(),
            worker_count=2,
            max_active_games=2,
            simulations_per_move=1,
            max_plies=1,
            seed=70,
            game_retry_limit=1,
            game_factory=_crash_first_worker,
        ) as pipeline,
    ):
        completed = list(pipeline.generate(game_numbers=(1, 2)))

    assert {item.game_number for item in completed} == {1, 2}
    recovered = next(item for item in completed if item.game_number == 1)
    assert recovered.game.termination == "recovered-71"
    assert len(pipeline.processes) == 3


def test_pipeline_stop_refilling_drains_only_inflight_games() -> None:
    with CudaSelfPlayPipeline(
        RecordingBatchEvaluator(),
        worker_count=2,
        max_active_games=2,
        simulations_per_move=1,
        max_plies=1,
        seed=40,
        game_factory=_second_game_is_slow,
    ) as pipeline:
        generated = pipeline.generate(game_numbers=(1, 2, 3))
        first = next(generated)
        pipeline.stop_refilling()
        remaining = list(generated)

    assert {item.game_number for item in (first, *remaining)} == {1, 2}


def test_pipeline_emits_heartbeat_before_a_slow_game_finishes() -> None:
    heartbeats: list[int] = []
    with CudaSelfPlayPipeline(
        RecordingBatchEvaluator(),
        worker_count=1,
        max_active_games=1,
        simulations_per_move=1,
        max_plies=1,
        seed=50,
        game_factory=_second_game_is_slow,
        heartbeat_interval=0.02,
        on_heartbeat=lambda pipeline: heartbeats.append(pipeline.active_games),
    ) as pipeline:
        completed = list(pipeline.generate(game_numbers=(2,)))

    assert completed[0].game_number == 2
    assert heartbeats
    assert set(heartbeats) == {1}
