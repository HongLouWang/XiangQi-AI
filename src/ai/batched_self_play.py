from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ai.encoding import encode_board
from ai.mcts import MCTS, EvaluationRequest, SearchSession, SearchState
from ai.self_play import GameResult, _finish, _select_move
from xiangqi.board import Board
from xiangqi.domain import Color
from xiangqi.rules import all_legal_moves, evaluate_position


class BatchEvaluator(Protocol):
    def evaluate_many(
        self, states: tuple[SearchState, ...]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...


@dataclass(slots=True)
class _GameSlot:
    game_number: int
    seed: int
    state: SearchState = field(
        default_factory=lambda: SearchState(Board.standard(), Color.RED)
    )
    ply: int = 0
    pending: list[
        tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.float32], Color]
    ] = field(default_factory=list)
    rng: np.random.Generator = field(init=False)
    session: SearchSession | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)


class BatchedSelfPlay:
    """一个进程内同步推进多盘棋，并将 MCTS 叶子合并推理。"""

    def __init__(
        self,
        evaluator: BatchEvaluator,
        *,
        simulations: int,
        max_plies: int,
        seed: int,
        temperature_plies: int = 30,
    ) -> None:
        if simulations <= 0 or max_plies <= 0:
            raise ValueError("simulations 和 max_plies 必须是正整数")
        self.evaluator = evaluator
        self.simulations = simulations
        self.max_plies = max_plies
        self.seed = seed
        self.temperature_plies = temperature_plies
        self.last_batch_size = 0

    def generate(self, *, count: int, parallel_games: int) -> list[GameResult]:
        if count <= 0 or parallel_games <= 0:
            raise ValueError("count 和 parallel_games 必须是正整数")
        active: list[_GameSlot] = []
        completed: dict[int, GameResult] = {}
        launched = 0

        while len(completed) < count:
            while len(active) < parallel_games and launched < count:
                launched += 1
                active.append(_GameSlot(launched, self.seed + launched))

            requests: list[tuple[_GameSlot, EvaluationRequest]] = []
            finished_slots: list[_GameSlot] = []
            for slot in active:
                if slot.session is None:
                    search = MCTS(
                        self.evaluator,  # evaluate_many 由调度器调用
                        simulations=self.simulations,
                        c_puct=1.5,
                        seed=slot.seed + slot.ply,
                    )
                    slot.session = search.start_search(slot.state, add_noise=True)
                request = slot.session.next_evaluation()
                if request is not None:
                    requests.append((slot, request))
                elif slot.session.done:
                    result = self._play_completed_search(slot)
                    if result is not None:
                        completed[slot.game_number] = result
                        finished_slots.append(slot)

            for slot in finished_slots:
                active.remove(slot)

            if requests:
                states = tuple(request.state for _, request in requests)
                policies, values = self.evaluator.evaluate_many(states)
                if policies.shape[0] != len(requests) or values.shape != (
                    len(requests),
                ):
                    raise ValueError("批量评估结果数量与请求不一致")
                self.last_batch_size = len(requests)
                for row, (slot, request) in enumerate(requests):
                    assert slot.session is not None
                    slot.session.accept_evaluation(
                        request, policies[row], float(values[row])
                    )

        return [completed[number] for number in sorted(completed)]

    def _play_completed_search(self, slot: _GameSlot) -> GameResult | None:
        assert slot.session is not None
        policy = slot.session.policy()
        move, indices, probabilities = _select_move(
            policy,
            state=slot.state,
            adapter=_XIANGQI_ADAPTER,
            stochastic=slot.ply < self.temperature_plies,
            rng=slot.rng,
        )
        encoded = encode_board(slot.state.board, slot.state.side).copy()
        encoded.setflags(write=False)
        indices.setflags(write=False)
        probabilities.setflags(write=False)
        slot.pending.append((encoded, indices, probabilities, slot.state.side))
        slot.state = slot.state.play(move)
        slot.ply += 1
        slot.session = None
        position = evaluate_position(slot.state.board, slot.state.side)
        if position.kind.value != "ongoing":
            return _finish(slot.pending, position.winner, slot.ply, position.kind.value)
        if slot.ply >= self.max_plies:
            return _finish(slot.pending, None, slot.ply, "move_limit")
        return None


class _XiangqiAdapter:
    @staticmethod
    def legal_moves(state: SearchState):
        return all_legal_moves(state.board, state.side)

    @staticmethod
    def encode_action(state: SearchState, move):
        from ai.encoding import encode_action

        return encode_action(move, state.side)


_XIANGQI_ADAPTER = _XiangqiAdapter()
