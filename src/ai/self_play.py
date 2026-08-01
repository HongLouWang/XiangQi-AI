from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ai.encoding import encode_action, encode_board
from ai.mcts import SearchState
from xiangqi.board import Board
from xiangqi.domain import Color, Move, PositionResult
from xiangqi.rules import all_legal_moves, evaluate_position


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """一次落子前保存的训练样本；value 使用该样本行棋方视角。"""

    state: NDArray[np.float32]
    policy_indices: NDArray[np.int64]
    policy_probabilities: NDArray[np.float32]
    side: Color
    value: float


@dataclass(frozen=True, slots=True)
class GameResult:
    samples: tuple[TrainingSample, ...]
    winner: Color | None
    plies: int
    termination: str


class Search(Protocol):
    def search(self, state: Any, *, add_noise: bool) -> dict[Any, float]: ...


class GameAdapter(Protocol):
    """将自我对弈循环与具体棋局状态隔离，便于精确测试长局上限。"""

    def initial_state(self) -> Any: ...

    def side(self, state: Any) -> Color: ...

    def position(self, state: Any) -> PositionResult: ...

    def encode_state(self, state: Any) -> NDArray[np.float32]: ...

    def encode_action(self, state: Any, move: Any) -> int: ...

    def legal_moves(self, state: Any) -> tuple[Any, ...]: ...

    def play(self, state: Any, move: Any) -> Any: ...


class XiangqiGameAdapter:
    """生产环境适配器，直接复用项目的棋盘和规则实现。"""

    def initial_state(self) -> SearchState:
        return SearchState(Board.standard(), Color.RED)

    def side(self, state: SearchState) -> Color:
        return state.side

    def position(self, state: SearchState) -> PositionResult:
        return evaluate_position(state.board, state.side)

    def encode_state(self, state: SearchState) -> NDArray[np.float32]:
        return encode_board(state.board, state.side)

    def encode_action(self, state: SearchState, move: Move) -> int:
        return encode_action(move, state.side)

    def legal_moves(self, state: SearchState) -> tuple[Move, ...]:
        return all_legal_moves(state.board, state.side)

    def play(self, state: SearchState, move: Move) -> SearchState:
        return state.play(move)


def play_game(
    search: Search,
    max_plies: int = 1024,
    *,
    seed: int = 0,
    temperature_plies: int = 30,
    initial_state: Any | None = None,
    adapter: GameAdapter | None = None,
) -> GameResult:
    """完成一盘自我对弈；默认的 1024 ply 等于 512 个完整回合。"""
    if (
        not isinstance(max_plies, Integral)
        or isinstance(max_plies, (bool, np.bool_))
        or max_plies <= 0
    ):
        raise ValueError("max_plies 必须是正整数")
    if (
        not isinstance(temperature_plies, Integral)
        or isinstance(temperature_plies, (bool, np.bool_))
        or temperature_plies < 0
    ):
        raise ValueError("temperature_plies 必须是非负整数")

    game = adapter if adapter is not None else XiangqiGameAdapter()
    state = game.initial_state() if initial_state is None else initial_state
    rng = np.random.default_rng(seed)
    pending: list[
        tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.float32], Color]
    ] = []

    for ply in range(int(max_plies)):
        position = game.position(state)
        if position.kind.value != "ongoing":
            return _finish(pending, position.winner, ply, position.kind.value)

        policy = search.search(state, add_noise=True)
        move, indices, probabilities = _select_move(
            policy,
            state=state,
            adapter=game,
            stochastic=ply < int(temperature_plies),
            rng=rng,
        )
        encoded_state = np.asarray(game.encode_state(state), dtype=np.float32).copy()
        encoded_state.setflags(write=False)
        indices.setflags(write=False)
        probabilities.setflags(write=False)
        pending.append((encoded_state, indices, probabilities, game.side(state)))
        state = game.play(state, move)

    final_position = game.position(state)
    if final_position.kind.value != "ongoing":
        return _finish(
            pending,
            final_position.winner,
            int(max_plies),
            final_position.kind.value,
        )
    return _finish(pending, None, int(max_plies), "move_limit")


def _select_move(
    policy: dict[Any, float],
    *,
    state: Any,
    adapter: GameAdapter,
    stochastic: bool,
    rng: np.random.Generator,
) -> tuple[Any, NDArray[np.int64], NDArray[np.float32]]:
    if not policy:
        raise ValueError("持续局面中的搜索策略不能为空")

    legal = set(adapter.legal_moves(state))
    if any(move not in legal for move in policy):
        raise ValueError("搜索策略包含非法着法")

    entries: list[tuple[int, Any, float]] = []
    for move, probability in policy.items():
        try:
            value = float(probability)
        except (TypeError, ValueError) as error:
            raise ValueError("策略概率必须是有限非负数") from error
        if not np.isfinite(value) or value < 0:
            raise ValueError("策略概率必须是有限非负数")
        entries.append((adapter.encode_action(state, move), move, value))
    entries.sort(key=lambda entry: entry[0])

    indices = np.asarray([entry[0] for entry in entries], dtype=np.int64)
    if np.unique(indices).size != indices.size:
        raise ValueError("策略包含重复的动作编码")
    probabilities64 = np.asarray([entry[2] for entry in entries], dtype=np.float64)
    total = float(probabilities64.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("策略概率之和必须为有限正数")
    probabilities64 /= total
    probabilities = probabilities64.astype(np.float32)

    selected = (
        int(rng.choice(len(entries), p=probabilities64))
        if stochastic
        else int(np.argmax(probabilities64))
    )
    return entries[selected][1], indices, probabilities


def _finish(
    pending: list[
        tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.float32], Color]
    ],
    winner: Color | None,
    plies: int,
    termination: str,
) -> GameResult:
    samples = tuple(
        TrainingSample(
            state=state,
            policy_indices=indices,
            policy_probabilities=probabilities,
            side=side,
            value=(0.0 if winner is None else 1.0 if side is winner else -1.0),
        )
        for state, indices, probabilities, side in pending
    )
    return GameResult(
        samples=samples,
        winner=winner,
        plies=plies,
        termination=termination,
    )
