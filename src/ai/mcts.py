from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ai.encoding import encode_action, legal_policy
from xiangqi.board import Board
from xiangqi.domain import Color, Move
from xiangqi.rules import all_legal_moves


@dataclass(frozen=True, slots=True)
class SearchState:
    """MCTS 所需的最小、不可变棋局状态。"""

    board: Board
    side: Color

    def play(self, move: Move) -> SearchState:
        if move.piece.color is not self.side:
            raise ValueError("只能推进当前行棋方的着法")
        return SearchState(
            self.board.move_unchecked(move.start, move.end), self.side.opponent
        )


class Evaluator(Protocol):
    """返回动作 logits 和当前行棋方视角局面价值的推理接口。"""

    def evaluate(self, state: SearchState) -> tuple[NDArray[np.floating], float]: ...


@dataclass(slots=True)
class Node:
    """一条父节点到本节点的搜索边；价值始终采用父方视角。"""

    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[Move, Node] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


class MCTS:
    def __init__(
        self,
        evaluator: Evaluator,
        simulations: int,
        c_puct: float,
        seed: int,
        dirichlet_alpha: float = 0.3,
        dirichlet_fraction: float = 0.25,
    ) -> None:
        if (
            not isinstance(simulations, Integral)
            or isinstance(simulations, (bool, np.bool_))
            or simulations <= 0
        ):
            raise ValueError("simulations 必须是正整数")
        self._require_finite_real("c_puct", c_puct)
        if c_puct < 0:
            raise ValueError("c_puct 必须是有限的非负数")
        self._require_finite_real("dirichlet_alpha", dirichlet_alpha)
        if dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha 必须是有限的正数")
        self._require_finite_real("dirichlet_fraction", dirichlet_fraction)
        if not 0 <= dirichlet_fraction <= 1:
            raise ValueError("dirichlet_fraction 必须是 0 到 1 之间的有限数")

        self.evaluator = evaluator
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_fraction = dirichlet_fraction
        self._rng = np.random.default_rng(seed)
        self.root = Node(prior=1.0)

    @staticmethod
    def _require_finite_real(name: str, value: float) -> None:
        if (
            not isinstance(value, Real)
            or isinstance(value, (bool, np.bool_))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} 必须是有限实数")

    def search(self, state: SearchState, *, add_noise: bool) -> dict[Move, float]:
        session = self.start_search(state, add_noise=add_noise)
        while not session.done:
            request = session.next_evaluation()
            if request is None:
                continue
            logits, value = self.evaluator.evaluate(request.state)
            session.accept_evaluation(request, logits, value)
        return session.policy()

    def start_search(self, state: SearchState, *, add_noise: bool) -> SearchSession:
        self.root = Node(prior=1.0)
        return SearchSession(self, state, add_noise=add_noise)

    def _evaluate_and_expand(
        self,
        node: Node,
        state: SearchState,
        legal_moves: tuple[Move, ...],
    ) -> float:
        logits, value = self.evaluator.evaluate(state)
        return self._expand_with_evaluation(node, state, legal_moves, logits, value)

    def _expand_with_evaluation(
        self,
        node: Node,
        state: SearchState,
        legal_moves: tuple[Move, ...],
        logits: NDArray[np.floating],
        value: float,
    ) -> float:
        if not np.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("评估价值必须是 [-1, 1] 内的有限数")
        priors = legal_policy(logits, legal_moves, state.side)
        node.children = {
            move: Node(prior=float(priors[encode_action(move, state.side)]))
            for move in legal_moves
        }
        return float(value)

    def _select_child(self, parent: Node) -> tuple[Move, Node]:
        parent_scale = math.sqrt(parent.visit_count)

        def score(item: tuple[Move, Node]) -> float:
            _, child = item
            exploration = (
                self.c_puct * child.prior * parent_scale / (1 + child.visit_count)
            )
            return child.mean_value + exploration

        return max(parent.children.items(), key=score)

    def _add_root_noise(self) -> None:
        children = tuple(self.root.children.values())
        noise = self._rng.dirichlet(
            np.full(len(children), self.dirichlet_alpha, dtype=np.float64)
        )
        keep = 1.0 - self.dirichlet_fraction
        for child, sample in zip(children, noise, strict=True):
            child.prior = keep * child.prior + self.dirichlet_fraction * float(sample)

    def _backpropagate(self, path: list[Node], leaf_value: float) -> None:
        value = leaf_value
        for node in reversed(path):
            value = -value
            node.visit_count += 1
            node.value_sum += value
        self.root.visit_count += 1
        self.root.value_sum += value

    def _visit_policy(self) -> dict[Move, float]:
        if not self.root.children:
            return {}
        visits = sum(child.visit_count for child in self.root.children.values())
        return {
            move: child.visit_count / visits
            for move, child in self.root.children.items()
        }


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    state: SearchState
    node: Node
    path: tuple[Node, ...]
    legal_moves: tuple[Move, ...]
    is_root: bool = False


class SearchSession:
    """每次最多暴露一个神经网络叶子评估请求的 MCTS 搜索。"""

    def __init__(self, search: MCTS, state: SearchState, *, add_noise: bool) -> None:
        self.search = search
        self.state = state
        self.add_noise = add_noise
        self.completed_simulations = 0
        self._initialized = False
        self._pending: EvaluationRequest | None = None
        root_moves = all_legal_moves(state.board, state.side)
        if not root_moves:
            for _ in range(search.simulations):
                search._backpropagate([], -1.0)
                self.completed_simulations += 1
            self._initialized = True
        else:
            self._pending = EvaluationRequest(
                state, search.root, (), root_moves, is_root=True
            )

    @property
    def done(self) -> bool:
        return (
            self._initialized and self.completed_simulations >= self.search.simulations
        )

    def next_evaluation(self) -> EvaluationRequest | None:
        if self._pending is not None:
            return self._pending
        while not self.done:
            node = self.search.root
            simulation_state = self.state
            path: list[Node] = []
            while node.children:
                move, node = self.search._select_child(node)
                simulation_state = simulation_state.play(move)
                path.append(node)
            legal_moves = all_legal_moves(simulation_state.board, simulation_state.side)
            if not legal_moves:
                self.search._backpropagate(path, -1.0)
                self.completed_simulations += 1
                continue
            self._pending = EvaluationRequest(
                simulation_state, node, tuple(path), legal_moves
            )
            return self._pending
        return None

    def accept_evaluation(
        self,
        request: EvaluationRequest,
        logits: NDArray[np.floating],
        value: float,
    ) -> None:
        if request is not self._pending:
            raise ValueError("评估结果与当前待处理请求不匹配")
        leaf_value = self.search._expand_with_evaluation(
            request.node, request.state, request.legal_moves, logits, value
        )
        self._pending = None
        if request.is_root:
            self._initialized = True
            if self.add_noise:
                self.search._add_root_noise()
            return
        self.search._backpropagate(list(request.path), leaf_value)
        self.completed_simulations += 1

    def policy(self) -> dict[Move, float]:
        if not self.done:
            raise RuntimeError("搜索尚未完成")
        return self.search._visit_policy()
