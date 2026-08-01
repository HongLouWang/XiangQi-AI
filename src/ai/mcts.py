from __future__ import annotations

import math
from dataclasses import dataclass, field
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

    def evaluate(
        self, state: SearchState
    ) -> tuple[NDArray[np.floating], float]: ...


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
        if simulations <= 0:
            raise ValueError("simulations 必须大于 0")
        if c_puct < 0:
            raise ValueError("c_puct 不能小于 0")
        if dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha 必须大于 0")
        if not 0 <= dirichlet_fraction <= 1:
            raise ValueError("dirichlet_fraction 必须在 0 到 1 之间")

        self.evaluator = evaluator
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_fraction = dirichlet_fraction
        self._rng = np.random.default_rng(seed)
        self.root = Node(prior=1.0)

    def search(
        self, state: SearchState, *, add_noise: bool
    ) -> dict[Move, float]:
        self.root = Node(prior=1.0)

        root_moves = all_legal_moves(state.board, state.side)
        if not root_moves:
            for _ in range(self.simulations):
                self._backpropagate([], -1.0)
            return {}

        self._evaluate_and_expand(self.root, state, root_moves)
        if add_noise:
            self._add_root_noise()

        for _ in range(self.simulations):
            node = self.root
            simulation_state = state
            path: list[Node] = []

            while node.children:
                move, node = self._select_child(node)
                simulation_state = simulation_state.play(move)
                path.append(node)

            legal_moves = all_legal_moves(
                simulation_state.board, simulation_state.side
            )
            if not legal_moves:
                leaf_value = -1.0
            else:
                leaf_value = self._evaluate_and_expand(
                    node, simulation_state, legal_moves
                )

            self._backpropagate(path, float(leaf_value))

        return self._visit_policy()

    def _evaluate_and_expand(
        self,
        node: Node,
        state: SearchState,
        legal_moves: tuple[Move, ...],
    ) -> float:
        logits, value = self.evaluator.evaluate(state)
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
                self.c_puct
                * child.prior
                * parent_scale
                / (1 + child.visit_count)
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
