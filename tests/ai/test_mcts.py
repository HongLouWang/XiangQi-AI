from __future__ import annotations

import numpy as np
import pytest

from ai.encoding import ACTION_SIZE, encode_action
from ai.mcts import MCTS, Node, SearchState
from xiangqi.board import Board
from xiangqi.domain import Color
from xiangqi.rules import all_legal_moves


class CountingEvaluator:
    def __init__(self, value: float = 0.25) -> None:
        self.calls = 0
        self.value = value

    def evaluate(self, state: SearchState) -> tuple[np.ndarray, float]:
        self.calls += 1
        return np.zeros(ACTION_SIZE, dtype=np.float32), self.value


@pytest.mark.parametrize("simulations", [0, -1, 1.5, True])
def test_simulations_must_be_a_positive_integer(simulations: object) -> None:
    with pytest.raises((TypeError, ValueError), match="simulations"):
        MCTS(CountingEvaluator(), simulations=simulations, c_puct=1.5, seed=3)  # type: ignore[arg-type]


@pytest.mark.parametrize("c_puct", [-0.1, float("nan"), float("inf"), -float("inf")])
def test_c_puct_must_be_finite_and_non_negative(c_puct: float) -> None:
    with pytest.raises(ValueError, match="c_puct"):
        MCTS(CountingEvaluator(), simulations=1, c_puct=c_puct, seed=3)


@pytest.mark.parametrize(
    "alpha", [0.0, -0.1, float("nan"), float("inf"), -float("inf")]
)
def test_dirichlet_alpha_must_be_finite_and_positive(alpha: float) -> None:
    with pytest.raises(ValueError, match="dirichlet_alpha"):
        MCTS(
            CountingEvaluator(),
            simulations=1,
            c_puct=1.5,
            seed=3,
            dirichlet_alpha=alpha,
        )


@pytest.mark.parametrize(
    "fraction", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")]
)
def test_dirichlet_fraction_must_be_finite_and_in_range(fraction: float) -> None:
    with pytest.raises(ValueError, match="dirichlet_fraction"):
        MCTS(
            CountingEvaluator(),
            simulations=1,
            c_puct=1.5,
            seed=3,
            dirichlet_fraction=fraction,
        )


def test_search_visits_only_legal_root_moves_and_normalizes_policy() -> None:
    state = SearchState(Board.standard(), Color.RED)

    policy = MCTS(
        CountingEvaluator(), simulations=8, c_puct=1.5, seed=3
    ).search(state, add_noise=False)

    assert set(policy) == set(all_legal_moves(state.board, state.side))
    assert sum(policy.values()) == pytest.approx(1.0)
    assert all(probability >= 0.0 for probability in policy.values())


def test_search_executes_exactly_the_requested_simulation_count() -> None:
    search = MCTS(CountingEvaluator(), simulations=8, c_puct=1.5, seed=3)

    search.search(SearchState(Board.standard(), Color.RED), add_noise=False)

    assert search.root.visit_count == 8
    assert sum(child.visit_count for child in search.root.children.values()) == 8


def test_one_simulation_visits_one_root_edge() -> None:
    search = MCTS(CountingEvaluator(), simulations=1, c_puct=1.5, seed=3)

    policy = search.search(
        SearchState(Board.standard(), Color.RED), add_noise=False
    )

    assert sum(child.visit_count for child in search.root.children.values()) == 1
    assert sum(policy.values()) == pytest.approx(1.0)


def test_child_value_is_negated_to_its_parent_perspective() -> None:
    state = SearchState(Board.standard(), Color.RED)
    search = MCTS(CountingEvaluator(), simulations=2, c_puct=1.5, seed=3)

    search.search(state, add_noise=False)

    assert any(child.value_sum < 0 for child in search.root.children.values())


def test_real_move_advances_board_and_changes_side() -> None:
    state = SearchState(Board.standard(), Color.RED)
    move = all_legal_moves(state.board, state.side)[0]

    next_state = state.play(move)

    assert next_state.side is Color.BLACK
    assert next_state.board.at(move.start) is None
    assert next_state.board.at(move.end) == move.piece
    assert state.board.at(move.start) == move.piece


def test_terminal_state_does_not_call_evaluator() -> None:
    evaluator = CountingEvaluator()
    terminal = SearchState(
        Board.from_fen("4k4/3RRR3/9/9/9/9/9/9/9/4K4"), Color.BLACK
    )

    search = MCTS(evaluator, simulations=4, c_puct=1.5, seed=3)
    policy = search.search(terminal, add_noise=False)

    assert policy == {}
    assert evaluator.calls == 0
    assert search.root.value_sum == -4.0


def test_backpropagation_alternates_value_perspective_at_every_level() -> None:
    search = MCTS(CountingEvaluator(), simulations=1, c_puct=1.5, seed=3)
    first_edge = Node(prior=1.0)
    second_edge = Node(prior=1.0)

    search._backpropagate([first_edge, second_edge], leaf_value=0.4)

    assert second_edge.value_sum == pytest.approx(-0.4)
    assert first_edge.value_sum == pytest.approx(0.4)


def test_illegal_high_logit_is_excluded_before_root_prior_softmax() -> None:
    state = SearchState(Board.standard(), Color.RED)
    legal_moves = all_legal_moves(state.board, state.side)

    class IllegalBiasedEvaluator:
        def evaluate(self, state: SearchState) -> tuple[np.ndarray, float]:
            logits = np.zeros(ACTION_SIZE, dtype=np.float32)
            legal_indices = {
                encode_action(move, state.side)
                for move in all_legal_moves(state.board, state.side)
            }
            illegal_index = next(
                index for index in range(ACTION_SIZE) if index not in legal_indices
            )
            logits[illegal_index] = 1000.0
            return logits, 0.0

    search = MCTS(
        IllegalBiasedEvaluator(), simulations=1, c_puct=1.5, seed=3
    )

    search.search(state, add_noise=False)

    assert set(search.root.children) == set(legal_moves)
    assert all(
        child.prior == pytest.approx(1 / len(legal_moves))
        for child in search.root.children.values()
    )


def test_root_dirichlet_noise_is_used_only_when_requested() -> None:
    state = SearchState(Board.standard(), Color.RED)
    without_noise = MCTS(
        CountingEvaluator(), simulations=1, c_puct=1.5, seed=3
    )
    with_noise = MCTS(
        CountingEvaluator(),
        simulations=1,
        c_puct=1.5,
        seed=3,
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
    )

    without_noise.search(state, add_noise=False)
    with_noise.search(state, add_noise=True)

    plain_priors = [child.prior for child in without_noise.root.children.values()]
    noisy_priors = [child.prior for child in with_noise.root.children.values()]
    assert len(set(plain_priors)) == 1
    assert noisy_priors != pytest.approx(plain_priors)
    assert sum(noisy_priors) == pytest.approx(1.0)


def test_seed_reproduces_noise_and_visit_policy() -> None:
    state = SearchState(Board.standard(), Color.RED)

    first = MCTS(CountingEvaluator(), simulations=12, c_puct=1.5, seed=17)
    second = MCTS(CountingEvaluator(), simulations=12, c_puct=1.5, seed=17)

    first_policy = first.search(state, add_noise=True)
    second_policy = second.search(state, add_noise=True)

    assert first_policy == second_policy
    assert [child.prior for child in first.root.children.values()] == pytest.approx(
        [child.prior for child in second.root.children.values()]
    )
