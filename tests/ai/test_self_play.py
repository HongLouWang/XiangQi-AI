from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from ai.mcts import SearchState
from ai.self_play import GameResult, TrainingSample, play_game
from xiangqi.board import Board
from xiangqi.domain import Color, PositionKind, PositionResult
from xiangqi.rules import all_legal_moves


@dataclass(frozen=True)
class FakeState:
    ply: int = 0


class LoopingAdapter:
    def __init__(self, *, terminal_ply: int | None = None, winner: Color | None = None):
        self.terminal_ply = terminal_ply
        self.winner = winner
        self.play_calls = 0
        self.position_calls: list[int] = []

    def initial_state(self) -> FakeState:
        return FakeState()

    def side(self, state: FakeState) -> Color:
        return Color.RED if state.ply % 2 == 0 else Color.BLACK

    def position(self, state: FakeState) -> PositionResult:
        self.position_calls.append(state.ply)
        terminal = self.terminal_ply is not None and state.ply >= self.terminal_ply
        return PositionResult(
            kind=PositionKind.CHECKMATE if terminal else PositionKind.ONGOING,
            side_to_move=self.side(state),
            winner=self.winner if terminal else None,
            in_check=terminal,
        )

    def encode_state(self, state: FakeState) -> np.ndarray:
        return np.full((1, 1, 1), state.ply, dtype=np.float32)

    def encode_action(self, state: FakeState, move: int) -> int:
        return move + 100

    def legal_moves(self, state: FakeState) -> tuple[int, ...]:
        return (0, 1)

    def play(self, state: FakeState, move: int) -> FakeState:
        assert state.ply < 1024, "不得请求第 1025 个 ply"
        self.play_calls += 1
        return FakeState(state.ply + 1)


class FixedSearch:
    def __init__(self, policy: dict[int, float] | None = None):
        self.policy = {0: 0.75, 1: 0.25} if policy is None else policy
        self.calls: list[tuple[int, bool]] = []

    def search(self, state: FakeState, *, add_noise: bool) -> dict[int, float]:
        self.calls.append((state.ply, add_noise))
        return self.policy


def test_512_full_moves_means_exactly_1024_plies_without_requesting_more() -> None:
    adapter = LoopingAdapter()
    search = FixedSearch({0: 1.0})

    result = play_game(search, max_plies=512 * 2, adapter=adapter)

    assert result.plies == 1024
    assert result.termination == "move_limit"
    assert result.winner is None
    assert adapter.play_calls == 1024
    assert len(search.calls) == 1024
    assert adapter.position_calls == list(range(1025))
    assert all(sample.value == 0.0 for sample in result.samples)


def test_terminal_result_on_last_allowed_ply_precedes_move_limit_draw() -> None:
    adapter = LoopingAdapter(terminal_ply=2, winner=Color.RED)
    search = FixedSearch({0: 1.0})

    result = play_game(search, max_plies=2, adapter=adapter)

    assert result.plies == 2
    assert result.termination == "checkmate"
    assert result.winner is Color.RED
    assert adapter.play_calls == 2
    assert len(search.calls) == 2
    assert adapter.position_calls == [0, 1, 2]
    assert [sample.value for sample in result.samples] == [1.0, -1.0]


def test_scripted_winner_is_converted_to_each_sample_side() -> None:
    adapter = LoopingAdapter(terminal_ply=2, winner=Color.RED)

    result = play_game(FixedSearch(), adapter=adapter)

    assert result.winner is Color.RED
    assert result.termination == PositionKind.CHECKMATE.value
    assert [sample.side for sample in result.samples] == [Color.RED, Color.BLACK]
    assert [sample.value for sample in result.samples] == [1.0, -1.0]


def test_policy_is_stored_as_sparse_encoded_indices_and_probabilities() -> None:
    adapter = LoopingAdapter(terminal_ply=1, winner=Color.RED)

    sample = play_game(FixedSearch({0: 0.25, 1: 0.75}), adapter=adapter).samples[0]

    np.testing.assert_array_equal(sample.policy_indices, [100, 101])
    np.testing.assert_allclose(sample.policy_probabilities, [0.25, 0.75])
    assert sample.policy_indices.dtype == np.int64
    assert sample.policy_probabilities.dtype == np.float32


def test_temperature_samples_opening_then_chooses_maximum_deterministically() -> None:
    adapter = LoopingAdapter(terminal_ply=5, winner=Color.BLACK)
    played: list[int] = []
    original_play = adapter.play

    def recording_play(state: FakeState, move: int) -> FakeState:
        played.append(move)
        return original_play(state, move)

    adapter.play = recording_play  # type: ignore[method-assign]
    play_game(
        FixedSearch({0: 0.01, 1: 0.99}),
        adapter=adapter,
        temperature_plies=2,
        seed=7,
    )

    assert played[2:] == [1, 1, 1]


def test_same_seed_reproduces_opening_samples() -> None:
    def moves_for(seed: int) -> list[int]:
        adapter = LoopingAdapter(terminal_ply=20, winner=Color.RED)
        moves: list[int] = []
        original_play = adapter.play

        def recording_play(state: FakeState, move: int) -> FakeState:
            moves.append(move)
            return original_play(state, move)

        adapter.play = recording_play  # type: ignore[method-assign]
        play_game(FixedSearch({0: 0.5, 1: 0.5}), adapter=adapter, seed=123)
        return moves

    assert moves_for(123) == moves_for(123)


@pytest.mark.parametrize(
    "policy",
    [
        {},
        {0: -0.1, 1: 1.1},
        {0: float("nan"), 1: 1.0},
        {0: 0.0, 1: 0.0},
        {2: 1.0},
    ],
)
def test_empty_invalid_or_illegal_policy_is_rejected(policy: dict[int, float]) -> None:
    with pytest.raises(ValueError):
        play_game(FixedSearch(policy), max_plies=1, adapter=LoopingAdapter())


@pytest.mark.parametrize("max_plies", [0, -1, 1.5, True])
def test_max_plies_must_be_a_positive_integer(max_plies: object) -> None:
    with pytest.raises(ValueError, match="max_plies"):
        play_game(FixedSearch(), max_plies=max_plies)  # type: ignore[arg-type]


def test_default_adapter_uses_real_board_and_rules_for_terminal_position() -> None:
    terminal = SearchState(
        Board.from_fen("4k4/3RRR3/9/9/9/9/9/9/9/4K4"), Color.BLACK
    )

    result = play_game(FixedSearch(), initial_state=terminal)

    assert result == GameResult(
        samples=(), winner=Color.RED, plies=0, termination="checkmate"
    )


def test_real_rules_fill_winner_value_after_a_checkmating_move() -> None:
    state = SearchState(
        Board.from_fen("4k4/4RR3/3R5/9/9/9/9/9/9/4K4"), Color.RED
    )
    move = next(
        move
        for move in all_legal_moves(state.board, state.side)
        if (move.start.file, move.start.rank, move.end.file, move.end.rank)
        == (3, 2, 3, 1)
    )

    class CheckmatingSearch:
        def search(
            self, state: SearchState, *, add_noise: bool
        ) -> dict[object, float]:
            return {move: 1.0}

    result = play_game(CheckmatingSearch(), initial_state=state)

    assert result.winner is Color.RED
    assert result.termination == "checkmate"
    assert result.plies == 1
    assert len(result.samples) == 1
    assert result.samples[0].side is Color.RED
    assert result.samples[0].value == 1.0


def test_result_and_samples_are_immutable_value_objects() -> None:
    assert TrainingSample.__dataclass_params__.frozen
    assert GameResult.__dataclass_params__.frozen
