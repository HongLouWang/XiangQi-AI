from __future__ import annotations

import pytest

from xiangqi.adjudication import (
    AdjudicationKind,
    Asian2003Adjudicator,
    Chinese2020Adjudicator,
    MoveNature,
    PositionFrame,
    Ruleset,
)
from xiangqi.board import Board
from xiangqi.domain import Color, Coord, Move, Piece, PieceType


def _piece(color: Color, kind: PieceType) -> Piece:
    return Piece(color, kind)


def _play(
    board: Board,
    side: Color,
    start: tuple[int, int],
    end: tuple[int, int],
) -> tuple[Board, PositionFrame]:
    start_coord = Coord(*start)
    end_coord = Coord(*end)
    piece = board.at(start_coord)
    assert piece is not None
    move = Move(start_coord, end_coord, piece, board.at(end_coord))
    after = board.move_unchecked(start_coord, end_coord)
    return after, PositionFrame.from_transition(board, side, move, after)


def _perpetual_check_history() -> list[PositionFrame]:
    board = (
        Board.empty()
        .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
        .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
        .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
        .place(Coord(3, 1), _piece(Color.RED, PieceType.ROOK))
    )
    history: list[PositionFrame] = []
    side = Color.RED
    cycle = (
        ((3, 1), (4, 1)),
        ((4, 0), (3, 0)),
        ((4, 1), (3, 1)),
        ((3, 0), (4, 0)),
    )
    for _ in range(2):
        for start, end in cycle:
            board, frame = _play(board, side, start, end)
            history.append(frame)
            side = side.opponent
    return history


def _ordinary_repetition_history() -> list[PositionFrame]:
    board = (
        Board.empty()
        .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
        .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
        .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
        .place(Coord(0, 8), _piece(Color.RED, PieceType.ROOK))
        .place(Coord(8, 1), _piece(Color.BLACK, PieceType.ROOK))
    )
    history: list[PositionFrame] = []
    side = Color.RED
    cycle = (
        ((0, 8), (1, 8)),
        ((8, 1), (7, 1)),
        ((1, 8), (0, 8)),
        ((7, 1), (8, 1)),
    )
    for _ in range(2):
        for start, end in cycle:
            board, frame = _play(board, side, start, end)
            history.append(frame)
            side = side.opponent
    return history


def test_position_frame_classifies_real_check_and_idle_moves() -> None:
    check_history = _perpetual_check_history()
    idle_history = _ordinary_repetition_history()

    assert [frame.nature for frame in check_history[:2]] == [
        MoveNature.CHECK,
        MoveNature.IDLE,
    ]
    assert all(frame.nature is MoveNature.IDLE for frame in idle_history)


@pytest.mark.parametrize(
    "adjudicator",
    [Chinese2020Adjudicator(), Asian2003Adjudicator()],
)
def test_single_side_perpetual_check_must_change(adjudicator) -> None:
    result = adjudicator.evaluate(_perpetual_check_history())

    assert result.kind is AdjudicationKind.MUST_CHANGE
    assert result.cycle_start == 0
    assert result.responsible is Color.RED
    assert result.responsible_natures == (MoveNature.CHECK,) * 4
    assert result.ruleset is adjudicator.ruleset
    assert result.rule_reference


@pytest.mark.parametrize(
    ("adjudicator", "ruleset"),
    [
        (Chinese2020Adjudicator(), Ruleset.CHINESE_2020),
        (Asian2003Adjudicator(), Ruleset.ASIAN_2003),
    ],
)
def test_threefold_ordinary_repetition_is_draw(adjudicator, ruleset) -> None:
    result = adjudicator.evaluate(_ordinary_repetition_history())

    assert result.kind is AdjudicationKind.DRAW
    assert result.cycle_start == 0
    assert result.responsible is None
    assert result.ruleset is ruleset
    assert set(result.move_natures) == {MoveNature.IDLE}


@pytest.mark.parametrize(
    "adjudicator",
    [Chinese2020Adjudicator(), Asian2003Adjudicator()],
)
def test_two_occurrences_are_not_enough_for_a_decision(adjudicator) -> None:
    history = _ordinary_repetition_history()[:4]

    result = adjudicator.evaluate(history)

    assert result.kind is AdjudicationKind.NO_DECISION
    assert result.cycle_start is None
