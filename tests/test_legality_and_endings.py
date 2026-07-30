from xiangqi.board import Board
from xiangqi.domain import Color, Coord, PieceType, PositionKind
from xiangqi.rules import (
    all_legal_moves,
    evaluate_position,
    is_in_check,
    is_square_attacked,
    legal_destinations,
)


def test_rook_attack_is_detected_through_open_file() -> None:
    board = Board.from_fen("4k4/9/9/9/9/9/9/9/4R4/4K4")

    assert is_square_attacked(board, Coord(4, 0), Color.RED)
    assert is_in_check(board, Color.BLACK)


def test_flying_generals_are_in_check() -> None:
    board = Board.from_fen("4k4/9/9/9/9/9/9/9/9/4K4")

    assert is_in_check(board, Color.RED)
    assert is_in_check(board, Color.BLACK)


def test_exposing_flying_generals_is_illegal() -> None:
    board = Board.from_fen("4k4/9/9/9/4R4/9/9/9/9/4K4")

    moves = set(legal_destinations(board, Coord(4, 4), Color.RED))

    assert Coord(3, 4) not in moves
    assert Coord(5, 4) not in moves


def test_move_that_exposes_own_general_to_rook_is_illegal() -> None:
    board = Board.from_fen("3k5/9/9/9/4r4/9/4P4/9/9/4K4")

    moves = set(legal_destinations(board, Coord(4, 6), Color.RED))

    assert Coord(3, 6) not in moves


def test_side_in_check_must_answer_check() -> None:
    board = Board.from_fen("3k5/9/9/9/9/9/9/4r4/P8/4K4")

    assert legal_destinations(board, Coord(0, 8), Color.RED) == ()
    assert all(
        move.piece.kind is PieceType.GENERAL
        for move in all_legal_moves(board, Color.RED)
    )


def test_capturing_enemy_general_is_not_a_normal_legal_move() -> None:
    board = Board.from_fen("4k4/4R4/9/9/9/9/9/9/9/4K4")

    assert Coord(4, 0) not in legal_destinations(board, Coord(4, 1), Color.RED)


def test_checkmate_is_win_for_attacker() -> None:
    board = Board.from_fen("4k4/3RRR3/9/9/9/9/9/9/9/4K4")

    result = evaluate_position(board, Color.BLACK)

    assert result.kind is PositionKind.CHECKMATE
    assert result.winner is Color.RED
    assert result.in_check


def test_stalemate_is_loss_for_side_without_moves() -> None:
    board = Board.from_fen("4k4/3R1R3/4P4/9/9/9/9/9/9/4K4")

    result = evaluate_position(board, Color.BLACK)

    assert result.kind is PositionKind.STALEMATE
    assert result.winner is Color.RED
    assert not result.in_check


def test_ordinary_position_is_ongoing() -> None:
    result = evaluate_position(Board.standard(), Color.RED)

    assert result.kind is PositionKind.ONGOING
    assert result.winner is None
    assert not result.in_check


def test_legal_destinations_rejects_piece_of_other_side() -> None:
    board = Board.standard()

    assert legal_destinations(board, Coord(0, 0), Color.RED) == ()
