import pytest

from xiangqi.board import Board
from xiangqi.domain import Color, Coord, Move, Piece, PieceType


def test_coord_rejects_outside_board() -> None:
    for file, rank in [(-1, 0), (9, 0), (0, -1), (0, 10)]:
        try:
            Coord(file, rank)
        except ValueError:
            pass
        else:
            raise AssertionError("越界坐标必须被拒绝")


def test_move_is_serializable_without_ui_types() -> None:
    move = Move(
        start=Coord(1, 7),
        end=Coord(1, 0),
        piece=Piece(Color.RED, PieceType.CANNON),
    )
    assert move.to_dict()["start"] == [1, 7]


def test_standard_board_has_32_pieces_and_generals() -> None:
    board = Board.standard()
    assert len(board.pieces) == 32
    assert board.at(Coord(4, 9)).kind is PieceType.GENERAL
    assert board.at(Coord(4, 9)).color is Color.RED
    assert board.at(Coord(4, 0)).color is Color.BLACK


def test_apply_returns_new_board_and_preserves_original() -> None:
    board = Board.standard()
    next_board = board.move_unchecked(Coord(0, 6), Coord(0, 5))
    assert board.at(Coord(0, 6)) is not None
    assert next_board.at(Coord(0, 6)) is None
    assert next_board.at(Coord(0, 5)).kind is PieceType.PAWN


def test_board_operations_are_persistent_and_pieces_are_read_only() -> None:
    pawn = Piece(Color.RED, PieceType.PAWN)
    empty = Board.empty()
    placed = empty.place(Coord(4, 6), pawn)
    removed = placed.remove(Coord(4, 6))

    assert empty.at(Coord(4, 6)) is None
    assert placed.at(Coord(4, 6)) == pawn
    assert removed.at(Coord(4, 6)) is None
    with pytest.raises(TypeError):
        placed.pieces[Coord(0, 0)] = pawn


def test_standard_board_fen_round_trip() -> None:
    board = Board.standard()
    fen = board.to_fen()
    assert fen == "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR"
    assert Board.from_fen(fen).pieces == board.pieces


def test_position_key_is_deterministic_and_includes_side_to_move() -> None:
    board = Board.standard()
    same_position = Board.from_fen(board.to_fen())

    assert board.position_key(Color.RED) == same_position.position_key(Color.RED)
    assert board.position_key(Color.RED) != board.position_key(Color.BLACK)
