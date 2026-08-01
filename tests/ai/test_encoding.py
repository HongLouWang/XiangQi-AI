import numpy as np
import pytest

from ai.encoding import (
    ACTION_SIZE,
    INPUT_CHANNELS,
    decode_action,
    encode_action,
    encode_board,
    legal_policy,
)
from xiangqi.board import Board
from xiangqi.domain import Color, Coord, Piece, PieceType
from xiangqi.rules import all_legal_moves


@pytest.mark.parametrize("side", [Color.RED, Color.BLACK])
def test_action_round_trip_for_every_standard_legal_move(side: Color) -> None:
    board = Board.standard()

    for move in all_legal_moves(board, side):
        assert decode_action(encode_action(move, side), board, side) == move


def test_black_view_rotates_coordinates_and_makes_black_friendly() -> None:
    board = (
        Board.empty()
        .place(Coord(1, 2), Piece(Color.BLACK, PieceType.HORSE))
        .place(Coord(4, 8), Piece(Color.RED, PieceType.ROOK))
    )

    encoded = encode_board(board, Color.BLACK)

    assert encoded.shape == (INPUT_CHANNELS, 10, 9)
    assert encoded.dtype == np.float32
    assert encoded[3, 7, 7] == 1.0
    assert encoded[11, 1, 4] == 1.0
    assert np.count_nonzero(encoded[:14]) == 2
    assert np.count_nonzero(encoded[14]) == 0


def test_red_view_keeps_coordinates_and_marks_side_plane() -> None:
    board = Board.empty().place(Coord(1, 2), Piece(Color.RED, PieceType.HORSE))

    encoded = encode_board(board, Color.RED)

    assert encoded[3, 2, 1] == 1.0
    assert np.all(encoded[14] == 1.0)


@pytest.mark.parametrize("side", [Color.RED, Color.BLACK])
def test_legal_policy_masks_illegal_actions_in_current_side_view(side: Color) -> None:
    board = Board.standard()
    moves = all_legal_moves(board, side)
    logits = np.linspace(-1.0, 1.0, ACTION_SIZE, dtype=np.float32)

    probabilities = legal_policy(logits, moves, side)
    legal_indices = {encode_action(move, side) for move in moves}

    assert np.isclose(probabilities.sum(), 1.0)
    assert set(np.flatnonzero(probabilities)) == legal_indices


@pytest.mark.parametrize("index", [-1, ACTION_SIZE])
def test_decode_action_rejects_out_of_range_index(index: int) -> None:
    with pytest.raises(ValueError, match="动作索引"):
        decode_action(index, Board.standard(), Color.RED)


def test_decode_action_rejects_non_integer_index() -> None:
    with pytest.raises(TypeError, match="整数"):
        decode_action(1.5, Board.standard(), Color.RED)  # type: ignore[arg-type]


def test_decode_action_rejects_empty_start_square() -> None:
    index = Coord(4, 4).rank * 9 + Coord(4, 4).file
    with pytest.raises(ValueError, match="起点没有棋子"):
        decode_action(index * 90, Board.standard(), Color.RED)


def test_legal_policy_rejects_wrong_logit_shape() -> None:
    moves = all_legal_moves(Board.standard(), Color.RED)
    with pytest.raises(ValueError, match="形状"):
        legal_policy(np.zeros((90, 90), dtype=np.float32), moves, Color.RED)


def test_legal_policy_rejects_empty_legal_moves() -> None:
    with pytest.raises(ValueError, match="无合法走法"):
        legal_policy(np.zeros(ACTION_SIZE, dtype=np.float32), (), Color.RED)


def test_side_arguments_must_be_color() -> None:
    board = Board.standard()
    move = all_legal_moves(board, Color.RED)[0]
    with pytest.raises(TypeError, match="side"):
        encode_board(board, "red")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="side"):
        encode_action(move, "red")  # type: ignore[arg-type]
