from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from xiangqi.board import Board
from xiangqi.domain import Color, Coord, Move, PieceType

BOARD_FILES = 9
BOARD_RANKS = 10
BOARD_SQUARES = BOARD_FILES * BOARD_RANKS
ACTION_SIZE = BOARD_SQUARES * BOARD_SQUARES
INPUT_CHANNELS = 15

_PIECE_TYPES = tuple(PieceType)


def _require_side(side: Color) -> None:
    if not isinstance(side, Color):
        raise TypeError("side 必须是 Color")


def _view(coord: Coord, side: Color) -> Coord:
    if side is Color.RED:
        return coord
    return Coord(BOARD_FILES - 1 - coord.file, BOARD_RANKS - 1 - coord.rank)


def _square_index(coord: Coord) -> int:
    return coord.rank * BOARD_FILES + coord.file


def encode_board(board: Board, side: Color) -> NDArray[np.float32]:
    """从当前行棋方视角编码局面，前七个平面始终表示己方棋子。"""
    if not isinstance(board, Board):
        raise TypeError("board 必须是 Board")
    _require_side(side)

    encoded = np.zeros(
        (INPUT_CHANNELS, BOARD_RANKS, BOARD_FILES), dtype=np.float32
    )
    for coord, piece in board.pieces.items():
        viewed = _view(coord, side)
        owner_offset = 0 if piece.color is side else len(_PIECE_TYPES)
        channel = owner_offset + _PIECE_TYPES.index(piece.kind)
        encoded[channel, viewed.rank, viewed.file] = 1.0

    if side is Color.RED:
        encoded[14].fill(1.0)
    return encoded


def encode_action(move: Move, side: Color) -> int:
    """将实际棋盘坐标中的着法映射到当前方视角的固定动作空间。"""
    if not isinstance(move, Move):
        raise TypeError("move 必须是 Move")
    _require_side(side)
    if move.piece.color is not side:
        raise ValueError("动作棋子不属于当前行棋方")

    start = _square_index(_view(move.start, side))
    end = _square_index(_view(move.end, side))
    return start * BOARD_SQUARES + end


def decode_action(index: int, board: Board, side: Color) -> Move:
    """将当前方视角动作索引还原为实际棋盘坐标中的着法。"""
    if not isinstance(index, Integral) or isinstance(index, (bool, np.bool_)):
        raise TypeError("动作索引必须是整数")
    if not 0 <= index < ACTION_SIZE:
        raise ValueError(f"动作索引越界: {index}")
    if not isinstance(board, Board):
        raise TypeError("board 必须是 Board")
    _require_side(side)

    start_index, end_index = divmod(int(index), BOARD_SQUARES)
    viewed_start = Coord(start_index % BOARD_FILES, start_index // BOARD_FILES)
    viewed_end = Coord(end_index % BOARD_FILES, end_index // BOARD_FILES)
    start = _view(viewed_start, side)
    end = _view(viewed_end, side)
    piece = board.at(start)
    if piece is None:
        raise ValueError("动作起点没有棋子")
    if piece.color is not side:
        raise ValueError("动作起点棋子不属于当前行棋方")
    return Move(start=start, end=end, piece=piece, captured=board.at(end))


def legal_policy(
    logits: NDArray[np.floating], moves: Sequence[Move], side: Color
) -> NDArray[np.float32]:
    """只在当前方合法动作上计算 softmax，并将其放回完整动作空间。"""
    _require_side(side)
    values = np.asarray(logits)
    if values.shape != (ACTION_SIZE,):
        raise ValueError(f"logits 形状必须是 ({ACTION_SIZE},)")
    if not moves:
        raise ValueError("无合法走法时不能生成策略")

    indices = np.asarray([encode_action(move, side) for move in moves])
    if np.unique(indices).size != indices.size:
        raise ValueError("合法走法包含重复动作")
    selected = values[indices].astype(np.float64)
    if not np.all(np.isfinite(selected)):
        raise ValueError("合法动作的 logits 必须是有限值")
    selected = np.exp(selected - selected.max())

    result = np.zeros(ACTION_SIZE, dtype=np.float32)
    result[indices] = selected / selected.sum()
    return result
