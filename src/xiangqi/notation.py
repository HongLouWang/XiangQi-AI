"""Chinese descriptive notation formatting and replay."""

from __future__ import annotations

from dataclasses import dataclass

from xiangqi.board import Board
from xiangqi.domain import Color, Move, PieceType
from xiangqi.rules import all_legal_moves, legal_destinations


class NotationError(ValueError):
    """A notation line cannot be mapped to exactly one legal move."""


_PIECE_NAMES = {
    Color.RED: {
        PieceType.GENERAL: "帅",
        PieceType.ADVISOR: "仕",
        PieceType.ELEPHANT: "相",
        PieceType.HORSE: "马",
        PieceType.ROOK: "车",
        PieceType.CANNON: "炮",
        PieceType.PAWN: "兵",
    },
    Color.BLACK: {
        PieceType.GENERAL: "将",
        PieceType.ADVISOR: "士",
        PieceType.ELEPHANT: "象",
        PieceType.HORSE: "马",
        PieceType.ROOK: "车",
        PieceType.CANNON: "炮",
        PieceType.PAWN: "卒",
    },
}
_RED_NUMBERS = "一二三四五六七八九"
_BLACK_NUMBERS = "１２３４５６７８９"


def _file_number(color: Color, file: int) -> int:
    # Internal files are numbered from Red's right to left.
    return file + 1 if color is Color.RED else 9 - file


def _number(color: Color, value: int) -> str:
    return (_RED_NUMBERS if color is Color.RED else _BLACK_NUMBERS)[value - 1]


def _prefix(board: Board, move: Move) -> str:
    name = _PIECE_NAMES[move.piece.color][move.piece.kind]
    peers = sorted(
        (
            coord
            for coord, piece in board.pieces.items()
            if piece == move.piece and coord.file == move.start.file
        ),
        key=lambda coord: coord.rank,
        reverse=move.piece.color is Color.BLACK,
    )
    if len(peers) == 1:
        return name + _number(
            move.piece.color, _file_number(move.piece.color, move.start.file)
        )
    labels_by_count = {
        2: ("前", "后"),
        3: ("前", "中", "后"),
        4: ("前", "二", "三", "后"),
        5: ("前", "二", "三", "四", "后"),
    }
    if len(peers) not in labels_by_count:
        raise NotationError("同一路同类棋子数量无法使用标准纵线记谱")
    labels = labels_by_count[len(peers)]
    return labels[peers.index(move.start)] + name


def format_move(board: Board, move: Move) -> str:
    """Format one move using standard Chinese vertical-line notation."""
    if board.at(move.start) != move.piece or board.at(move.end) != move.captured:
        raise NotationError("着法与走前局面不一致")
    if move.end not in legal_destinations(board, move.start, move.piece.color):
        raise NotationError("不能为非法着法生成棋谱")

    prefix = _prefix(board, move)
    if move.start.rank == move.end.rank:
        action = "平"
        operand = _file_number(move.piece.color, move.end.file)
    else:
        forward = (
            move.end.rank < move.start.rank
            if move.piece.color is Color.RED
            else move.end.rank > move.start.rank
        )
        action = "进" if forward else "退"
        if move.piece.kind in (
            PieceType.HORSE,
            PieceType.ELEPHANT,
            PieceType.ADVISOR,
        ):
            operand = _file_number(move.piece.color, move.end.file)
        else:
            operand = abs(move.end.rank - move.start.rank)
    return prefix + action + _number(move.piece.color, operand)


@dataclass(frozen=True, slots=True)
class NotatedMove:
    move: Move
    notation: str
    position_after: str


@dataclass(frozen=True, slots=True)
class ReplayResult:
    board: Board
    side_to_move: Color
    moves: tuple[NotatedMove, ...]


def replay_text(
    text: str,
    *,
    initial_board: Board | None = None,
    side_to_move: Color = Color.RED,
) -> ReplayResult:
    """Replay text transactionally; no caller-owned state is ever mutated."""
    board = initial_board or Board.standard()
    side = side_to_move
    replayed: list[NotatedMove] = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        notation = raw_line.strip()
        if not notation:
            continue
        matches = [
            move
            for move in all_legal_moves(board, side)
            if format_move(board, move) == notation
        ]
        if not matches:
            raise NotationError(f"第 {line_number} 行不是当前局面的合法着法: {notation}")
        if len(matches) > 1:
            raise NotationError(f"第 {line_number} 行着法有歧义: {notation}")
        move = matches[0]
        board = board.move_unchecked(move.start, move.end)
        side = side.opponent
        replayed.append(
            NotatedMove(move, notation, board.position_key(side))
        )
    return ReplayResult(board, side, tuple(replayed))
