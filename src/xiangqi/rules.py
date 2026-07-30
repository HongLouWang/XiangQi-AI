from __future__ import annotations

from collections.abc import Iterable, Iterator

from xiangqi.board import Board
from xiangqi.domain import Color, Coord, Piece, PieceType


_ORTHOGONAL_DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
_HORSE_STEPS = (
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
)
_DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _in_bounds(file: int, rank: int) -> bool:
    return 0 <= file < 9 and 0 <= rank < 10


def _coord_if_in_bounds(file: int, rank: int) -> Coord | None:
    if not _in_bounds(file, rank):
        return None
    return Coord(file, rank)


def _ray_destinations(board: Board, start: Coord) -> Iterator[Coord]:
    for file_step, rank_step in _ORTHOGONAL_DIRECTIONS:
        file = start.file + file_step
        rank = start.rank + rank_step
        while _in_bounds(file, rank):
            destination = Coord(file, rank)
            yield destination
            if board.at(destination) is not None:
                break
            file += file_step
            rank += rank_step


def _rook_destinations(board: Board, start: Coord) -> Iterable[Coord]:
    return _ray_destinations(board, start)


def _horse_destinations(board: Board, start: Coord) -> Iterator[Coord]:
    for file_step, rank_step in _HORSE_STEPS:
        destination = _coord_if_in_bounds(
            start.file + file_step, start.rank + rank_step
        )
        if destination is None:
            continue
        if abs(file_step) == 2:
            leg = Coord(start.file + file_step // 2, start.rank)
        else:
            leg = Coord(start.file, start.rank + rank_step // 2)
        if board.at(leg) is None:
            yield destination


def _elephant_destinations(
    board: Board, start: Coord, piece: Piece
) -> Iterator[Coord]:
    for file_direction, rank_direction in _DIAGONALS:
        destination = _coord_if_in_bounds(
            start.file + 2 * file_direction,
            start.rank + 2 * rank_direction,
        )
        if destination is None:
            continue
        if piece.color is Color.RED and destination.rank < 5:
            continue
        if piece.color is Color.BLACK and destination.rank > 4:
            continue
        eye = Coord(start.file + file_direction, start.rank + rank_direction)
        if board.at(eye) is None:
            yield destination


def _palace_contains(color: Color, coord: Coord) -> bool:
    ranks = range(7, 10) if color is Color.RED else range(0, 3)
    return 3 <= coord.file <= 5 and coord.rank in ranks


def _advisor_destinations(start: Coord, piece: Piece) -> Iterator[Coord]:
    for file_step, rank_step in _DIAGONALS:
        destination = _coord_if_in_bounds(
            start.file + file_step, start.rank + rank_step
        )
        if destination is not None and _palace_contains(piece.color, destination):
            yield destination


def _general_destinations(start: Coord, piece: Piece) -> Iterator[Coord]:
    for file_step, rank_step in _ORTHOGONAL_DIRECTIONS:
        destination = _coord_if_in_bounds(
            start.file + file_step, start.rank + rank_step
        )
        if destination is not None and _palace_contains(piece.color, destination):
            yield destination


def _cannon_destinations(board: Board, start: Coord) -> Iterator[Coord]:
    for file_step, rank_step in _ORTHOGONAL_DIRECTIONS:
        file = start.file + file_step
        rank = start.rank + rank_step
        screen_found = False
        while _in_bounds(file, rank):
            destination = Coord(file, rank)
            occupant = board.at(destination)
            if not screen_found:
                if occupant is None:
                    yield destination
                else:
                    screen_found = True
            elif occupant is not None:
                yield destination
                break
            file += file_step
            rank += rank_step


def _pawn_destinations(start: Coord, piece: Piece) -> Iterator[Coord]:
    forward = -1 if piece.color is Color.RED else 1
    destination = _coord_if_in_bounds(start.file, start.rank + forward)
    if destination is not None:
        yield destination

    crossed_river = (
        start.rank <= 4 if piece.color is Color.RED else start.rank >= 5
    )
    if crossed_river:
        for file_step in (-1, 1):
            destination = _coord_if_in_bounds(start.file + file_step, start.rank)
            if destination is not None:
                yield destination


def _candidate_destinations(
    board: Board, start: Coord, piece: Piece
) -> Iterable[Coord]:
    if piece.kind is PieceType.ROOK:
        return _rook_destinations(board, start)
    if piece.kind is PieceType.HORSE:
        return _horse_destinations(board, start)
    if piece.kind is PieceType.ELEPHANT:
        return _elephant_destinations(board, start, piece)
    if piece.kind is PieceType.ADVISOR:
        return _advisor_destinations(start, piece)
    if piece.kind is PieceType.GENERAL:
        return _general_destinations(start, piece)
    if piece.kind is PieceType.CANNON:
        return _cannon_destinations(board, start)
    return _pawn_destinations(start, piece)


def pseudo_legal_destinations(board: Board, start: Coord) -> tuple[Coord, ...]:
    """Return piece-rule moves without checking whether the mover remains in check."""
    piece = board.at(start)
    if piece is None:
        return ()

    return tuple(
        destination
        for destination in _candidate_destinations(board, start, piece)
        if (occupant := board.at(destination)) is None
        or occupant.color is not piece.color
    )
