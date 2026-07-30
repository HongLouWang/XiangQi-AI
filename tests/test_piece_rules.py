import pytest

from xiangqi.board import Board
from xiangqi.domain import Color, Coord, Piece, PieceType
from xiangqi.rules import pseudo_legal_destinations


def destinations(board: Board, start: Coord) -> set[Coord]:
    return set(pseudo_legal_destinations(board, start))


@pytest.mark.parametrize(
    ("piece", "start", "legal", "illegal"),
    [
        (Piece(Color.RED, PieceType.ROOK), Coord(4, 5), Coord(4, 1), Coord(5, 4)),
        (Piece(Color.RED, PieceType.HORSE), Coord(4, 5), Coord(6, 4), Coord(4, 3)),
        (Piece(Color.RED, PieceType.ELEPHANT), Coord(4, 9), Coord(2, 7), Coord(6, 5)),
        (Piece(Color.RED, PieceType.ADVISOR), Coord(4, 9), Coord(3, 8), Coord(2, 7)),
        (Piece(Color.RED, PieceType.GENERAL), Coord(4, 9), Coord(4, 8), Coord(4, 7)),
        (Piece(Color.RED, PieceType.CANNON), Coord(1, 7), Coord(1, 4), Coord(2, 6)),
        (Piece(Color.RED, PieceType.PAWN), Coord(4, 6), Coord(4, 5), Coord(3, 6)),
    ],
)
def test_piece_pseudo_moves(
    piece: Piece, start: Coord, legal: Coord, illegal: Coord
) -> None:
    board = Board.empty().place(start, piece)

    moves = destinations(board, start)

    assert legal in moves
    assert illegal not in moves


def test_destination_with_friendly_piece_is_excluded_and_enemy_is_capturable() -> None:
    red_rook = Piece(Color.RED, PieceType.ROOK)
    board = (
        Board.empty()
        .place(Coord(4, 5), red_rook)
        .place(Coord(4, 3), Piece(Color.RED, PieceType.PAWN))
        .place(Coord(4, 7), Piece(Color.BLACK, PieceType.PAWN))
    )

    moves = destinations(board, Coord(4, 5))

    assert Coord(4, 3) not in moves
    assert Coord(4, 7) in moves


def test_rook_cannot_move_through_any_piece() -> None:
    board = (
        Board.empty()
        .place(Coord(4, 5), Piece(Color.RED, PieceType.ROOK))
        .place(Coord(4, 3), Piece(Color.BLACK, PieceType.PAWN))
        .place(Coord(6, 5), Piece(Color.RED, PieceType.PAWN))
    )

    moves = destinations(board, Coord(4, 5))

    assert Coord(4, 4) in moves
    assert Coord(4, 3) in moves
    assert Coord(4, 2) not in moves
    assert Coord(5, 5) in moves
    assert Coord(6, 5) not in moves
    assert Coord(7, 5) not in moves


@pytest.mark.parametrize(
    ("leg", "blocked_destinations"),
    [
        (Coord(4, 4), {Coord(3, 3), Coord(5, 3)}),
        (Coord(5, 5), {Coord(6, 4), Coord(6, 6)}),
    ],
)
def test_horse_cannot_jump_over_its_leg(
    leg: Coord, blocked_destinations: set[Coord]
) -> None:
    board = (
        Board.empty()
        .place(Coord(4, 5), Piece(Color.RED, PieceType.HORSE))
        .place(leg, Piece(Color.BLACK, PieceType.PAWN))
    )

    moves = destinations(board, Coord(4, 5))

    assert moves.isdisjoint(blocked_destinations)


def test_elephant_cannot_cross_river_or_jump_over_eye() -> None:
    elephant = Piece(Color.RED, PieceType.ELEPHANT)
    blocked = (
        Board.empty()
        .place(Coord(4, 9), elephant)
        .place(Coord(3, 8), Piece(Color.BLACK, PieceType.PAWN))
    )
    river_edge = Board.empty().place(Coord(2, 7), elephant)

    assert Coord(2, 7) not in destinations(blocked, Coord(4, 9))
    assert Coord(4, 5) in destinations(river_edge, Coord(2, 7))
    assert Coord(0, 5) in destinations(river_edge, Coord(2, 7))
    assert all(coord.rank >= 5 for coord in destinations(river_edge, Coord(2, 7)))


@pytest.mark.parametrize(
    ("color", "start", "inside", "outside"),
    [
        (Color.RED, Coord(3, 9), Coord(4, 8), Coord(2, 8)),
        (Color.BLACK, Coord(5, 2), Coord(4, 1), Coord(6, 1)),
    ],
)
def test_advisor_stays_in_own_palace(
    color: Color, start: Coord, inside: Coord, outside: Coord
) -> None:
    board = Board.empty().place(start, Piece(color, PieceType.ADVISOR))

    moves = destinations(board, start)

    assert inside in moves
    assert outside not in moves


@pytest.mark.parametrize(
    ("color", "start", "inside", "outside"),
    [
        (Color.RED, Coord(3, 8), Coord(3, 7), Coord(2, 8)),
        (Color.BLACK, Coord(5, 1), Coord(5, 2), Coord(6, 1)),
    ],
)
def test_general_moves_one_step_orthogonally_inside_own_palace(
    color: Color, start: Coord, inside: Coord, outside: Coord
) -> None:
    board = Board.empty().place(start, Piece(color, PieceType.GENERAL))

    moves = destinations(board, start)

    assert inside in moves
    assert outside not in moves


def test_cannon_moves_without_screen_and_captures_over_exactly_one_screen() -> None:
    board = (
        Board.empty()
        .place(Coord(4, 7), Piece(Color.RED, PieceType.CANNON))
        .place(Coord(4, 5), Piece(Color.RED, PieceType.PAWN))
        .place(Coord(4, 3), Piece(Color.BLACK, PieceType.ROOK))
        .place(Coord(4, 2), Piece(Color.BLACK, PieceType.HORSE))
    )

    moves = destinations(board, Coord(4, 7))

    assert Coord(4, 6) in moves
    assert Coord(4, 5) not in moves
    assert Coord(4, 4) not in moves
    assert Coord(4, 3) in moves
    assert Coord(4, 2) not in moves


@pytest.mark.parametrize(
    ("color", "before", "forward", "sideways", "backward", "after"),
    [
        (
            Color.RED,
            Coord(4, 6),
            Coord(4, 5),
            Coord(3, 6),
            Coord(4, 7),
            Coord(4, 4),
        ),
        (
            Color.BLACK,
            Coord(4, 3),
            Coord(4, 4),
            Coord(3, 3),
            Coord(4, 2),
            Coord(4, 5),
        ),
    ],
)
def test_pawn_gains_sideways_moves_only_after_crossing_river(
    color: Color,
    before: Coord,
    forward: Coord,
    sideways: Coord,
    backward: Coord,
    after: Coord,
) -> None:
    pawn = Piece(color, PieceType.PAWN)

    before_moves = destinations(Board.empty().place(before, pawn), before)
    after_moves = destinations(Board.empty().place(after, pawn), after)

    assert forward in before_moves
    assert sideways not in before_moves
    assert backward not in before_moves
    assert Coord(after.file - 1, after.rank) in after_moves
    assert Coord(after.file + 1, after.rank) in after_moves
    assert before not in after_moves


def test_empty_start_has_no_pseudo_legal_destinations() -> None:
    assert destinations(Board.empty(), Coord(4, 5)) == set()
