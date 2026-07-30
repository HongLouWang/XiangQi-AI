import pytest

from xiangqi.board import Board
from xiangqi.domain import Color, Coord, Move, Piece, PieceType
from xiangqi.notation import NotationError, format_move, replay_text


def _move(board: Board, start: Coord, end: Coord) -> Move:
    piece = board.at(start)
    assert piece is not None
    return Move(start, end, piece, board.at(end))


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (Coord(1, 7), Coord(4, 7), "炮二平五"),
        (Coord(1, 0), Coord(2, 2), "马８进７"),
    ],
)
def test_format_chinese_notation(start: Coord, end: Coord, expected: str) -> None:
    board = Board.standard()
    assert format_move(board, _move(board, start, end)) == expected


def test_formats_capture_and_replays_multiple_moves() -> None:
    text = "炮二进七\n马２进３\n炮二退一"
    replay = replay_text(text)

    assert [item.notation for item in replay.moves] == text.splitlines()
    assert replay.board.at(Coord(1, 1)) == Piece(Color.RED, PieceType.CANNON)
    assert replay.moves[0].move.captured == Piece(Color.BLACK, PieceType.HORSE)
    assert replay.side_to_move is Color.BLACK


def test_same_file_pieces_use_front_middle_rear_disambiguation() -> None:
    red_rook = Piece(Color.RED, PieceType.ROOK)
    board = (
        Board.empty()
        .place(Coord(4, 9), Piece(Color.RED, PieceType.GENERAL))
        .place(Coord(3, 0), Piece(Color.BLACK, PieceType.GENERAL))
        .place(Coord(0, 8), red_rook)
        .place(Coord(0, 6), red_rook)
        .place(Coord(0, 4), red_rook)
    )

    assert format_move(board, _move(board, Coord(0, 4), Coord(0, 3))) == "前车进一"
    assert format_move(board, _move(board, Coord(0, 6), Coord(1, 6))) == "中车平二"
    assert format_move(board, _move(board, Coord(0, 8), Coord(0, 7))) == "后车进一"


def test_parse_text_reports_exact_line_and_does_not_mutate_previous_replay() -> None:
    previous = replay_text("炮二平五")

    with pytest.raises(NotationError, match="第 2 行"):
        replay_text("炮二平五\n帅五进三")

    assert len(previous.moves) == 1
    assert previous.board.at(Coord(4, 7)) == Piece(Color.RED, PieceType.CANNON)


def test_blank_lines_are_ignored_but_report_physical_line_number() -> None:
    with pytest.raises(NotationError, match="第 3 行"):
        replay_text("炮二平五\n\n帅五进三")


@pytest.mark.parametrize(
    ("ranks", "expected"),
    [
        ((2, 4, 6, 8), ("前兵平二", "二兵平二", "三兵进一", "后兵进一")),
        (
            (0, 2, 4, 6, 8),
            ("前兵平二", "二兵平二", "三兵平二", "四兵进一", "后兵进一"),
        ),
    ],
)
def test_four_or_five_same_file_pawns_format_and_replay_exactly(
    ranks: tuple[int, ...], expected: tuple[str, ...]
) -> None:
    pawn = Piece(Color.RED, PieceType.PAWN)
    board = (
        Board.empty()
        .place(Coord(4, 9), Piece(Color.RED, PieceType.GENERAL))
        .place(Coord(3, 0), Piece(Color.BLACK, PieceType.GENERAL))
    )
    for rank in ranks:
        board = board.place(Coord(0, rank), pawn)

    destinations = tuple(
        Coord(1, rank) if rank <= 4 else Coord(0, rank - 1) for rank in ranks
    )
    actual = tuple(
        format_move(board, _move(board, Coord(0, rank), destination))
        for rank, destination in zip(ranks, destinations, strict=True)
    )

    assert actual == expected
    for notation, rank in zip(expected, ranks, strict=True):
        replay = replay_text(notation, initial_board=board)
        assert replay.moves[0].move.start == Coord(0, rank)


def test_same_file_black_pawns_are_ordered_from_black_players_view() -> None:
    pawn = Piece(Color.BLACK, PieceType.PAWN)
    ranks = (7, 5, 3, 1)
    board = (
        Board.empty()
        .place(Coord(4, 9), Piece(Color.RED, PieceType.GENERAL))
        .place(Coord(3, 0), Piece(Color.BLACK, PieceType.GENERAL))
    )
    for rank in ranks:
        board = board.place(Coord(0, rank), pawn)
    destinations = (Coord(1, 7), Coord(1, 5), Coord(0, 4), Coord(0, 2))

    actual = tuple(
        format_move(board, _move(board, Coord(0, rank), destination))
        for rank, destination in zip(ranks, destinations, strict=True)
    )

    assert actual == ("前卒平８", "二卒平８", "三卒进１", "后卒进１")
