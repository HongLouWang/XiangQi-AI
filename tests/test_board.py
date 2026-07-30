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
