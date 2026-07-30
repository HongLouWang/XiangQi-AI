from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, Qt

from xiangqi.controller import ControllerKind, GameController
from xiangqi.domain import Color, Coord
from xiangqi.ui.board_widget import BoardWidget


@pytest.fixture
def controller() -> GameController:
    return GameController.new()


@pytest.fixture
def widget(qtbot, controller: GameController) -> BoardWidget:
    board = BoardWidget(controller)
    board.resize(720, 760)
    qtbot.addWidget(board)
    board.show()
    qtbot.waitExposed(board)
    return board


def click_coord(qtbot, widget: BoardWidget, coord: Coord) -> None:
    qtbot.mouseClick(
        widget,
        Qt.MouseButton.LeftButton,
        pos=widget.point_for(coord),
    )


def test_point_mapping_is_fixed_red_bottom_and_scales(
    widget: BoardWidget,
) -> None:
    red_general = widget.point_for(Coord(4, 9))
    black_general = widget.point_for(Coord(4, 0))
    assert red_general.y() > black_general.y()
    assert widget.coord_at(red_general) == Coord(4, 9)
    assert widget.coord_at(black_general) == Coord(4, 0)

    old_spacing = widget.grid_spacing
    widget.resize(360, 440)
    assert widget.grid_spacing < old_spacing
    assert widget.coord_at(widget.point_for(Coord(8, 9))) == Coord(8, 9)


def test_edge_pieces_have_enough_margin_to_render_without_clipping(
    widget: BoardWidget,
) -> None:
    piece_radius = widget.grid_spacing * 0.34
    top_left = widget.point_for(Coord(0, 0))
    bottom_right = widget.point_for(Coord(8, 9))

    assert top_left.x() >= piece_radius
    assert top_left.y() >= piece_radius
    assert bottom_right.x() <= widget.width() - piece_radius
    assert bottom_right.y() <= widget.height() - piece_radius


def test_left_click_selects_piece_and_shows_all_legal_targets(
    qtbot,
    widget: BoardWidget,
) -> None:
    click_coord(qtbot, widget, Coord(0, 6))

    assert widget.selected == Coord(0, 6)
    assert widget.legal_targets == {Coord(0, 5)}


def test_clicking_selected_piece_or_meaningless_empty_space_cancels_selection(
    qtbot,
    widget: BoardWidget,
) -> None:
    click_coord(qtbot, widget, Coord(0, 6))
    click_coord(qtbot, widget, Coord(0, 6))
    assert widget.selected is None
    assert widget.legal_targets == set()

    click_coord(qtbot, widget, Coord(0, 6))
    click_coord(qtbot, widget, Coord(4, 5))
    assert widget.selected is None
    assert widget.legal_targets == set()


def test_clicking_illegal_target_does_not_move(
    qtbot,
    widget: BoardWidget,
    controller: GameController,
) -> None:
    click_coord(qtbot, widget, Coord(0, 6))
    click_coord(qtbot, widget, Coord(1, 5))

    state = controller.get_state()
    assert state.ply == 0
    assert state.board.at(Coord(0, 6)) is not None


def test_clicking_legal_target_moves_and_keeps_last_move_highlight(
    qtbot,
    widget: BoardWidget,
    controller: GameController,
) -> None:
    click_coord(qtbot, widget, Coord(0, 6))
    click_coord(qtbot, widget, Coord(0, 5))

    assert controller.get_state().ply == 1
    assert widget.selected is None
    assert widget.last_move == (Coord(0, 6), Coord(0, 5))
    assert widget.highlighted_piece == Coord(0, 5)


def test_opponent_last_piece_stays_highlighted_after_move(
    widget: BoardWidget,
    controller: GameController,
) -> None:
    controller.make_move(Coord(0, 6), Coord(0, 5))
    controller.make_move(Coord(0, 3), Coord(0, 4))

    assert widget.last_move == (Coord(0, 3), Coord(0, 4))
    assert widget.highlighted_piece == Coord(0, 4)


def test_selection_and_last_move_highlights_coexist_with_distinct_colors(
    qtbot,
    widget: BoardWidget,
    controller: GameController,
) -> None:
    controller.make_move(Coord(0, 6), Coord(0, 5))
    click_coord(qtbot, widget, Coord(0, 3))

    assert widget.selected == Coord(0, 3)
    assert widget.last_move == (Coord(0, 6), Coord(0, 5))
    assert widget.selection_color != widget.last_move_color


def test_external_controlled_side_cannot_be_moved_with_mouse(
    qtbot,
    widget: BoardWidget,
    controller: GameController,
) -> None:
    controller.claim_side(Color.RED, "test-bot", ControllerKind.PYTHON)
    click_coord(qtbot, widget, Coord(0, 6))

    assert widget.selected is None
    assert widget.legal_targets == set()
    assert controller.get_state().ply == 0


def test_undo_and_replay_cursor_keep_last_move_in_sync(
    widget: BoardWidget,
    controller: GameController,
) -> None:
    controller.make_move(Coord(0, 6), Coord(0, 5))
    controller.make_move(Coord(0, 3), Coord(0, 4))
    controller.undo()
    assert widget.last_move == (Coord(0, 6), Coord(0, 5))

    controller.make_move(Coord(2, 3), Coord(2, 4))
    controller.set_replay_cursor(0)
    assert widget.last_move is None

    controller.set_replay_cursor(1)
    assert widget.last_move == (Coord(0, 6), Coord(0, 5))


def test_click_outside_board_cancels_selection(
    qtbot,
    widget: BoardWidget,
) -> None:
    click_coord(qtbot, widget, Coord(0, 6))
    qtbot.mouseClick(
        widget,
        Qt.MouseButton.LeftButton,
        pos=QPoint(1, 1),
    )
    assert widget.selected is None
