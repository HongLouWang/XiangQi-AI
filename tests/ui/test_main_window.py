from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest
from PySide6.QtCore import Qt

from xiangqi.adjudication import Ruleset
from xiangqi.app import DesktopRuntime
from xiangqi.controller import ControllerKind, GameController
from xiangqi.domain import Color, Coord
from xiangqi.record import PlayerRecord
from xiangqi.ui.dialogs import NewGameDialog
from xiangqi.ui.main_window import MainWindow


@pytest.fixture
def played_controller() -> GameController:
    controller = GameController.new()
    controller.make_move(Coord(0, 6), Coord(0, 5))
    controller.make_move(Coord(0, 3), Coord(0, 4))
    return controller


@pytest.fixture
def window(qtbot, played_controller: GameController) -> MainWindow:
    main = MainWindow(played_controller)
    qtbot.addWidget(main)
    main.show()
    return main


def test_new_game_keeps_player_colors_mutually_exclusive_and_swappable(
    qtbot,
) -> None:
    dialog = NewGameDialog()
    qtbot.addWidget(dialog)

    dialog.player1_color.setCurrentIndex(dialog.player1_color.findData(Color.BLACK))
    assert dialog.player2_color.currentData() == Color.RED
    assert dialog.can_accept()

    dialog.player2_color.blockSignals(True)
    dialog.player2_color.setCurrentIndex(dialog.player2_color.findData(Color.BLACK))
    dialog.player2_color.blockSignals(False)
    assert not dialog.can_accept()


def test_new_game_collects_names_controls_and_ruleset(qtbot) -> None:
    dialog = NewGameDialog()
    qtbot.addWidget(dialog)
    dialog.player1_name.setText("甲")
    dialog.player2_name.setText("乙")
    dialog.player1_control.setCurrentIndex(
        dialog.player1_control.findData(ControllerKind.PYTHON)
    )
    dialog.player2_control.setCurrentIndex(
        dialog.player2_control.findData(ControllerKind.NETWORK)
    )
    dialog.ruleset.setCurrentIndex(dialog.ruleset.findData(Ruleset.ASIAN_2003))

    settings = dialog.settings()

    assert settings.players == (
        PlayerRecord(name="甲", color=Color.RED, controller="python"),
        PlayerRecord(name="乙", color=Color.BLACK, controller="network"),
    )
    assert settings.ruleset is Ruleset.ASIAN_2003


def test_main_window_shows_fixed_board_players_moves_and_locked_ruleset(
    window: MainWindow,
) -> None:
    assert window.board_widget.point_for(Coord(4, 9)).y() > (
        window.board_widget.point_for(Coord(4, 0)).y()
    )
    assert window.red_player_label.text()
    assert window.black_player_label.text()
    assert window.move_list.count() == 2
    assert window.rule_display.currentData() == Ruleset.CHINESE_2020
    assert not window.rule_display.isEnabled()
    assert "轮到红方" in window.status_label.text()


def test_unlimited_undo_and_draw_response_is_opponent_only(
    window: MainWindow,
) -> None:
    window.undo()
    window.undo()
    assert window.controller.get_state().ply == 0

    window.offer_draw(Color.RED)
    with pytest.raises(ValueError, match="对方"):
        window.respond_draw(Color.RED, True)
    window.respond_draw(Color.BLACK, False)
    assert window.controller.get_state().result is None

    window.offer_draw(Color.RED)
    window.respond_draw(Color.BLACK, True)
    assert window.controller.get_state().result.status == "draw"
    window.undo()
    assert window.controller.get_state().result is None


def test_replay_navigation_does_not_modify_record(
    window: MainWindow,
) -> None:
    original = window.controller.record.model_copy(deep=True)

    window.enter_replay()
    window.replay_first()
    assert window.controller.get_state().replay_cursor == 0
    window.replay_next()
    assert window.controller.get_state().replay_cursor == 1
    window.replay_last()
    assert window.controller.get_state().replay_cursor == 2
    window.replay_previous()
    assert window.controller.get_state().replay_cursor == 1

    assert window.controller.record == original


def test_replay_timer_speed_and_branching(
    qtbot,
    window: MainWindow,
) -> None:
    window.enter_replay()
    window.replay_first()
    window.replay_speed.setValue(20)
    window.start_replay()
    assert window.replay_timer.isActive()
    qtbot.waitUntil(
        lambda: window.controller.get_state().replay_cursor == 2,
        timeout=500,
    )
    assert not window.replay_timer.isActive()

    window.replay_previous()
    window.continue_from_replay()
    state = window.controller.get_state()
    assert state.replay_cursor is None
    assert state.ply == 1
    assert len(window.controller.record.moves) == 1


def test_board_moves_are_disabled_while_replaying(
    qtbot,
    window: MainWindow,
) -> None:
    window.enter_replay()
    window.replay_first()
    qtbot.mouseClick(
        window.board_widget,
        Qt.MouseButton.LeftButton,
        pos=window.board_widget.point_for(Coord(0, 6)),
    )
    assert window.board_widget.selected is None
    assert window.controller.get_state().ply == 0


def test_configured_program_side_is_not_mouse_controllable(
    qtbot,
    window: MainWindow,
) -> None:
    window.new_game(
        players=(
            PlayerRecord(
                name="程序红",
                color=Color.RED,
                controller="python",
            ),
            PlayerRecord(name="人工黑", color=Color.BLACK),
        )
    )
    qtbot.mouseClick(
        window.board_widget,
        Qt.MouseButton.LeftButton,
        pos=window.board_widget.point_for(Coord(0, 6)),
    )
    assert window.board_widget.selected is None
    assert window.controller.get_state().ply == 0


def test_draw_response_buttons_only_enable_for_pending_offer(
    window: MainWindow,
) -> None:
    assert not window.accept_draw_button.isEnabled()
    assert not window.reject_draw_button.isEnabled()
    window.offer_draw(Color.RED)
    assert window.accept_draw_button.isEnabled()
    assert window.reject_draw_button.isEnabled()
    window.reject_pending_draw()
    assert window.controller.get_state().pending_draw is None
    assert not window.accept_draw_button.isEnabled()


def test_json_and_text_export_then_import(
    tmp_path: Path,
    window: MainWindow,
) -> None:
    json_path = tmp_path / "game.json"
    text_path = tmp_path / "game.txt"
    window.export_path(json_path)
    window.export_path(text_path)
    assert json_path.read_text(encoding="utf-8").startswith("{")
    assert text_path.read_text(encoding="utf-8").splitlines() == [
        "兵一进一",
        "卒９进１",
    ]

    window.new_game(
        ruleset=Ruleset.ASIAN_2003,
        players=(
            PlayerRecord(name="新红", color=Color.RED),
            PlayerRecord(name="新黑", color=Color.BLACK),
        ),
    )
    window.import_path(json_path)
    assert window.controller.record.ruleset is Ruleset.CHINESE_2020
    assert len(window.controller.record.moves) == 2

    window.new_game()
    window.import_path(text_path)
    assert len(window.controller.record.moves) == 2


def test_invalid_import_does_not_replace_current_game(
    tmp_path: Path,
    window: MainWindow,
) -> None:
    bad_path = tmp_path / "bad.txt"
    bad_path.write_text("这不是棋谱\n", encoding="utf-8")
    original = window.controller.record

    with pytest.raises(ValueError, match="第 1 行"):
        window.import_path(bad_path)

    assert window.controller.record == original


def test_close_stops_replay_timer(window: MainWindow) -> None:
    window.enter_replay()
    window.start_replay()
    assert window.replay_timer.isActive()
    window.close()
    assert not window.replay_timer.isActive()


def test_close_stops_replay_and_local_api_thread(qtbot) -> None:
    created: dict[str, object] = {}

    class FakeServer:
        should_exit = False

        def __init__(self) -> None:
            self.started = Event()

        def run(self) -> None:
            self.started.set()
            while not self.should_exit:
                self.started.wait(0.01)

    def server_factory(app, *, host: str, port: int):
        created.update(app=app, host=host, port=port)
        server = FakeServer()
        created["server"] = server
        return server

    runtime = DesktopRuntime(
        api_enabled=True,
        port=9876,
        server_factory=server_factory,
    )
    qtbot.addWidget(runtime.window)
    runtime.start()
    server = created["server"]
    assert isinstance(server, FakeServer)
    assert server.started.wait(1)
    runtime.window.enter_replay()
    runtime.window.start_replay()

    runtime.window.close()

    assert created["host"] == "127.0.0.1"
    assert created["port"] == 9876
    assert not runtime.window.replay_timer.isActive()
    assert server.should_exit
    assert runtime.api_thread is not None
    assert runtime.api_thread.wait(3000)


def test_runtime_api_tracks_new_game_and_import_controllers(
    qtbot, tmp_path: Path
) -> None:
    created: dict[str, object] = {}

    class FakeServer:
        should_exit = False

        def run(self) -> None:
            return None

    def server_factory(app, *, host: str, port: int):
        created["app"] = app
        return FakeServer()

    runtime = DesktopRuntime(api_enabled=True, server_factory=server_factory)
    qtbot.addWidget(runtime.window)
    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(
        created["app"]
    )

    runtime.window.new_game(ruleset=Ruleset.ASIAN_2003)

    assert runtime.controller is runtime.window.controller
    assert client.get("/state").json()["ruleset"] == "asian_2003"

    imported = GameController.new()
    imported.make_move(Coord(0, 6), Coord(0, 5))
    record_path = tmp_path / "runtime-import.json"
    imported.export_record(record_path)
    runtime.window.import_path(record_path)

    assert runtime.controller is runtime.window.controller
    assert client.get("/state").json()["ply"] == 1
