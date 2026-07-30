"""PySide6 main window for local Chinese chess play and replay."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from xiangqi.adjudication import Ruleset
from xiangqi.controller import ControlError, GameController
from xiangqi.domain import Color
from xiangqi.notation import replay_text
from xiangqi.record import MoveRecord, PlayerRecord
from xiangqi.rules import evaluate_position
from xiangqi.ui.board_widget import BoardWidget
from xiangqi.ui.dialogs import NewGameDialog


class MainWindow(QMainWindow):
    """Game controls, player summary, moves and non-destructive replay."""

    controller_changed = Signal()
    controller_replaced = Signal(object)
    external_controller_replacement = Signal(object)
    closing = Signal()

    def __init__(
        self,
        controller: GameController | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("中国象棋")
        self.resize(1100, 760)
        self.controller = controller or GameController.new()
        self.replay_timer = QTimer(self)
        self.replay_timer.timeout.connect(self.replay_next)
        self.external_controller_replacement.connect(self._replace_controller)

        self.red_player_label = QLabel()
        self.black_player_label = QLabel()
        player_panel = QWidget()
        player_layout = QVBoxLayout(player_panel)
        player_layout.addWidget(QLabel("玩家"))
        player_layout.addWidget(self.red_player_label)
        player_layout.addWidget(self.black_player_label)
        player_layout.addStretch()

        self.board_widget = BoardWidget(self.controller)
        self.move_list = QListWidget()
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.rule_display = QComboBox()
        self.rule_display.addItem("中国棋规（2020）", Ruleset.CHINESE_2020)
        self.rule_display.addItem("亚洲棋规（2003）", Ruleset.ASIAN_2003)
        self.rule_display.setEnabled(False)
        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)
        info_layout.addWidget(QLabel("着法记录"))
        info_layout.addWidget(self.move_list, 1)
        info_layout.addWidget(QLabel("规则"))
        info_layout.addWidget(self.rule_display)
        info_layout.addWidget(self.status_label)

        self.splitter = QSplitter()
        self.splitter.addWidget(player_panel)
        self.splitter.addWidget(self.board_widget)
        self.splitter.addWidget(info_panel)
        self.splitter.setStretchFactor(1, 1)

        actions = QHBoxLayout()
        for label, handler in (
            ("新局", self.show_new_game_dialog),
            ("悔棋", self.undo),
            ("提和", self.offer_draw_for_current_side),
            ("导入", self.choose_import),
            ("导出", self.choose_export),
            ("回放", self.enter_replay),
        ):
            button = QPushButton(label)
            button.clicked.connect(handler)
            actions.addWidget(button)
        self.accept_draw_button = QPushButton("同意和棋")
        self.accept_draw_button.clicked.connect(self.accept_pending_draw)
        actions.addWidget(self.accept_draw_button)
        self.reject_draw_button = QPushButton("拒绝和棋")
        self.reject_draw_button.clicked.connect(self.reject_pending_draw)
        actions.addWidget(self.reject_draw_button)

        replay = QHBoxLayout()
        for label, handler in (
            ("首步", self.replay_first),
            ("前一步", self.replay_previous),
            ("播放/暂停", self.toggle_replay),
            ("后一步", self.replay_next),
            ("末步", self.replay_last),
            ("从此继续", self.continue_from_replay),
        ):
            button = QPushButton(label)
            button.clicked.connect(handler)
            replay.addWidget(button)
        self.replay_speed = QSpinBox()
        self.replay_speed.setRange(20, 5000)
        self.replay_speed.setValue(800)
        self.replay_speed.setSuffix(" ms")
        replay.addWidget(QLabel("速度"))
        replay.addWidget(self.replay_speed)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addWidget(self.splitter, 1)
        layout.addLayout(actions)
        layout.addLayout(replay)
        self.setCentralWidget(root)

        self.controller_changed.connect(self.refresh)
        self._bind_controller()
        self.refresh()

    def _bind_controller(self) -> None:
        self.controller.register_callback(lambda _event: self.controller_changed.emit())

    def _replace_controller(self, controller: GameController) -> None:
        if controller is self.controller:
            return
        self.stop_replay()
        old_board = self.board_widget
        self.controller = controller
        self.board_widget = BoardWidget(controller)
        self.splitter.replaceWidget(self.splitter.indexOf(old_board), self.board_widget)
        old_board.deleteLater()
        self._bind_controller()
        self.refresh()
        self.controller_replaced.emit(controller)

    def new_game(
        self,
        *,
        ruleset: Ruleset = Ruleset.CHINESE_2020,
        players: tuple[PlayerRecord, PlayerRecord] | None = None,
    ) -> None:
        self._replace_controller(GameController.new(ruleset=ruleset, players=players))

    def show_new_game_dialog(self) -> None:
        dialog = NewGameDialog(self)
        if dialog.exec():
            settings = dialog.settings()
            self.new_game(ruleset=settings.ruleset, players=settings.players)

    def undo(self) -> None:
        self.stop_replay()
        self.controller.undo()

    def offer_draw_for_current_side(self) -> None:
        self.offer_draw(self.controller.get_state().side_to_move)

    def offer_draw(self, actor: Color) -> None:
        self.controller.offer_draw(actor)

    def respond_draw(self, actor: Color, accept: bool) -> None:
        pending = self.controller.get_state().pending_draw
        if pending is None or actor is not pending.opponent:
            raise ValueError("只能由提和方的对方回应")
        self.controller.respond_draw(actor, accept)

    def accept_pending_draw(self) -> None:
        pending = self.controller.get_state().pending_draw
        if pending is not None:
            self.respond_draw(pending.opponent, True)

    def reject_pending_draw(self) -> None:
        pending = self.controller.get_state().pending_draw
        if pending is not None:
            self.respond_draw(pending.opponent, False)

    def enter_replay(self) -> None:
        self.stop_replay()
        self.controller.set_replay_cursor(0)

    def replay_first(self) -> None:
        self._ensure_replay()
        self.controller.set_replay_cursor(0)

    def replay_previous(self) -> None:
        self._ensure_replay()
        cursor = self.controller.get_state().replay_cursor
        assert cursor is not None
        self.controller.set_replay_cursor(max(0, cursor - 1))

    def replay_next(self) -> None:
        self._ensure_replay()
        cursor = self.controller.get_state().replay_cursor
        assert cursor is not None
        last = len(self.controller.record.moves)
        self.controller.set_replay_cursor(min(last, cursor + 1))
        if cursor + 1 >= last:
            self.stop_replay()

    def replay_last(self) -> None:
        self._ensure_replay()
        self.controller.set_replay_cursor(len(self.controller.record.moves))
        self.stop_replay()

    def _ensure_replay(self) -> None:
        if self.controller.get_state().replay_cursor is None:
            self.enter_replay()

    def start_replay(self) -> None:
        self._ensure_replay()
        if self.controller.get_state().replay_cursor == len(
            self.controller.record.moves
        ):
            self.controller.set_replay_cursor(0)
        self.replay_timer.start(self.replay_speed.value())

    def stop_replay(self) -> None:
        self.replay_timer.stop()

    def toggle_replay(self) -> None:
        if self.replay_timer.isActive():
            self.stop_replay()
        else:
            self.start_replay()

    def continue_from_replay(self) -> None:
        self.stop_replay()
        self.controller.branch_from_replay()

    def export_path(self, path: str | Path) -> None:
        target = Path(path)
        self.controller.export_record(
            target, "json" if target.suffix.lower() == ".json" else "text"
        )

    def import_path(self, path: str | Path) -> None:
        source = Path(path)
        if source.suffix.lower() == ".json":
            replacement = GameController.from_record(GameController.new().record)
            replacement.load_record(source)
        else:
            text = source.read_text(encoding="utf-8")
            replayed = replay_text(text)
            board = GameController.new().record
            moves: list[MoveRecord] = []
            before = GameController.new().get_state().board
            side = Color.RED
            for item in replayed.moves:
                after = before.move_unchecked(item.move.start, item.move.end)
                position = evaluate_position(after, side.opponent)
                moves.append(
                    MoveRecord.from_move(
                        item.move,
                        notation=item.notation,
                        position_after=item.position_after,
                        in_check=position.in_check,
                    )
                )
                before = after
                side = side.opponent
            record = board.model_copy(update={"moves": tuple(moves)})
            replacement = GameController.from_record(record)
        self._replace_controller(replacement)

    def choose_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入棋谱",
            "",
            "棋谱 (*.json *.txt);;JSON (*.json);;中文棋谱 (*.txt)",
        )
        if path:
            self._run_file_action(lambda: self.import_path(path))

    def choose_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出棋谱",
            "xiangqi.json",
            "JSON (*.json);;中文棋谱 (*.txt)",
        )
        if path:
            self._run_file_action(lambda: self.export_path(path))

    def _run_file_action(self, action) -> None:
        try:
            action()
        except (OSError, ValueError, ControlError) as error:
            QMessageBox.critical(self, "棋谱操作失败", str(error))

    def refresh(self) -> None:
        record = self.controller.record
        players = {player.color: player for player in record.players}
        self.red_player_label.setText(self._player_text(players[Color.RED]))
        self.black_player_label.setText(self._player_text(players[Color.BLACK]))
        self.move_list.clear()
        self.move_list.addItems([move.notation for move in record.moves])
        self.rule_display.setCurrentIndex(self.rule_display.findData(record.ruleset))
        state = self.controller.get_state()
        if state.result is not None:
            status = f"对局结束：{state.result.reason or state.result.status}"
        elif state.position.in_check:
            status = f"{self._side_name(state.side_to_move)}被将军"
        else:
            status = f"轮到{self._side_name(state.side_to_move)}"
        if state.replay_cursor is not None:
            status += f"；回放 {state.replay_cursor}/{len(record.moves)}"
        if state.pending_draw is not None:
            status += f"；{self._side_name(state.pending_draw)}提和待回应"
        pending_response = state.pending_draw is not None
        self.accept_draw_button.setEnabled(pending_response)
        self.reject_draw_button.setEnabled(pending_response)
        self.status_label.setText(status)

    @staticmethod
    def _player_text(player: PlayerRecord) -> str:
        controls = {"human": "人工", "python": "Python", "network": "网络"}
        return (
            f"{MainWindow._side_name(player.color)}：{player.name}"
            f"（{controls[player.controller]}）"
        )

    @staticmethod
    def _side_name(color: Color) -> str:
        return "红方" if color is Color.RED else "黑方"

    def closeEvent(self, event) -> None:
        self.stop_replay()
        self.closing.emit()
        super().closeEvent(event)
