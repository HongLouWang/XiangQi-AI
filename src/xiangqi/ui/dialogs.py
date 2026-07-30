"""Dialogs used by the desktop application."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from xiangqi.adjudication import Ruleset
from xiangqi.controller import ControllerKind
from xiangqi.domain import Color
from xiangqi.record import PlayerRecord


@dataclass(frozen=True, slots=True)
class NewGameSettings:
    players: tuple[PlayerRecord, PlayerRecord]
    ruleset: Ruleset


class NewGameDialog(QDialog):
    """Collect player identities, colors, control modes and rules."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("新建对局")
        self.player1_name = QLineEdit("玩家 1")
        self.player2_name = QLineEdit("玩家 2")
        self.player1_color = self._color_combo(Color.RED)
        self.player2_color = self._color_combo(Color.BLACK)
        self.player1_control = self._control_combo()
        self.player2_control = self._control_combo()
        self.ruleset = QComboBox()
        self.ruleset.addItem("中国棋规（2020）", Ruleset.CHINESE_2020)
        self.ruleset.addItem("亚洲棋规（2003）", Ruleset.ASIAN_2003)

        form = QFormLayout()
        form.addRow("玩家 1 姓名", self.player1_name)
        form.addRow("玩家 1 执棋", self.player1_color)
        form.addRow("玩家 1 控制", self.player1_control)
        form.addRow("玩家 2 姓名", self.player2_name)
        form.addRow("玩家 2 执棋", self.player2_color)
        form.addRow("玩家 2 控制", self.player2_control)
        form.addRow("棋规", self.ruleset)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

        self.player1_color.currentIndexChanged.connect(
            lambda: self._sync_color(self.player1_color, self.player2_color)
        )
        self.player2_color.currentIndexChanged.connect(
            lambda: self._sync_color(self.player2_color, self.player1_color)
        )

    @staticmethod
    def _color_combo(selected: Color) -> QComboBox:
        combo = QComboBox()
        combo.addItem("红方", Color.RED)
        combo.addItem("黑方", Color.BLACK)
        combo.setCurrentIndex(combo.findData(selected))
        return combo

    @staticmethod
    def _control_combo() -> QComboBox:
        combo = QComboBox()
        combo.addItem("人工", ControllerKind.HUMAN)
        combo.addItem("Python 程序", ControllerKind.PYTHON)
        combo.addItem("本机网络", ControllerKind.NETWORK)
        return combo

    @staticmethod
    def _sync_color(source: QComboBox, target: QComboBox) -> None:
        opposite = (
            Color.BLACK if Color(source.currentData()) is Color.RED else Color.RED
        )
        if Color(target.currentData()) is not opposite:
            target.setCurrentIndex(target.findData(opposite))

    def can_accept(self) -> bool:
        return bool(
            self.player1_name.text().strip()
            and self.player2_name.text().strip()
            and Color(self.player1_color.currentData())
            is not Color(self.player2_color.currentData())
        )

    def settings(self) -> NewGameSettings:
        if not self.can_accept():
            raise ValueError("玩家姓名不能为空，且两名玩家不能选择相同颜色")
        players = (
            PlayerRecord(
                name=self.player1_name.text().strip(),
                color=Color(self.player1_color.currentData()),
                controller=ControllerKind(self.player1_control.currentData()).value,
            ),
            PlayerRecord(
                name=self.player2_name.text().strip(),
                color=Color(self.player2_color.currentData()),
                controller=ControllerKind(self.player2_control.currentData()).value,
            ),
        )
        return NewGameSettings(players, Ruleset(self.ruleset.currentData()))

    def _accept_if_valid(self) -> None:
        if self.can_accept():
            self.accept()
        else:
            QMessageBox.warning(self, "无法开始", "请填写姓名并为双方选择不同颜色。")
