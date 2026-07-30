"""Scalable, mouse-driven Chinese chess board widget."""

from __future__ import annotations

from typing import ClassVar

from PySide6.QtCore import QObject, QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from xiangqi.controller import ControllerKind, GameController
from xiangqi.domain import Color, Coord, Piece, PieceType


class _RefreshBridge(QObject):
    requested = Signal()


class BoardWidget(QWidget):
    """Draw and operate a fixed-orientation Chinese chess board."""

    board_color = QColor("#E7BE78")
    line_color = QColor("#4A2D16")
    red_piece_color = QColor("#B42318")
    black_piece_color = QColor("#202020")
    selection_color = QColor("#2F80ED")
    last_move_color = QColor("#F2994A")
    legal_target_color = QColor("#27AE60")

    _piece_names: ClassVar[dict[tuple[Color, PieceType], str]] = {
        (Color.RED, PieceType.GENERAL): "帅",
        (Color.RED, PieceType.ADVISOR): "仕",
        (Color.RED, PieceType.ELEPHANT): "相",
        (Color.RED, PieceType.HORSE): "马",
        (Color.RED, PieceType.ROOK): "车",
        (Color.RED, PieceType.CANNON): "炮",
        (Color.RED, PieceType.PAWN): "兵",
        (Color.BLACK, PieceType.GENERAL): "将",
        (Color.BLACK, PieceType.ADVISOR): "士",
        (Color.BLACK, PieceType.ELEPHANT): "象",
        (Color.BLACK, PieceType.HORSE): "马",
        (Color.BLACK, PieceType.ROOK): "车",
        (Color.BLACK, PieceType.CANNON): "砲",
        (Color.BLACK, PieceType.PAWN): "卒",
    }

    def __init__(
        self,
        controller: GameController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.selected: Coord | None = None
        self.legal_targets: set[Coord] = set()
        self.setMinimumSize(300, 360)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self._refresh_bridge = _RefreshBridge(self)
        self._refresh_bridge.requested.connect(self._controller_changed)
        self.controller.register_callback(
            lambda _event: self._refresh_bridge.requested.emit()
        )

    @property
    def grid_spacing(self) -> float:
        available_width = max(1.0, self.width() - 72.0)
        available_height = max(1.0, self.height() - 72.0)
        return max(1.0, min(available_width / 8.0, available_height / 9.0))

    @property
    def board_origin(self) -> QPointF:
        spacing = self.grid_spacing
        return QPointF(
            (self.width() - 8.0 * spacing) / 2.0,
            (self.height() - 9.0 * spacing) / 2.0,
        )

    @property
    def last_move(self) -> tuple[Coord, Coord] | None:
        move = self.controller.get_state().last_move
        return None if move is None else (move.start, move.end)

    @property
    def highlighted_piece(self) -> Coord | None:
        move = self.controller.get_state().last_move
        return None if move is None else move.end

    def point_for(self, coord: Coord) -> QPoint:
        origin = self.board_origin
        spacing = self.grid_spacing
        return QPoint(
            round(origin.x() + coord.file * spacing),
            round(origin.y() + coord.rank * spacing),
        )

    def coord_at(self, point: QPoint | QPointF) -> Coord | None:
        origin = self.board_origin
        spacing = self.grid_spacing
        file = round((point.x() - origin.x()) / spacing)
        rank = round((point.y() - origin.y()) / spacing)
        if not (0 <= file < 9 and 0 <= rank < 10):
            return None
        intersection = self.point_for(Coord(file, rank))
        hit_radius = spacing * 0.42
        if (
            abs(point.x() - intersection.x()) > hit_radius
            or abs(point.y() - intersection.y()) > hit_radius
        ):
            return None
        return Coord(file, rank)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() is not Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        coord = self.coord_at(event.position())
        state = self.controller.get_state()
        if state.replay_cursor is not None:
            self._clear_selection()
            return
        player = next(
            player
            for player in self.controller.record.players
            if player.color is state.side_to_move
        )
        if player.controller != ControllerKind.HUMAN.value:
            self._clear_selection()
            return
        if state.controllers[state.side_to_move].kind is not ControllerKind.HUMAN:
            self._clear_selection()
            return

        if self.selected is not None and coord in self.legal_targets:
            self.controller.make_move(self.selected, coord)
            self._clear_selection()
            return

        if coord == self.selected:
            self._clear_selection()
            return

        piece = None if coord is None else state.board.at(coord)
        if (
            coord is not None
            and piece is not None
            and piece.color is state.side_to_move
        ):
            legal = self.controller.get_legal_moves().get(coord, ())
            self.selected = coord
            self.legal_targets = set(legal)
            self.update()
            return

        self._clear_selection()

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.board_color)
        self._draw_board(painter)
        self._draw_highlights(painter)
        self._draw_pieces(painter)

    def _controller_changed(self) -> None:
        self.selected = None
        self.legal_targets.clear()
        self.update()

    def _clear_selection(self) -> None:
        self.selected = None
        self.legal_targets.clear()
        self.update()

    def _draw_board(self, painter: QPainter) -> None:
        painter.setPen(QPen(self.line_color, max(1.0, self.grid_spacing * 0.018)))
        for rank in range(10):
            painter.drawLine(
                self.point_for(Coord(0, rank)),
                self.point_for(Coord(8, rank)),
            )
        for file in range(9):
            if file in (0, 8):
                painter.drawLine(
                    self.point_for(Coord(file, 0)),
                    self.point_for(Coord(file, 9)),
                )
            else:
                painter.drawLine(
                    self.point_for(Coord(file, 0)),
                    self.point_for(Coord(file, 4)),
                )
                painter.drawLine(
                    self.point_for(Coord(file, 5)),
                    self.point_for(Coord(file, 9)),
                )

        for top in (0, 7):
            painter.drawLine(
                self.point_for(Coord(3, top)),
                self.point_for(Coord(5, top + 2)),
            )
            painter.drawLine(
                self.point_for(Coord(5, top)),
                self.point_for(Coord(3, top + 2)),
            )

        origin = self.board_origin
        spacing = self.grid_spacing
        river = QRectF(origin.x(), origin.y() + 4 * spacing, 8 * spacing, spacing)
        painter.setFont(QFont("Songti SC", max(10, round(spacing * 0.28))))
        painter.drawText(
            river.adjusted(spacing * 0.8, 0, -spacing * 0.8, 0),
            Qt.AlignmentFlag.AlignCenter,
            "楚 河        汉 界",
        )

    def _draw_highlights(self, painter: QPainter) -> None:
        radius = self.grid_spacing * 0.39
        last = self.last_move
        if last is not None:
            for coord in last:
                painter.setPen(
                    QPen(self.last_move_color, max(3.0, self.grid_spacing * 0.07))
                )
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QPointF(self.point_for(coord)), radius, radius)

        if self.selected is not None:
            painter.setPen(
                QPen(self.selection_color, max(3.0, self.grid_spacing * 0.075))
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                QPointF(self.point_for(self.selected)),
                radius * 1.08,
                radius * 1.08,
            )

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.legal_target_color)
        for coord in self.legal_targets:
            painter.drawEllipse(
                QPointF(self.point_for(coord)),
                self.grid_spacing * 0.105,
                self.grid_spacing * 0.105,
            )

    def _draw_pieces(self, painter: QPainter) -> None:
        state = self.controller.get_state()
        radius = self.grid_spacing * 0.34
        font = QFont("Songti SC", max(11, round(self.grid_spacing * 0.35)))
        font.setBold(True)
        painter.setFont(font)
        for coord, piece in state.board.pieces.items():
            self._draw_piece(painter, coord, piece, radius)

    def _draw_piece(
        self,
        painter: QPainter,
        coord: Coord,
        piece: Piece,
        radius: float,
    ) -> None:
        center = QPointF(self.point_for(coord))
        piece_color = (
            self.red_piece_color if piece.color is Color.RED else self.black_piece_color
        )
        painter.setBrush(QColor("#F8E6B5"))
        painter.setPen(QPen(piece_color, max(2.0, self.grid_spacing * 0.035)))
        painter.drawEllipse(center, radius, radius)
        painter.setPen(piece_color)
        painter.drawText(
            QRectF(
                center.x() - radius,
                center.y() - radius,
                radius * 2,
                radius * 2,
            ),
            Qt.AlignmentFlag.AlignCenter,
            self._piece_names[(piece.color, piece.kind)],
        )
