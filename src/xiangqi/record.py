"""Versioned JSON game records with transactional validation and atomic export."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from xiangqi.adjudication import AdjudicationKind, Ruleset
from xiangqi.board import Board
from xiangqi.domain import Color, Coord, Move, Piece, PieceType
from xiangqi.notation import NotationError, format_move
from xiangqi.rules import legal_destinations


class RecordError(ValueError):
    """A record is malformed, unsupported, or fails replay validation."""


class _RecordModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PlayerRecord(_RecordModel):
    name: str = Field(min_length=1)
    color: Color
    controller: Literal["human", "python", "network"] = "human"


class AdjudicationRecord(_RecordModel):
    kind: AdjudicationKind
    reason: str
    responsible: Color | None = None
    rule_reference: str | None = None


class MoveRecord(_RecordModel):
    start: tuple[int, int]
    end: tuple[int, int]
    piece_color: Color
    piece_kind: PieceType
    captured_color: Color | None = None
    captured_kind: PieceType | None = None
    notation: str
    position_after: str = Field(pattern=r"^[0-9a-f]{64}$")
    in_check: bool = False
    adjudication: AdjudicationRecord | None = None

    @model_validator(mode="after")
    def capture_fields_are_paired(self) -> MoveRecord:
        if (self.captured_color is None) != (self.captured_kind is None):
            raise ValueError("被吃棋子的颜色和类型必须同时提供")
        Coord(*self.start)
        Coord(*self.end)
        return self

    @classmethod
    def from_move(
        cls,
        move: Move,
        *,
        notation: str,
        position_after: str,
        in_check: bool = False,
        adjudication: AdjudicationRecord | dict[str, Any] | None = None,
    ) -> MoveRecord:
        return cls(
            start=(move.start.file, move.start.rank),
            end=(move.end.file, move.end.rank),
            piece_color=move.piece.color,
            piece_kind=move.piece.kind,
            captured_color=None if move.captured is None else move.captured.color,
            captured_kind=None if move.captured is None else move.captured.kind,
            notation=notation,
            position_after=position_after,
            in_check=in_check,
            adjudication=adjudication,
        )

    def to_move(self) -> Move:
        captured = (
            None
            if self.captured_color is None
            else Piece(self.captured_color, self.captured_kind)  # type: ignore[arg-type]
        )
        return Move(
            Coord(*self.start),
            Coord(*self.end),
            Piece(self.piece_color, self.piece_kind),
            captured,
        )


class ResultRecord(_RecordModel):
    status: Literal["ongoing", "red_win", "black_win", "draw"]
    reason: str | None = None
    winner: Color | None = None


class GameRecord(_RecordModel):
    format_version: Literal[1] = 1
    created_at: datetime | None = None
    ruleset: Ruleset = Ruleset.CHINESE_2020
    players: tuple[PlayerRecord, PlayerRecord] = (
        PlayerRecord(name="红方", color=Color.RED),
        PlayerRecord(name="黑方", color=Color.BLACK),
    )
    initial_fen: str
    initial_side: Color = Color.RED
    moves: tuple[MoveRecord, ...] = ()
    result: ResultRecord = ResultRecord(status="ongoing")

    @model_validator(mode="after")
    def players_cover_both_colors(self) -> GameRecord:
        if {player.color for player in self.players} != {Color.RED, Color.BLACK}:
            raise ValueError("两名玩家必须分别控制红方和黑方")
        return self


def load_json(path: str | os.PathLike[str]) -> GameRecord:
    try:
        raw = Path(path).read_text(encoding="utf-8")
        return GameRecord.model_validate_json(raw)
    except (OSError, UnicodeError, ValidationError, json.JSONDecodeError) as error:
        raise RecordError(f"无法读取棋谱: {error}") from error


def validate_record(record: GameRecord) -> GameRecord:
    """Replay into local temporary state and return only after full validation."""
    try:
        board = Board.from_fen(record.initial_fen)
    except ValueError as error:
        raise RecordError(f"初始局面无效: {error}") from error
    side = record.initial_side
    for index, item in enumerate(record.moves, 1):
        try:
            move = item.to_move()
            if move.piece.color is not side:
                raise ValueError("棋子颜色与当前行棋方不符")
            if board.at(move.start) != move.piece:
                raise ValueError("起点棋子与记录不符")
            if board.at(move.end) != move.captured:
                raise ValueError("吃子信息与局面不符")
            if move.end not in legal_destinations(board, move.start, side):
                raise ValueError("非法着法")
            if format_move(board, move) != item.notation:
                raise ValueError("中文记谱与着法不符")
            board = board.move_unchecked(move.start, move.end)
            side = side.opponent
            if board.position_key(side) != item.position_after:
                raise ValueError("走后局面摘要不符")
        except (ValueError, NotationError) as error:
            raise RecordError(f"第 {index} 步校验失败: {error}") from error
    return record


def load_and_validate(path: str | os.PathLike[str]) -> GameRecord:
    return validate_record(load_json(path))


def export_json(record: GameRecord, path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(record.model_dump_json(indent=2))
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    except OSError as error:
        raise RecordError(f"无法导出棋谱: {error}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
