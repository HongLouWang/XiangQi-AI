"""Thread-safe game state controller shared by UI, Python and network clients."""

from __future__ import annotations

import os
import secrets
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import RLock
from types import MappingProxyType

from xiangqi.adjudication import (
    Adjudication,
    AdjudicationKind,
    Asian2003Adjudicator,
    Chinese2020Adjudicator,
    PositionFrame,
    Ruleset,
)
from xiangqi.board import Board
from xiangqi.domain import Color, Coord, Move, PositionKind, PositionResult
from xiangqi.notation import format_move, replay_text
from xiangqi.record import (
    AdjudicationRecord,
    DrawEventRecord,
    GameRecord,
    MoveRecord,
    PlayerRecord,
    ResultRecord,
    export_json,
    load_and_validate,
    validate_record,
)
from xiangqi.rules import evaluate_position, legal_destinations


class ControlError(ValueError):
    """A command is invalid for the current controller state."""


class StaleVersionError(ControlError):
    """The caller based a mutation on an old position version."""


class ControllerKind(StrEnum):
    HUMAN = "human"
    PYTHON = "python"
    NETWORK = "network"


class GameEventKind(StrEnum):
    MOVE = "move"
    UNDO = "undo"
    DRAW_OFFERED = "draw_offered"
    DRAW_RESPONDED = "draw_responded"
    RECORD_LOADED = "record_loaded"
    REPLAY_CHANGED = "replay_changed"
    BRANCHED = "branched"
    CONTROL_CHANGED = "control_changed"


@dataclass(frozen=True, slots=True)
class SideControl:
    kind: ControllerKind
    controller_id: str | None = None
    token: str | None = None


@dataclass(frozen=True, slots=True)
class CallbackFailure:
    callback: Callable[[GameEvent], object]
    event: GameEvent
    error: Exception


@dataclass(frozen=True, slots=True)
class ControllerState:
    board: Board
    side_to_move: Color
    ruleset: Ruleset
    ply: int
    version: int
    position: PositionResult
    result: ResultRecord | None
    pending_draw: Color | None
    replay_cursor: int | None
    last_move: Move | None
    adjudication: Adjudication | None
    controllers: Mapping[Color, SideControl]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "controllers", MappingProxyType(dict(self.controllers))
        )


@dataclass(frozen=True, slots=True)
class GameEvent:
    kind: GameEventKind
    version: int
    before_board: Board
    after_board: Board
    move: Move | None
    next_side: Color
    in_check: bool
    checkmate: bool
    stalemate: bool
    position: PositionResult
    adjudication: Adjudication | None
    result: ResultRecord | None
    state: ControllerState


@dataclass(frozen=True, slots=True)
class _ReplayState:
    board: Board
    side: Color
    frames: tuple[PositionFrame, ...]
    records: tuple[MoveRecord, ...]
    position: PositionResult
    adjudication: Adjudication | None
    result: ResultRecord | None


Callback = Callable[[GameEvent], object]


class GameController:
    """Serialize all game mutations and publish immutable state transitions."""

    def __init__(self, record: GameRecord) -> None:
        validated = validate_record(record)
        self._lock = RLock()
        self._callbacks: list[Callback] = []
        self._callback_errors: list[CallbackFailure] = []
        self._controls = {
            Color.RED: SideControl(ControllerKind.HUMAN),
            Color.BLACK: SideControl(ControllerKind.HUMAN),
        }
        self._version = 0
        self._pending_draw: Color | None = (
            validated.draw_events[-1].actor
            if validated.draw_events and validated.draw_events[-1].action == "offer"
            else None
        )
        self._replay_cursor: int | None = None
        self._base_record = validated.model_copy(
            update={"result": ResultRecord(status="ongoing")}
        )
        self._current = self._replay(self._base_record, len(self._base_record.moves))
        imported_result = validated.result
        if imported_result.status == "draw" and self._current.result is None:
            self._current = _ReplayState(
                self._current.board,
                self._current.side,
                self._current.frames,
                self._current.records,
                self._current.position,
                self._current.adjudication,
                imported_result,
            )
            self._base_record = self._record_from_current()
        elif imported_result.status != "ongoing":
            if (
                self._current.result is None
                or imported_result.status != self._current.result.status
                or imported_result.winner is not self._current.result.winner
            ):
                raise ControlError("棋谱结果与逐步重放得到的局面不一致")

    @classmethod
    def new(
        cls,
        *,
        ruleset: Ruleset = Ruleset.CHINESE_2020,
        players: tuple[PlayerRecord, PlayerRecord] | None = None,
        initial_board: Board | None = None,
        initial_side: Color = Color.RED,
    ) -> GameController:
        board = initial_board or Board.standard()
        record = GameRecord(
            created_at=datetime.now(UTC),
            ruleset=ruleset,
            players=players or GameRecord.model_fields["players"].default,
            initial_fen=board.to_fen(),
            initial_side=initial_side,
        )
        return cls(record)

    @classmethod
    def from_record(cls, record: GameRecord) -> GameController:
        return cls(record)

    @property
    def record(self) -> GameRecord:
        with self._lock:
            return self._record_from_current()

    @property
    def state(self) -> ControllerState:
        return self.get_state()

    @property
    def callback_errors(self) -> tuple[CallbackFailure, ...]:
        with self._lock:
            return tuple(self._callback_errors)

    def get_state(self) -> ControllerState:
        with self._lock:
            displayed = (
                self._replay(self._base_record, self._replay_cursor)
                if self._replay_cursor is not None
                else self._current
            )
            return ControllerState(
                board=displayed.board,
                side_to_move=displayed.side,
                ruleset=self._base_record.ruleset,
                ply=(
                    self._replay_cursor
                    if self._replay_cursor is not None
                    else len(self._current.records)
                ),
                version=self._version,
                position=displayed.position,
                result=displayed.result,
                pending_draw=self._pending_draw,
                replay_cursor=self._replay_cursor,
                last_move=(None if not displayed.frames else displayed.frames[-1].move),
                adjudication=displayed.adjudication,
                controllers=self._controls,
            )

    def get_legal_moves(
        self, side: Color | None = None
    ) -> Mapping[Coord, tuple[Coord, ...]]:
        with self._lock:
            state = self.get_state()
            requested = state.side_to_move if side is None else side
            grouped: dict[Coord, tuple[Coord, ...]] = {}
            for coord, piece in state.board.pieces.items():
                if piece.color is requested:
                    destinations = legal_destinations(state.board, coord, requested)
                    if destinations:
                        grouped[coord] = destinations
            return MappingProxyType(grouped)

    def make_move(
        self,
        from_pos: Coord,
        to_pos: Coord,
        actor: str | None = None,
        *,
        expected_version: int | None = None,
    ) -> GameEvent:
        with self._lock:
            self._check_version(expected_version)
            if self._replay_cursor is not None:
                raise ControlError("回放模式中不能走棋，请先从当前局面继续")
            if self._current.result is not None:
                raise ControlError("对局已经结束，须先悔棋或从历史局面继续")
            if self._pending_draw is not None:
                raise ControlError("正在等待对方回应和棋请求")
            self._authorize(self._current.side, actor)
            if to_pos not in legal_destinations(
                self._current.board, from_pos, self._current.side
            ):
                raise ControlError("非法着法")

            before = self._current.board
            piece = before.at(from_pos)
            assert piece is not None
            move = Move(from_pos, to_pos, piece, before.at(to_pos))
            notation = format_move(before, move)
            after = before.move_unchecked(from_pos, to_pos)
            next_side = self._current.side.opponent
            frame = PositionFrame.from_transition(
                before,
                self._current.side,
                move,
                after,
                analyze_kill=False,
            )
            frames = (*self._current.frames, frame)
            position = evaluate_position(after, next_side)
            adjudicator = self._adjudicator()
            adjudication = adjudicator.evaluate(frames)
            if self._continues_prohibited_cycle(
                self._current.frames,
                self._current.adjudication,
                frame,
                adjudication,
            ):
                adjudication = adjudicator.loss_for_ignored_must_change(adjudication)
            result = self._result_for(position, adjudication)
            item = MoveRecord.from_move(
                move,
                notation=notation,
                position_after=after.position_key(next_side),
                in_check=position.in_check,
                adjudication=self._adjudication_record(adjudication, len(frames)),
            )
            self._current = _ReplayState(
                board=after,
                side=next_side,
                frames=frames,
                records=(*self._current.records, item),
                position=position,
                adjudication=adjudication,
                result=result,
            )
            self._base_record = self._record_from_current()
            self._pending_draw = None
            return self._finish_event(
                GameEventKind.MOVE, before_board=before, move=move
            )

    def undo(
        self,
        steps: int = 1,
        *,
        expected_version: int | None = None,
    ) -> GameEvent:
        if steps < 1:
            raise ControlError("悔棋步数必须大于零")
        with self._lock:
            self._check_version(expected_version)
            before = self.get_state().board
            count = max(0, len(self._current.records) - steps)
            if (
                not self._current.records
                and self._current.result is None
                and self._pending_draw is None
            ):
                raise ControlError("已经在开局，无法继续悔棋")
            self._base_record = self._base_record.model_copy(
                update={
                    "moves": self._base_record.moves[:count],
                    "result": ResultRecord(status="ongoing"),
                    "draw_events": self._draw_events_for_resume(count),
                }
            )
            self._current = self._replay(self._base_record, count)
            self._pending_draw = None
            self._replay_cursor = None
            return self._finish_event(GameEventKind.UNDO, before_board=before)

    def offer_draw(
        self,
        actor: Color,
        *,
        control_token: str | None = None,
        expected_version: int | None = None,
    ) -> GameEvent:
        with self._lock:
            self._check_version(expected_version)
            if self._current.result is not None:
                raise ControlError("对局已经结束")
            if self._pending_draw is not None:
                raise ControlError("已有待回应的和棋请求")
            self._authorize(actor, control_token)
            before = self._current.board
            self._pending_draw = actor
            self._base_record = self._base_record.model_copy(
                update={
                    "draw_events": (
                        *self._base_record.draw_events,
                        DrawEventRecord(
                            action="offer",
                            actor=actor,
                            ply=len(self._current.records),
                        ),
                    )
                }
            )
            return self._finish_event(GameEventKind.DRAW_OFFERED, before_board=before)

    def respond_draw(
        self,
        actor: Color,
        accept: bool,
        *,
        control_token: str | None = None,
        expected_version: int | None = None,
    ) -> GameEvent:
        with self._lock:
            self._check_version(expected_version)
            if self._pending_draw is None:
                raise ControlError("当前没有待回应的和棋请求")
            if actor is not self._pending_draw.opponent:
                raise ControlError("只能由提和方的对方回应")
            self._authorize(actor, control_token)
            before = self._current.board
            self._pending_draw = None
            response = DrawEventRecord(
                action="accept" if accept else "reject",
                actor=actor,
                ply=len(self._current.records),
            )
            self._base_record = self._base_record.model_copy(
                update={
                    "draw_events": (
                        *self._base_record.draw_events,
                        response,
                    )
                }
            )
            if accept:
                self._current = _ReplayState(
                    board=self._current.board,
                    side=self._current.side,
                    frames=self._current.frames,
                    records=self._current.records,
                    position=self._current.position,
                    adjudication=self._current.adjudication,
                    result=ResultRecord(status="draw", reason="双方同意和棋"),
                )
                self._base_record = self._record_from_current()
            return self._finish_event(GameEventKind.DRAW_RESPONDED, before_board=before)

    def register_callback(self, callback: Callback) -> None:
        if not callable(callback):
            raise TypeError("callback 必须可调用")
        with self._lock:
            self._callbacks.append(callback)

    def claim_side(
        self,
        side: Color,
        controller_id: str,
        kind: ControllerKind = ControllerKind.PYTHON,
        *,
        expected_version: int | None = None,
    ) -> SideControl:
        if kind is ControllerKind.HUMAN:
            raise ControlError("外部控制权类型必须是 python 或 network")
        if not controller_id:
            raise ControlError("控制器标识不能为空")
        with self._lock:
            self._check_version(expected_version)
            current = self._controls[side]
            if current.kind is not ControllerKind.HUMAN:
                raise ControlError(f"{side.value} 已被其他控制器占用")
            lease = SideControl(kind, controller_id, secrets.token_urlsafe(32))
            before = self.get_state().board
            self._controls[side] = lease
            self._finish_event(GameEventKind.CONTROL_CHANGED, before_board=before)
            return lease

    def release_side(
        self,
        side: Color,
        controller_id: str,
        token: str,
        *,
        expected_version: int | None = None,
    ) -> GameEvent:
        with self._lock:
            self._check_version(expected_version)
            current = self._controls[side]
            if (
                current.kind is ControllerKind.HUMAN
                or current.controller_id != controller_id
                or not secrets.compare_digest(current.token or "", token)
            ):
                raise ControlError("控制权凭据不匹配")
            before = self.get_state().board
            self._controls[side] = SideControl(ControllerKind.HUMAN)
            return self._finish_event(
                GameEventKind.CONTROL_CHANGED, before_board=before
            )

    def set_replay_cursor(
        self,
        ply: int,
        *,
        expected_version: int | None = None,
    ) -> ControllerState:
        with self._lock:
            self._check_version(expected_version)
            if not 0 <= ply <= len(self._base_record.moves):
                raise ControlError("回放游标超出棋谱范围")
            before = self.get_state().board
            self._replay_cursor = ply
            self._finish_event(GameEventKind.REPLAY_CHANGED, before_board=before)
            return self.get_state()

    def branch_from_replay(self, *, expected_version: int | None = None) -> GameEvent:
        with self._lock:
            self._check_version(expected_version)
            if self._replay_cursor is None:
                raise ControlError("当前不在回放模式")
            before = self.get_state().board
            count = self._replay_cursor
            self._base_record = self._base_record.model_copy(
                update={
                    "moves": self._base_record.moves[:count],
                    "result": ResultRecord(status="ongoing"),
                    "draw_events": self._draw_events_for_resume(count),
                }
            )
            self._current = self._replay(self._base_record, count)
            self._replay_cursor = None
            self._pending_draw = None
            return self._finish_event(GameEventKind.BRANCHED, before_board=before)

    def load_record(
        self,
        path: str | os.PathLike[str],
        *,
        expected_version: int | None = None,
    ) -> GameEvent:
        with self._lock:
            self._check_version(expected_version)
            loaded = load_and_validate(path)
            replacement = GameController.from_record(loaded)
            before = self.get_state().board
            self._base_record = replacement._base_record
            self._current = replacement._current
            self._pending_draw = replacement._pending_draw
            self._replay_cursor = None
            return self._finish_event(GameEventKind.RECORD_LOADED, before_board=before)

    def load_text_record(
        self,
        path: str | os.PathLike[str],
        *,
        expected_version: int | None = None,
    ) -> GameEvent:
        """Transactionally load vertical-line notation into this controller."""
        with self._lock:
            self._check_version(expected_version)
            text = Path(path).read_text(encoding="utf-8")
            initial_board = Board.from_fen(self._base_record.initial_fen)
            replayed = replay_text(
                text,
                initial_board=initial_board,
                side_to_move=self._base_record.initial_side,
            )
            board = initial_board
            side = self._base_record.initial_side
            moves: list[MoveRecord] = []
            for item in replayed.moves:
                after = board.move_unchecked(item.move.start, item.move.end)
                position = evaluate_position(after, side.opponent)
                moves.append(
                    MoveRecord.from_move(
                        item.move,
                        notation=item.notation,
                        position_after=item.position_after,
                        in_check=position.in_check,
                    )
                )
                board = after
                side = side.opponent
            candidate = self._base_record.model_copy(
                update={
                    "moves": tuple(moves),
                    "draw_events": (),
                    "result": ResultRecord(status="ongoing"),
                }
            )
            replacement = GameController.from_record(candidate)
            before = self.get_state().board
            self._base_record = replacement._base_record
            self._current = replacement._current
            self._pending_draw = replacement._pending_draw
            self._replay_cursor = None
            return self._finish_event(GameEventKind.RECORD_LOADED, before_board=before)

    def export_record(self, path: str | os.PathLike[str], format: str = "json") -> None:
        record = self.record
        if format == "json":
            export_json(record, path)
            return
        if format not in {"text", "txt", "notation"}:
            raise ControlError(f"不支持的棋谱格式: {format}")
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
                temporary.write("\n".join(move.notation for move in record.moves))
                temporary.write("\n" if record.moves else "")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, destination)
            temporary_name = None
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)

    def _check_version(self, expected: int | None) -> None:
        if expected is not None and expected != self._version:
            raise StaleVersionError(
                f"局面版本已变化: 预期 {expected}，当前 {self._version}"
            )

    def _authorize(self, side: Color, actor: str | None) -> None:
        control = self._controls[side]
        if control.kind is ControllerKind.HUMAN:
            if actor is not None:
                raise ControlError("该方当前由真人控制")
            return
        if actor is None or not secrets.compare_digest(control.token or "", actor):
            raise ControlError("没有该方控制权")

    def _adjudicator(self):
        if self._base_record.ruleset is Ruleset.CHINESE_2020:
            return Chinese2020Adjudicator()
        return Asian2003Adjudicator()

    @staticmethod
    def _continues_prohibited_cycle(
        prior_frames: tuple[PositionFrame, ...],
        decision: Adjudication | None,
        candidate: PositionFrame,
        candidate_decision: Adjudication,
    ) -> bool:
        if (
            decision is None
            or decision.kind is not AdjudicationKind.MUST_CHANGE
            or decision.responsible is not candidate.side
            or candidate_decision.kind is not AdjudicationKind.MUST_CHANGE
            or candidate_decision.responsible is not decision.responsible
        ):
            return False
        prior_keys = [
            prior_frames[0].before_key,
            *(frame.after_key for frame in prior_frames),
        ]
        return prior_keys.count(candidate.after_key) >= 2

    def _replay(self, record: GameRecord, count: int | None) -> _ReplayState:
        board = Board.from_fen(record.initial_fen)
        side = record.initial_side
        frames: list[PositionFrame] = []
        normalized: list[MoveRecord] = []
        prior_adjudication: Adjudication | None = None
        records = record.moves if count is None else record.moves[:count]
        for index, item in enumerate(records, 1):
            move = item.to_move()
            after = board.move_unchecked(move.start, move.end)
            frame = PositionFrame.from_transition(
                board, side, move, after, analyze_kill=False
            )
            frames.append(frame)
            board = after
            side = side.opponent
            position = evaluate_position(board, side)
            adjudicator = self._adjudicator()
            adjudication = adjudicator.evaluate(frames)
            if self._continues_prohibited_cycle(
                tuple(frames[:-1]),
                prior_adjudication,
                frame,
                adjudication,
            ):
                adjudication = adjudicator.loss_for_ignored_must_change(adjudication)
            if item.in_check != position.in_check:
                raise ControlError(f"第 {index} 步将军标记与重放局面不一致")
            expected_adjudication = self._adjudication_record(adjudication, len(frames))
            if (
                item.adjudication is not None
                and item.adjudication != expected_adjudication
            ):
                raise ControlError(f"第 {index} 步裁决信息与重放结果不一致")
            normalized.append(
                MoveRecord.from_move(
                    move,
                    notation=item.notation,
                    position_after=item.position_after,
                    in_check=position.in_check,
                    adjudication=expected_adjudication,
                )
            )
            prior_adjudication = adjudication
        position = evaluate_position(board, side)
        adjudication = (
            prior_adjudication
            if prior_adjudication is not None
            else self._adjudicator().evaluate(frames)
        )
        result = self._result_for(position, adjudication)
        return _ReplayState(
            board,
            side,
            tuple(frames),
            tuple(normalized),
            position,
            adjudication,
            result,
        )

    @staticmethod
    def _result_for(
        position: PositionResult, adjudication: Adjudication
    ) -> ResultRecord | None:
        if position.kind in (PositionKind.CHECKMATE, PositionKind.STALEMATE):
            assert position.winner is not None
            return ResultRecord(
                status=("red_win" if position.winner is Color.RED else "black_win"),
                reason=("将死" if position.kind is PositionKind.CHECKMATE else "困毙"),
                winner=position.winner,
            )
        if adjudication.kind is AdjudicationKind.DRAW:
            return ResultRecord(status="draw", reason=adjudication.reason)
        if (
            adjudication.kind is AdjudicationKind.LOSS
            and adjudication.responsible is not None
        ):
            winner = adjudication.responsible.opponent
            return ResultRecord(
                status="red_win" if winner is Color.RED else "black_win",
                reason=adjudication.reason,
                winner=winner,
            )
        return None

    def _record_from_current(self) -> GameRecord:
        return self._base_record.model_copy(
            update={
                "moves": self._current.records,
                "result": self._current.result or ResultRecord(status="ongoing"),
            }
        )

    def _draw_events_for_resume(self, ply: int) -> tuple[DrawEventRecord, ...]:
        events = [event for event in self._base_record.draw_events if event.ply <= ply]
        if events and events[-1].action == "accept":
            events = events[:-2]
        elif events and events[-1].action == "offer":
            events = events[:-1]
        return tuple(events)

    def _adjudication_record(
        self, adjudication: Adjudication, ply: int
    ) -> AdjudicationRecord:
        related_moves = (
            tuple(range(adjudication.cycle_start + 1, ply + 1))
            if adjudication.cycle_start is not None
            else ()
        )
        return AdjudicationRecord(
            kind=adjudication.kind,
            ruleset=adjudication.ruleset,
            cycle_start=adjudication.cycle_start,
            move_natures=adjudication.move_natures,
            responsible_natures=adjudication.responsible_natures,
            related_moves=related_moves,
            reason=adjudication.reason,
            responsible=adjudication.responsible,
            rule_reference=adjudication.rule_reference,
        )

    def _finish_event(
        self,
        kind: GameEventKind,
        *,
        before_board: Board,
        move: Move | None = None,
    ) -> GameEvent:
        self._version += 1
        state = self.get_state()
        event = GameEvent(
            kind=kind,
            version=self._version,
            before_board=before_board,
            after_board=state.board,
            move=move,
            next_side=state.side_to_move,
            in_check=state.position.in_check,
            checkmate=state.position.kind is PositionKind.CHECKMATE,
            stalemate=state.position.kind is PositionKind.STALEMATE,
            position=state.position,
            adjudication=state.adjudication,
            result=state.result,
            state=state,
        )
        for callback in tuple(self._callbacks):
            try:
                callback(event)
            except Exception as error:  # noqa: BLE001 - extension failures are isolated
                self._callback_errors.append(CallbackFailure(callback, event, error))
        return event
