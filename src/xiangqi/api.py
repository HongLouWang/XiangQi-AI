"""FastAPI protocol adapter for a :class:`GameController`.

The application deliberately does not choose a listening address.  Desktop
startup code owns that policy and binds the server to ``127.0.0.1``.
"""

from __future__ import annotations

import asyncio
import queue
import secrets
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from xiangqi.controller import (
    ControlError,
    ControllerKind,
    ControllerState,
    GameController,
    GameEvent,
    StaleVersionError,
)
from xiangqi.domain import Color, Coord
from xiangqi.notation import replay_text
from xiangqi.record import MoveRecord
from xiangqi.rules import evaluate_position


class ControllerHub:
    """Thread-safe, replaceable reference to the active game controller."""

    def __init__(self, controller: GameController) -> None:
        self._current = controller
        self._lock = RLock()
        self._listeners: list[Any] = []

    @property
    def current(self) -> GameController:
        with self._lock:
            return self._current

    def replace(self, controller: GameController) -> None:
        with self._lock:
            if controller is self._current:
                return
            self._current = controller
            listeners = tuple(self._listeners)
        for listener in listeners:
            listener(controller)

    def subscribe(self, listener: Any) -> None:
        with self._lock:
            self._listeners.append(listener)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.current, name)


class _Command(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    controller_id: str = Field(min_length=1)
    token: str | None = None
    expected_version: int = Field(ge=0)


class MoveCommand(_Command):
    start: tuple[int, int]
    end: tuple[int, int]


class UndoCommand(_Command):
    steps: int = Field(default=1, ge=1)


class DrawOfferCommand(_Command):
    side: Color


class DrawResponseCommand(_Command):
    side: Color
    accept: bool


class SideControlCommand(_Command):
    side: Color


class RecordImportCommand(_Command):
    path: str = Field(min_length=1)
    format: Literal["json", "text", "txt", "notation"] = "json"


class RecordExportCommand(_Command):
    path: str = Field(min_length=1)
    format: Literal["json", "text", "txt", "notation"] = "json"


class EventBroker:
    """Fan out controller callbacks without ever waiting for a client."""

    def __init__(self, queue_size: int = 64) -> None:
        if queue_size < 1:
            raise ValueError("event_queue_size 必须大于零")
        self._queue_size = queue_size
        self._subscribers: set[queue.Queue[Any]] = set()
        self._lock = RLock()

    def subscribe(self) -> queue.Queue[Any]:
        subscriber: queue.Queue[Any] = queue.Queue(self._queue_size)
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[Any]) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def publish(self, event: Any) -> None:
        # Replacing the oldest item bounds memory and guarantees that the newest
        # transition, especially a terminal result, is never silently discarded.
        with self._lock:
            for subscriber in tuple(self._subscribers):
                try:
                    subscriber.put_nowait(event)
                except queue.Full:
                    try:
                        queued = subscriber.get_nowait()
                    except queue.Empty:
                        queued = None
                    if (
                        queued is not None
                        and getattr(queued, "result", None) is not None
                        and getattr(event, "result", None) is None
                    ):
                        subscriber.put_nowait(queued)
                        continue
                    subscriber.put_nowait(event)


def _coord(value: tuple[int, int]) -> Coord:
    try:
        return Coord(*value)
    except (TypeError, ValueError) as error:
        raise ControlError(str(error)) from error


def _result(value: Any) -> dict[str, Any] | None:
    return None if value is None else value.model_dump(mode="json")


def _adjudication(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "kind": value.kind.value,
        "ruleset": value.ruleset.value,
        "cycle_start": value.cycle_start,
        "reason": value.reason,
        "responsible": (
            None if value.responsible is None else value.responsible.value
        ),
        "move_natures": [nature.value for nature in value.move_natures],
        "responsible_natures": [
            nature.value for nature in value.responsible_natures
        ],
        "rule_reference": value.rule_reference,
    }


def _board(board: Any) -> dict[str, Any]:
    return {
        f"{coord.file},{coord.rank}": {
            "color": piece.color.value,
            "kind": piece.kind.value,
        }
        for coord, piece in board.pieces.items()
    }


def _state(state: ControllerState) -> dict[str, Any]:
    return {
        "board": _board(state.board),
        "fen": state.board.to_fen(),
        "side_to_move": state.side_to_move.value,
        "ruleset": state.ruleset.value,
        "ply": state.ply,
        "version": state.version,
        "position": {
            "kind": state.position.kind.value,
            "side_to_move": state.position.side_to_move.value,
            "winner": (
                None
                if state.position.winner is None
                else state.position.winner.value
            ),
            "in_check": state.position.in_check,
        },
        "result": _result(state.result),
        "pending_draw": (
            None if state.pending_draw is None else state.pending_draw.value
        ),
        "replay_cursor": state.replay_cursor,
        "last_move": (
            None if state.last_move is None else state.last_move.to_dict()
        ),
        "adjudication": _adjudication(state.adjudication),
        "controllers": {
            side.value: {
                "kind": control.kind.value,
                "controller_id": control.controller_id,
            }
            for side, control in state.controllers.items()
        },
    }


def _event(event: GameEvent) -> dict[str, Any]:
    return {
        "kind": event.kind.value,
        "version": event.version,
        "before_board": _board(event.before_board),
        "after_board": _board(event.after_board),
        "move": None if event.move is None else event.move.to_dict(),
        "next_side": event.next_side.value,
        "in_check": event.in_check,
        "checkmate": event.checkmate,
        "stalemate": event.stalemate,
        "adjudication": _adjudication(event.adjudication),
        "result": _result(event.result),
    }


def _success(
    controller: GameController,
    *,
    event: GameEvent | None = None,
    request_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    state = controller.get_state()
    body = {
        "request_id": request_id,
        "version": state.version,
        "state": _state(state),
        **extra,
    }
    if event is not None:
        body["event"] = _event(event)
    return body


def _error(code: str, message: str, details: Any = None) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details}


def _verify_version(controller: GameController, expected: int) -> None:
    current = controller.get_state().version
    if expected != current:
        raise StaleVersionError(
            f"局面版本已变化: 预期 {expected}，当前 {current}"
        )


def _claim_side(
    controller: GameController, side: Color, command: _Command
) -> dict[str, Any]:
    lease = controller.claim_side(
        side,
        command.controller_id,
        ControllerKind.NETWORK,
        expected_version=command.expected_version,
    )
    return _success(
        controller,
        request_id=command.request_id,
        token=lease.token,
    )


def _release_side(
    controller: GameController, side: Color, command: _Command
) -> dict[str, Any]:
    event = controller.release_side(
        side,
        command.controller_id,
        command.token or "",
        expected_version=command.expected_version,
    )
    return _success(
        controller, event=event, request_id=command.request_id
    )


def _move(
    controller: GameController, command: MoveCommand
) -> dict[str, Any]:
    _authorize_identity(
        controller, command, controller.get_state().side_to_move
    )
    event = controller.make_move(
        _coord(command.start),
        _coord(command.end),
        actor=command.token,
        expected_version=command.expected_version,
    )
    return _success(
        controller, event=event, request_id=command.request_id
    )


def _undo(
    controller: GameController, command: UndoCommand
) -> dict[str, Any]:
    _authorize_identity(controller, command)
    event = controller.undo(
        command.steps, expected_version=command.expected_version
    )
    return _success(
        controller, event=event, request_id=command.request_id
    )


def _offer_draw(
    controller: GameController, command: DrawOfferCommand
) -> dict[str, Any]:
    _authorize_identity(controller, command, command.side)
    event = controller.offer_draw(
        command.side,
        control_token=command.token,
        expected_version=command.expected_version,
    )
    return _success(
        controller, event=event, request_id=command.request_id
    )


def _respond_draw(
    controller: GameController, command: DrawResponseCommand
) -> dict[str, Any]:
    _authorize_identity(controller, command, command.side)
    event = controller.respond_draw(
        command.side,
        command.accept,
        control_token=command.token,
        expected_version=command.expected_version,
    )
    return _success(
        controller, event=event, request_id=command.request_id
    )


def _export_record(
    controller: GameController, command: RecordExportCommand
) -> dict[str, Any]:
    _authorize_identity(controller, command)
    _verify_version(controller, command.expected_version)
    controller.export_record(command.path, command.format)
    return _success(
        controller, request_id=command.request_id, path=command.path
    )


def _authorize_identity(
    controller: GameController,
    command: _Command,
    side: Color | None = None,
) -> None:
    controls = controller.get_state().controllers
    candidates = (
        (controls[side],) if side is not None else tuple(controls.values())
    )
    external = [
        control
        for control in candidates
        if control.kind is not ControllerKind.HUMAN
    ]
    if not external:
        raise ControlError("网络客户端尚未取得控制权")
    for control in external:
        if (
            control.controller_id == command.controller_id
            and command.token is not None
            and secrets.compare_digest(control.token or "", command.token)
        ):
            return
    raise ControlError("控制权凭据不匹配")


def create_api(
    controller: GameController | ControllerHub, *, event_queue_size: int = 64
) -> FastAPI:
    """Create an unbound ASGI application around ``controller``."""

    hub = (
        controller
        if isinstance(controller, ControllerHub)
        else ControllerHub(controller)
    )
    controller = hub
    app = FastAPI(title="中国象棋本机控制接口", version="1.0")
    broker = EventBroker(event_queue_size)
    controller.register_callback(broker.publish)
    hub.subscribe(lambda active: active.register_callback(broker.publish))
    app.state.controller = controller
    app.state.controller_hub = hub
    app.state.event_broker = broker
    processed_requests: set[tuple[str, str]] = set()
    request_lock = RLock()

    def execute_once(command: _Command, action: Any) -> dict[str, Any]:
        key = (command.controller_id, command.request_id)
        with request_lock:
            if key in processed_requests:
                raise ControlError(
                    f"重复 request_id: {command.request_id}"
                )
            processed_requests.add(key)
        return action()

    def import_into_active(command: RecordImportCommand) -> dict[str, Any]:
        _authorize_identity(controller, command)
        if command.format == "json":
            event = controller.load_record(
                command.path, expected_version=command.expected_version
            )
            return _success(
                controller, event=event, request_id=command.request_id
            )
        _verify_version(controller, command.expected_version)
        text = Path(command.path).read_text(encoding="utf-8")
        replayed = replay_text(text)
        base = GameController.new().record
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
        replacement = GameController.from_record(
            base.model_copy(update={"moves": tuple(moves)})
        )
        hub.replace(replacement)
        return _success(
            controller,
            request_id=command.request_id,
            path=command.path,
        )

    @app.exception_handler(StaleVersionError)
    async def stale_error(
        _request: Request, error: StaleVersionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content=_error("stale_version", str(error)),
        )

    @app.exception_handler(ControlError)
    async def control_error(
        _request: Request, error: ControlError
    ) -> JSONResponse:
        message = str(error)
        illegal = "非法着法" in message or "坐标越界" in message
        return JSONResponse(
            status_code=422 if illegal else 409,
            content=_error(
                "invalid_command" if illegal else "control_error", message
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error(
                "validation_error",
                "请求参数无效",
                [
                    {
                        "location": list(item["loc"]),
                        "message": item["msg"],
                        "type": item["type"],
                    }
                    for item in error.errors()
                ],
            ),
        )

    @app.exception_handler(ValueError)
    async def value_error(
        _request: Request, error: ValueError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error("invalid_command", str(error)),
        )

    @app.get("/state")
    def get_state() -> dict[str, Any]:
        return _state(controller.get_state())

    @app.get("/legal-moves")
    def get_legal_moves(side: Color | None = None) -> dict[str, Any]:
        moves = controller.get_legal_moves(side)
        return {
            "side": (side or controller.get_state().side_to_move).value,
            "version": controller.get_state().version,
            "moves": {
                f"{start.file},{start.rank}": [
                    [end.file, end.rank] for end in destinations
                ]
                for start, destinations in moves.items()
            },
        }

    @app.post("/control/{side}/claim")
    def claim(side: Color, command: _Command) -> dict[str, Any]:
        return execute_once(
            command,
            lambda: _claim_side(controller, side, command),
        )

    @app.post("/control/{side}/release")
    def release(side: Color, command: _Command) -> dict[str, Any]:
        return execute_once(
            command,
            lambda: _release_side(controller, side, command),
        )

    @app.post("/move")
    def move(command: MoveCommand) -> dict[str, Any]:
        return execute_once(
            command,
            lambda: _move(controller, command),
        )

    @app.post("/undo")
    def undo(command: UndoCommand) -> dict[str, Any]:
        return execute_once(
            command,
            lambda: _undo(controller, command),
        )

    @app.post("/draw/offer")
    def offer_draw(command: DrawOfferCommand) -> dict[str, Any]:
        return execute_once(
            command,
            lambda: _offer_draw(controller, command),
        )

    @app.post("/draw/respond")
    def respond_draw(command: DrawResponseCommand) -> dict[str, Any]:
        return execute_once(
            command,
            lambda: _respond_draw(controller, command),
        )

    @app.post("/record/import")
    def import_record(command: RecordImportCommand) -> dict[str, Any]:
        return execute_once(
            command,
            lambda: import_into_active(command),
        )

    @app.post("/record/export")
    def export_record(command: RecordExportCommand) -> dict[str, Any]:
        return execute_once(
            command,
            lambda: _export_record(controller, command),
        )

    command_models: Mapping[str, type[_Command]] = {
        "claim": SideControlCommand,
        "release": SideControlCommand,
        "move": MoveCommand,
        "undo": UndoCommand,
        "offer_draw": DrawOfferCommand,
        "respond_draw": DrawResponseCommand,
        "import_record": RecordImportCommand,
        "export_record": RecordExportCommand,
    }

    def execute_ws(raw: dict[str, Any]) -> dict[str, Any]:
        command_name = raw.pop("command", None)
        model = command_models.get(command_name)
        if model is None:
            raise ControlError(f"不支持的 WebSocket 命令: {command_name}")
        command = model.model_validate(raw)
        def dispatch() -> dict[str, Any]:
            if isinstance(command, SideControlCommand):
                return (
                    _claim_side(controller, command.side, command)
                    if command_name == "claim"
                    else _release_side(controller, command.side, command)
                )
            if isinstance(command, MoveCommand):
                return _move(controller, command)
            if isinstance(command, UndoCommand):
                return _undo(controller, command)
            if isinstance(command, DrawOfferCommand):
                return _offer_draw(controller, command)
            if isinstance(command, DrawResponseCommand):
                return _respond_draw(controller, command)
            if isinstance(command, RecordImportCommand):
                return import_into_active(command)
            assert isinstance(command, RecordExportCommand)
            return _export_record(controller, command)

        return execute_once(command, dispatch)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        async def next_event(events: queue.Queue[Any]) -> Any:
            while True:
                try:
                    return events.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)

        await websocket.accept()
        events = broker.subscribe()
        await websocket.send_json(
            {
                "type": "ready",
                "version": controller.get_state().version,
            }
        )
        receive_task: asyncio.Task[dict[str, Any]] | None = asyncio.create_task(
            websocket.receive_json()
        )
        event_task: asyncio.Task[Any] | None = asyncio.create_task(
            next_event(events)
        )
        try:
            while True:
                assert receive_task is not None and event_task is not None
                done, _ = await asyncio.wait(
                    {receive_task, event_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if receive_task in done:
                    request_id: str | None = None
                    try:
                        raw = receive_task.result()
                        if not isinstance(raw, dict):
                            raise TypeError("WebSocket 命令必须是 JSON 对象")
                        request_id = raw.get("request_id")
                        body = execute_ws(dict(raw))
                        await websocket.send_json(
                            {
                                "type": "response",
                                "request_id": request_id,
                                "ok": True,
                                **body,
                            }
                        )
                    except WebSocketDisconnect:
                        break
                    except StaleVersionError as error:
                        await websocket.send_json(
                            {
                                "type": "response",
                                "request_id": request_id,
                                "ok": False,
                                **_error("stale_version", str(error)),
                            }
                        )
                    except (
                        AttributeError,
                        ControlError,
                        TypeError,
                        ValueError,
                    ) as error:
                        await websocket.send_json(
                            {
                                "type": "response",
                                "request_id": request_id,
                                "ok": False,
                                **_error("invalid_command", str(error)),
                            }
                        )
                    receive_task = asyncio.create_task(
                        websocket.receive_json()
                    )
                if event_task in done:
                    event = event_task.result()
                    await websocket.send_json(
                        {"type": "event", "event": _event(event)}
                    )
                    event_task = asyncio.create_task(
                        next_event(events)
                    )
        finally:
            broker.unsubscribe(events)
            for task in (receive_task, event_task):
                if task is not None:
                    task.cancel()

    return app
