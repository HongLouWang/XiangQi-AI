"""Desktop application startup and local API lifecycle."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import Protocol

import uvicorn
from fastapi import FastAPI
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from xiangqi.api import create_api
from xiangqi.controller import GameController
from xiangqi.ui.main_window import MainWindow

LOCAL_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8765


class Server(Protocol):
    should_exit: bool

    def run(self) -> None: ...


ServerFactory = Callable[..., Server]


def _uvicorn_server(
    app: FastAPI,
    *,
    host: str,
    port: int,
) -> Server:
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    return uvicorn.Server(config)


class ApiServerThread(QThread):
    """Run a stoppable ASGI server away from Qt's GUI thread."""

    def __init__(self, server: Server) -> None:
        super().__init__()
        self.server = server

    def run(self) -> None:
        self.server.run()

    def request_stop(self) -> None:
        self.server.should_exit = True


class DesktopRuntime:
    """Own the shared controller, window and optional local API thread."""

    def __init__(
        self,
        *,
        api_enabled: bool = True,
        port: int = DEFAULT_API_PORT,
        server_factory: ServerFactory = _uvicorn_server,
    ) -> None:
        if not 1 <= port <= 65535:
            raise ValueError("端口必须在 1 到 65535 之间")
        self.controller = GameController.new()
        self.window = MainWindow(self.controller)
        self.api_thread: ApiServerThread | None = None
        self._shutting_down = False
        if api_enabled:
            server = server_factory(
                create_api(self.controller),
                host=LOCAL_API_HOST,
                port=port,
            )
            self.api_thread = ApiServerThread(server)
        self.window.closing.connect(self.shutdown)

    def start(self) -> None:
        if self.api_thread is not None and not self.api_thread.isRunning():
            self.api_thread.start()
        self.window.show()

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self.window.stop_replay()
        if self.api_thread is not None and self.api_thread.isRunning():
            self.api_thread.request_stop()
            self.api_thread.wait(3000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="中国象棋桌面游戏")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_API_PORT,
        help=f"本机 API 端口（默认 {DEFAULT_API_PORT}）",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="不启动 HTTP/WebSocket 控制接口",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    application = QApplication.instance() or QApplication(sys.argv[:1])
    runtime = DesktopRuntime(api_enabled=not args.no_api, port=args.port)
    application.aboutToQuit.connect(runtime.shutdown)
    runtime.start()
    try:
        return application.exec()
    finally:
        runtime.shutdown()
