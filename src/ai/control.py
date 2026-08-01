from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

STATUS_SCHEMA_VERSION = 1
PAUSE_SCHEMA_VERSION = 1

RunPhase = Literal["new", "running", "pausing", "paused", "completed", "failed"]
_PHASES = frozenset({"new", "running", "pausing", "paused", "completed", "failed"})
_STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "phase",
        "completed_games",
        "target_games",
        "training_steps",
        "device",
        "message",
    }
)


class RunControlError(RuntimeError):
    """训练控制文件缺失、损坏或与当前版本不兼容。"""


class TrainingProgressLike(Protocol):
    completed_games: int
    target_games: int
    training_steps: int


@dataclass(frozen=True, slots=True)
class RunStatus:
    phase: RunPhase
    completed_games: int
    target_games: int
    training_steps: int
    device: str
    message: str = ""

    def __post_init__(self) -> None:
        if type(self.phase) is not str or self.phase not in _PHASES:
            raise ValueError(f"未知训练阶段：{self.phase!r}")
        for name in ("completed_games", "target_games", "training_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} 必须是非负整数")
        if self.target_games < self.completed_games:
            raise ValueError("target_games 不能小于 completed_games")
        if type(self.device) is not str or not self.device:
            raise ValueError("device 必须是非空字符串")
        if type(self.message) is not str:
            raise TypeError("message 必须是字符串")


class RunControl:
    """以原子文件和进程锁维护一个训练运行的控制状态。"""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.run_dir / "status.json"
        self.pause_path = self.run_dir / "pause.json"
        self.lock_path = self.run_dir / "control.lock"

    def request_pause(self) -> None:
        with self._locked():
            self._atomic_write_json(
                self.pause_path,
                {"schema_version": PAUSE_SCHEMA_VERSION, "requested": True},
            )

    def pause_requested(self) -> bool:
        if not self.pause_path.exists():
            return False
        payload = self._read_json(self.pause_path, "暂停请求文件")
        if (
            frozenset(payload) != {"schema_version", "requested"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != PAUSE_SCHEMA_VERSION
            or payload["requested"] is not True
        ):
            raise RunControlError("暂停请求文件结构或版本无效")
        return True

    def clear_pause(self) -> None:
        with self._locked():
            try:
                self.pause_path.unlink()
            except FileNotFoundError:
                return
            self._fsync_directory()

    def read_status(self) -> RunStatus:
        payload = self._read_json(self.status_path, "状态文件")
        if frozenset(payload) != _STATUS_FIELDS:
            raise RunControlError("状态文件字段无效")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != STATUS_SCHEMA_VERSION
        ):
            raise RunControlError("状态文件版本不兼容")
        try:
            return RunStatus(
                phase=payload["phase"],
                completed_games=payload["completed_games"],
                target_games=payload["target_games"],
                training_steps=payload["training_steps"],
                device=payload["device"],
                message=payload["message"],
            )
        except (TypeError, ValueError) as error:
            raise RunControlError(f"状态文件字段值无效：{error}") from error

    def write_status(self, status: RunStatus) -> None:
        if not isinstance(status, RunStatus):
            raise TypeError("status 必须是 RunStatus")
        with self._locked():
            self._write_status_unlocked(status)

    def mark_paused(self, progress: RunStatus | TrainingProgressLike) -> RunStatus:
        with self._locked():
            if isinstance(progress, RunStatus):
                paused = replace(progress, phase="paused")
            else:
                current = self.read_status()
                try:
                    paused = RunStatus(
                        phase="paused",
                        completed_games=progress.completed_games,
                        target_games=progress.target_games,
                        training_steps=progress.training_steps,
                        device=current.device,
                        message=current.message,
                    )
                except AttributeError as error:
                    raise TypeError(
                        "progress 必须包含 completed_games、target_games 和 training_steps"
                    ) from error
            self._write_status_unlocked(paused)
            return paused

    def extend(self, games: int) -> RunStatus:
        if type(games) is not int or games <= 0:
            raise ValueError("追加局数必须是正整数")
        with self._locked():
            current = self.read_status()
            updated = replace(current, target_games=current.target_games + games)
            self._write_status_unlocked(updated)
            return updated

    def _write_status_unlocked(self, status: RunStatus) -> None:
        payload = {"schema_version": STATUS_SCHEMA_VERSION, **asdict(status)}
        self._atomic_write_json(self.status_path, payload)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self.lock_path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _atomic_write_json(self, destination: Path, payload: object) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.run_dir,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
            self._fsync_directory()
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _read_json(self, path: Path, label: str) -> dict[str, object]:
        try:
            with path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except FileNotFoundError as error:
            raise RunControlError(f"{label}不存在：{path}") from error
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RunControlError(f"{label}损坏或无法读取：{error}") from error
        if not isinstance(payload, dict):
            raise RunControlError(f"{label}必须是 JSON 对象")
        return payload

    def _fsync_directory(self) -> None:
        descriptor = os.open(self.run_dir, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
