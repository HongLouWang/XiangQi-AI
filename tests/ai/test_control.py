from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
from pathlib import Path

import pytest

from ai.checkpoint import TrainingProgress
from ai.control import RunControl, RunControlError, RunStatus


def _extend(run_dir: str, games: int, ready: multiprocessing.synchronize.Event) -> None:
    ready.wait()
    RunControl(run_dir).extend(games)


def _read_pause(
    run_dir: str,
    ready: multiprocessing.synchronize.Event,
    finished: multiprocessing.synchronize.Event,
    result: multiprocessing.Queue[bool],
) -> None:
    ready.set()
    result.put(RunControl(run_dir).pause_requested())
    finished.set()


def _running(*, target_games: int = 10) -> RunStatus:
    return RunStatus(
        phase="running",
        completed_games=4,
        target_games=target_games,
        training_steps=2,
        device="cpu",
    )


def test_pause_request_does_not_change_phase_until_safe_point(tmp_path: Path) -> None:
    control = RunControl(tmp_path)
    control.write_status(_running())

    control.request_pause()

    assert control.pause_requested()
    assert control.read_status().phase == "running"
    control.mark_paused(_running())
    assert control.read_status().phase == "paused"


def test_clear_pause_supports_resume_without_changing_progress(tmp_path: Path) -> None:
    control = RunControl(tmp_path)
    control.write_status(_running())
    control.request_pause()

    control.clear_pause()

    assert not control.pause_requested()
    assert control.read_status() == _running()


def test_pause_request_version_is_strictly_validated(tmp_path: Path) -> None:
    control = RunControl(tmp_path)
    control.pause_path.write_text(
        json.dumps({"schema_version": True, "requested": True}), encoding="utf-8"
    )

    with pytest.raises(RunControlError, match="暂停请求文件"):
        control.pause_requested()


def test_pause_read_waits_for_clear_lock_and_observes_missing_request(
    tmp_path: Path,
) -> None:
    control = RunControl(tmp_path)
    control.request_pause()
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    finished = context.Event()
    result = context.Queue()

    with control.lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        process = context.Process(
            target=_read_pause,
            args=(str(tmp_path), ready, finished, result),
        )
        process.start()
        assert ready.wait(timeout=5)
        assert not finished.wait(timeout=0.3)
        control.pause_path.unlink()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    process.join(timeout=5)
    assert process.exitcode == 0
    assert result.get(timeout=1) is False


def test_mark_paused_accepts_checkpoint_progress_and_preserves_runtime_fields(
    tmp_path: Path,
) -> None:
    control = RunControl(tmp_path)
    control.write_status(_running(target_games=12))

    paused = control.mark_paused(TrainingProgress(5, 12, 3))

    assert paused == RunStatus("paused", 5, 12, 3, "cpu")
    assert control.read_status() == paused


def test_extend_adds_to_target_without_resetting_progress(tmp_path: Path) -> None:
    control = RunControl(tmp_path)
    control.write_status(_running())

    updated = control.extend(5)

    assert updated.completed_games == 4
    assert updated.training_steps == 2
    assert updated.target_games == 15


def test_completion_handshake_refuses_stale_progress_after_extend(
    tmp_path: Path,
) -> None:
    control = RunControl(tmp_path)
    control.write_status(RunStatus("running", 1, 1, 1, "cpu"))
    control.extend(2)

    completed = control.try_mark_completed(
        TrainingProgress(1, 1, 1), device="cpu", message="workers=1"
    )

    assert not completed
    assert control.read_status() == RunStatus("running", 1, 3, 1, "cpu", "workers=1")


def test_completion_handshake_atomically_marks_current_target(tmp_path: Path) -> None:
    control = RunControl(tmp_path)
    control.write_status(RunStatus("running", 3, 3, 2, "cpu"))

    completed = control.try_mark_completed(
        TrainingProgress(3, 3, 2), device="cpu", message="workers=1"
    )

    assert completed
    assert control.read_status() == RunStatus("completed", 3, 3, 2, "cpu", "workers=1")


def test_stale_write_cannot_reduce_an_extended_target(tmp_path: Path) -> None:
    control = RunControl(tmp_path)
    stale = _running()
    control.write_status(stale)
    control.extend(5)

    control.write_status(stale)

    assert control.read_status().target_games == 15


def test_stale_progress_cannot_reduce_target_when_marking_paused(
    tmp_path: Path,
) -> None:
    control = RunControl(tmp_path)
    control.write_status(_running())
    control.extend(5)

    paused = control.mark_paused(TrainingProgress(4, 10, 2))

    assert paused.target_games == 15
    assert control.read_status().target_games == 15


def test_stale_running_status_cannot_pause_a_completed_run(tmp_path: Path) -> None:
    control = RunControl(tmp_path)
    stale = RunStatus("running", 4, 10, 2, "cpu")
    control.write_status(RunStatus("completed", 10, 10, 8, "cpu"))

    with pytest.raises(ValueError, match="completed.*暂停"):
        control.mark_paused(stale)

    assert control.read_status() == RunStatus("completed", 10, 10, 8, "cpu")


def test_stale_pause_snapshot_cannot_reduce_completed_games_or_steps(
    tmp_path: Path,
) -> None:
    control = RunControl(tmp_path)
    control.write_status(RunStatus("pausing", 6, 10, 5, "cpu"))

    paused = control.mark_paused(RunStatus("running", 4, 10, 2, "cpu"))

    assert paused == RunStatus("paused", 6, 10, 5, "cpu")


@pytest.mark.parametrize("games", [0, -1, 1.5, True])
def test_extend_requires_a_positive_integer(tmp_path: Path, games: object) -> None:
    control = RunControl(tmp_path)
    control.write_status(_running())

    with pytest.raises(ValueError, match="正整数"):
        control.extend(games)  # type: ignore[arg-type]


def test_two_concurrent_extends_are_accumulated(tmp_path: Path) -> None:
    RunControl(tmp_path).write_status(_running())
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    processes = [
        context.Process(target=_extend, args=(str(tmp_path), amount, ready))
        for amount in (3, 7)
    ]
    for process in processes:
        process.start()
    ready.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert RunControl(tmp_path).read_status().target_games == 20


def test_failed_replace_preserves_previous_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = RunControl(tmp_path)
    original = _running()
    control.write_status(original)

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        control.write_status(RunStatus("paused", 4, 10, 2, "cpu"))

    assert control.read_status() == original
    assert not list(tmp_path.glob(".status.json.*.tmp"))


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps({"schema_version": 999}),
        json.dumps(
            {
                "schema_version": True,
                "phase": "running",
                "completed_games": 0,
                "target_games": 1,
                "training_steps": 0,
                "device": "cpu",
                "message": "",
            }
        ),
        json.dumps(
            {
                "schema_version": 1,
                "phase": "running",
                "completed_games": 0,
                "target_games": 1,
                "training_steps": 0,
                "device": "cpu",
                "message": "",
                "unexpected": 1,
            }
        ),
    ],
)
def test_corrupt_or_incompatible_status_is_rejected(
    tmp_path: Path, payload: str
) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "status.json").write_text(payload, encoding="utf-8")

    with pytest.raises(RunControlError, match="状态文件"):
        RunControl(tmp_path).read_status()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phase": "unknown"},
        {"completed_games": -1},
        {"target_games": 3},
        {"training_steps": -1},
        {"device": ""},
        {"message": 1},
    ],
)
def test_status_fields_are_strictly_validated(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "phase": "running",
        "completed_games": 4,
        "target_games": 10,
        "training_steps": 2,
        "device": "cpu",
        "message": "",
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        RunStatus(**values)  # type: ignore[arg-type]


def test_completed_phase_requires_progress_to_reach_target() -> None:
    with pytest.raises(ValueError, match="completed"):
        RunStatus("completed", 4, 10, 2, "cpu")


def test_new_phase_requires_zero_progress() -> None:
    with pytest.raises(ValueError, match="new"):
        RunStatus("new", 0, 10, 1, "cpu")


@pytest.mark.parametrize("phase", ["new", "completed", "failed"])
def test_mark_paused_rejects_invalid_source_phase(tmp_path: Path, phase: str) -> None:
    control = RunControl(tmp_path)
    status = RunStatus(
        phase=phase,  # type: ignore[arg-type]
        completed_games=10 if phase == "completed" else 0,
        target_games=10,
        training_steps=0,
        device="cpu",
    )
    control.write_status(status)

    with pytest.raises(ValueError, match="暂停"):
        control.mark_paused(status)


def test_constructor_creates_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "nested" / "run"
    RunControl(run_dir)
    assert run_dir.is_dir()
