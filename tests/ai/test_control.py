from __future__ import annotations

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


def test_constructor_creates_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "nested" / "run"
    RunControl(run_dir)
    assert run_dir.is_dir()
