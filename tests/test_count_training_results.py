from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import ai.count_training_results as counter
from ai.count_training_results import TrainingDataError, count_results, main


def _write_game(
    run_dir: Path,
    game_id: int,
    *,
    values: list[float],
    sides: list[int],
) -> None:
    game_path = run_dir / "replay" / "games" / f"{game_id:012d}.npz"
    game_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        game_path,
        values=np.asarray(values, dtype=np.float32),
        sides=np.asarray(sides, dtype=np.int8),
        plies=np.asarray([len(values)], dtype=np.int64),
    )


def _write_run(run_dir: Path) -> None:
    replay_dir = run_dir / "replay"
    replay_dir.mkdir(parents=True)
    (replay_dir / "manifest.json").write_text(
        json.dumps({"games": [1, 2, 3, 4], "total_games": 4})
    )
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "phase": "running",
                "completed_games": 4,
                "target_games": 10_000,
                "training_steps": 3,
            }
        )
    )
    _write_game(run_dir, 1, values=[1, -1], sides=[0, 1])
    _write_game(run_dir, 2, values=[-1, 1], sides=[0, 1])
    _write_game(run_dir, 3, values=[0, 0], sides=[0, 1])
    broken = replay_dir / "games" / "000000000004.npz"
    np.savez_compressed(broken, values=np.asarray([1], dtype=np.float32))


def test_count_results_classifies_games_and_isolates_invalid_file(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    report = count_results(run_dir)

    assert report.phase == "running"
    assert report.completed_games == 4
    assert report.target_games == 10_000
    assert report.training_steps == 3
    assert report.red_wins == 1
    assert report.black_wins == 1
    assert report.draws == 1
    assert report.invalid_games == 1
    assert report.classified_games == 3
    assert report.retained_games == 4
    assert report.total_games == 4
    assert report.errors[0].game_id == 4
    assert "sides" in report.errors[0].reason


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ("not-json", "JSON"),
        (json.dumps({"games": [1, 1], "total_games": 2}), "重复"),
        (json.dumps({"games": [True], "total_games": 1}), "棋局 ID"),
        (json.dumps({"games": [1, 2], "total_games": 1}), "total_games"),
    ],
)
def test_count_results_rejects_invalid_manifest(
    tmp_path: Path, manifest: str, message: str
) -> None:
    run_dir = tmp_path / "run"
    replay_dir = run_dir / "replay"
    replay_dir.mkdir(parents=True)
    (replay_dir / "manifest.json").write_text(manifest)

    with pytest.raises(TrainingDataError, match=message):
        count_results(run_dir)


def test_count_results_rejects_missing_run_directory(tmp_path: Path) -> None:
    with pytest.raises(TrainingDataError, match="训练目录不存在"):
        count_results(tmp_path / "missing")


def test_count_results_allows_missing_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "replay").mkdir(parents=True)
    (run_dir / "replay" / "manifest.json").write_text(
        json.dumps({"games": [], "total_games": 0})
    )

    report = count_results(run_dir)

    assert report.phase == "unknown"
    assert report.completed_games is None
    assert report.target_games is None
    assert report.training_steps is None


def test_count_results_rejects_game_with_invalid_plies(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    games_dir = run_dir / "replay" / "games"
    games_dir.mkdir(parents=True)
    (run_dir / "replay" / "manifest.json").write_text(
        json.dumps({"games": [1], "total_games": 1})
    )
    np.savez_compressed(
        games_dir / "000000000001.npz",
        values=np.asarray([1, -1], dtype=np.float32),
        sides=np.asarray([0, 1], dtype=np.int8),
        plies=np.asarray([1], dtype=np.int64),
    )

    report = count_results(run_dir)

    assert report.classified_games == 0
    assert report.invalid_games == 1
    assert "plies" in report.errors[0].reason


def test_count_results_retries_when_manifest_changes_during_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    replay_dir = run_dir / "replay"
    replay_dir.mkdir(parents=True)
    manifest_path = replay_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"games": [1], "total_games": 1}))
    _write_game(run_dir, 1, values=[1, -1], sides=[0, 1])
    _write_game(run_dir, 2, values=[-1, 1], sides=[0, 1])
    original = counter._classify_game
    changed = False

    def change_manifest_during_first_scan(path: Path) -> str:
        nonlocal changed
        if not changed:
            changed = True
            manifest_path.write_text(json.dumps({"games": [2], "total_games": 2}))
            path.unlink()
        return original(path)

    monkeypatch.setattr(counter, "_classify_game", change_manifest_during_first_scan)

    report = count_results(run_dir)

    assert report.total_games == 2
    assert report.retained_games == 1
    assert report.red_wins == 0
    assert report.black_wins == 1
    assert report.invalid_games == 0


def test_count_results_rejects_persistently_mixed_manifest_and_status_snapshot(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "replay").mkdir(parents=True)
    (run_dir / "replay" / "manifest.json").write_text(
        json.dumps({"games": [], "total_games": 1})
    )
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "phase": "running",
                "completed_games": 0,
                "target_games": 10,
                "training_steps": 0,
            }
        )
    )

    with pytest.raises(TrainingDataError, match="稳定快照"):
        count_results(run_dir)


@pytest.mark.parametrize(
    "status",
    [
        "not-json",
        json.dumps([]),
        json.dumps(
            {
                "phase": "running",
                "completed_games": True,
                "target_games": 10,
                "training_steps": 0,
            }
        ),
    ],
)
def test_count_results_rejects_invalid_status(tmp_path: Path, status: str) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "replay").mkdir(parents=True)
    (run_dir / "replay" / "manifest.json").write_text(
        json.dumps({"games": [], "total_games": 0})
    )
    (run_dir / "status.json").write_text(status)

    with pytest.raises(TrainingDataError):
        count_results(run_dir)


def test_main_default_run_dir_is_based_on_ai_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "AI-runs" / "cpu-main"
    (run_dir / "replay").mkdir(parents=True)
    (run_dir / "replay" / "manifest.json").write_text(
        json.dumps({"games": [], "total_games": 0})
    )
    monkeypatch.setattr(counter, "AI_ROOT", tmp_path)

    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"训练目录：{run_dir}" in captured.out


def test_main_prints_chinese_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    exit_code = main(["--run-dir", str(run_dir)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "红方胜：1 局" in captured.out
    assert "黑方胜：1 局" in captured.out
    assert "和棋：1 局" in captured.out
    assert "异常：1 局" in captured.out
    assert "棋局 4" in captured.out
    assert captured.err == ""


def test_main_reports_controlled_error_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["--run-dir", str(tmp_path / "missing")])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "统计失败：训练目录不存在" in captured.err
    assert "Traceback" not in captured.err
