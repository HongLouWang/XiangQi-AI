from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

from ai.checkpoint import CheckpointManager, TrainingProgress
from ai.cli import build_parser, main
from ai.config import TrainingConfig
from ai.control import RunControl, RunStatus
from ai.network import PolicyValueNetwork


@dataclass
class _CapturedTrainer:
    config: TrainingConfig | None = None
    resume: bool | None = None


def _capture_trainer(monkeypatch: pytest.MonkeyPatch) -> _CapturedTrainer:
    captured = _CapturedTrainer()

    class FakeTrainer:
        def __init__(self, config: TrainingConfig) -> None:
            captured.config = config

        def run(self, *, resume: bool = False) -> None:
            captured.resume = resume

    monkeypatch.setattr("ai.cli.Trainer", FakeTrainer)
    return captured


def _seed_status(run_dir: Path, *, completed: int = 3, target: int = 10) -> None:
    RunControl(run_dir).write_status(RunStatus("running", completed, target, 2, "cpu"))


def _seed_checkpoint(run_dir: Path, config: TrainingConfig) -> None:
    model = PolicyValueNetwork(
        channels=config.channels, residual_blocks=config.residual_blocks
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    CheckpointManager(run_dir).save(
        model,
        optimizer,
        TrainingProgress(3, config.target_games, 2),
        config,
        replay_manifest_hash="manifest",
        replay_manifest_version=1,
    )


def test_train_defaults_are_10000_games_and_512_full_moves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _capture_trainer(monkeypatch)

    assert main(["train", "--run-dir", str(tmp_path)]) == 0

    assert captured.config is not None
    assert captured.config.target_games == 10_000
    assert captured.config.max_full_moves == 512
    assert captured.config.max_plies == 1024
    assert captured.config.parallel_games == 16
    assert captured.resume is False


def test_train_help_explains_cuda_worker_and_parallel_game_limits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["train", "--help"]) == 0

    output = capsys.readouterr().out
    assert "CPU 生产进程" in output
    assert "最多在途棋局" in output


def test_train_accepts_runtime_and_network_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _capture_trainer(monkeypatch)

    result = main(
        [
            "train",
            "--run-dir",
            str(tmp_path),
            "--games",
            "12",
            "--full-moves",
            "5",
            "--device",
            "cpu",
            "--torch-threads",
            "3",
            "--self-play-workers",
            "2",
            "--parallel-games",
            "8",
            "--simulations",
            "7",
            "--channels",
            "16",
            "--residual-blocks",
            "2",
            "--batch-size",
            "8",
            "--replay-capacity-games",
            "40",
            "--learning-rate",
            "0.002",
            "--checkpoint-interval-games",
            "4",
            "--game-retry-limit",
            "5",
            "--seed",
            "9",
        ]
    )

    assert result == 0
    assert captured.config == TrainingConfig(
        target_games=12,
        max_full_moves=5,
        device="cpu",
        torch_threads=3,
        self_play_workers=2,
        parallel_games=8,
        simulations_per_move=7,
        residual_blocks=2,
        channels=16,
        batch_size=8,
        replay_capacity_games=40,
        learning_rate=0.002,
        checkpoint_interval_games=4,
        game_retry_limit=5,
        seed=9,
        run_dir=tmp_path,
    )


def test_resume_uses_persisted_target_and_architecture_without_cli_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    saved = TrainingConfig(
        target_games=7,
        max_full_moves=11,
        channels=2,
        residual_blocks=1,
        simulations_per_move=3,
        run_dir=tmp_path,
    )
    _seed_checkpoint(tmp_path, saved)
    _seed_status(tmp_path, target=12)
    captured = _capture_trainer(monkeypatch)

    assert main(["resume", "--run-dir", str(tmp_path)]) == 0

    assert captured.config == saved
    assert captured.config.target_games != 10_000
    assert captured.resume is True


def test_resume_only_overrides_device_threads_workers_and_parallel_games(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    saved = TrainingConfig(
        target_games=7,
        max_full_moves=11,
        channels=2,
        residual_blocks=1,
        run_dir=tmp_path,
    )
    _seed_checkpoint(tmp_path, saved)
    captured = _capture_trainer(monkeypatch)

    assert (
        main(
            [
                "resume",
                "--run-dir",
                str(tmp_path),
                "--device",
                "cuda:0",
                "--torch-threads",
                "6",
                "--self-play-workers",
                "4",
                "--parallel-games",
                "6",
            ]
        )
        == 0
    )

    assert captured.config is not None
    assert captured.config.device == "cuda:0"
    assert captured.config.torch_threads == 6
    assert captured.config.self_play_workers == 4
    assert captured.config.parallel_games == 6
    assert captured.config.channels == 2
    assert captured.config.residual_blocks == 1
    assert captured.config.max_full_moves == 11


def test_train_on_existing_checkpoint_resumes_with_saved_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    saved = TrainingConfig(
        target_games=20_000,
        max_full_moves=17,
        device="cpu",
        torch_threads=3,
        self_play_workers=2,
        simulations_per_move=5,
        channels=2,
        residual_blocks=1,
        batch_size=7,
        run_dir=tmp_path,
    )
    _seed_checkpoint(tmp_path, saved)
    _seed_status(tmp_path, target=25_000)
    captured = _capture_trainer(monkeypatch)

    assert main(["train", "--run-dir", str(tmp_path)]) == 0

    assert captured.config == replace(saved, target_games=25_000)
    assert captured.resume is True


def test_train_on_checkpoint_can_raise_but_not_shrink_cumulative_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    saved = TrainingConfig(
        target_games=7,
        max_full_moves=11,
        channels=2,
        residual_blocks=1,
        run_dir=tmp_path,
    )
    _seed_checkpoint(tmp_path, saved)
    _seed_status(tmp_path, target=12)
    captured = _capture_trainer(monkeypatch)

    assert main(["train", "--run-dir", str(tmp_path), "--games", "15"]) == 0

    assert captured.config == replace(saved, target_games=15)
    assert captured.resume is True


def test_train_with_damaged_checkpoint_fails_instead_of_restarting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "checkpoint-a.pt").write_bytes(b"damaged")
    captured = _capture_trainer(monkeypatch)

    assert main(["train", "--run-dir", str(tmp_path)]) == 1

    assert captured.config is None
    assert "checkpoint" in capsys.readouterr().err


def test_train_refuses_a_nonempty_directory_without_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "unexpected.txt").write_text("keep", encoding="utf-8")
    captured = _capture_trainer(monkeypatch)

    assert main(["train", "--run-dir", str(tmp_path)]) == 1

    assert captured.config is None
    assert "非空" in capsys.readouterr().err


def test_pause_writes_a_real_persistent_request(tmp_path: Path) -> None:
    _seed_status(tmp_path)

    assert main(["pause", "--run-dir", str(tmp_path)]) == 0

    assert RunControl(tmp_path).pause_requested()


def test_extend_updates_real_control_file_and_prints_stable_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_status(tmp_path)

    assert main(["extend", "--run-dir", str(tmp_path), "--games", "5"]) == 0

    assert RunControl(tmp_path).read_status().target_games == 15
    assert capsys.readouterr().out == (
        '{"completed_games": 3, "device": "cpu", "message": "", '
        '"phase": "running", "target_games": 15, "training_steps": 2}\n'
    )


def test_status_reads_real_control_file_as_stable_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_status(tmp_path)

    assert main(["status", "--run-dir", str(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert output == (
        '{"completed_games": 3, "device": "cpu", "message": "", '
        '"phase": "running", "target_games": 10, "training_steps": 2}\n'
    )
    assert json.loads(output)["completed_games"] == 3


def test_runtime_errors_go_to_stderr_and_return_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(["status", "--run-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert result != 0
    assert captured.out == ""
    assert "错误" in captured.err
    assert "状态文件不存在" in captured.err


def test_invalid_arguments_return_two_instead_of_raising_system_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(["train", "--games", "0"])

    assert result == 2
    assert "error" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_learning_rate_rejects_non_finite_values(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert not math.isfinite(float(value))

    assert main(["train", "--learning-rate", value]) == 2

    assert "error" in capsys.readouterr().err


def test_main_help_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    assert "1024 ply" in capsys.readouterr().out


def test_resume_without_checkpoint_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured = _capture_trainer(monkeypatch)

    assert main(["resume", "--run-dir", str(tmp_path)]) != 0

    assert captured.config is None
    assert "checkpoint" in capsys.readouterr().err


def test_help_explains_full_move_limit_and_all_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "512" in help_text
    assert "1024 ply" in help_text
    for command in ("train", "pause", "resume", "extend", "status"):
        assert command in help_text
