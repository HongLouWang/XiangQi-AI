"""只读统计 AlphaZero Replay 中训练棋局的胜负结果。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    import numpy as np
except ModuleNotFoundError:
    print("统计失败：缺少 NumPy，请使用项目 .venv/bin/python 运行", file=sys.stderr)
    raise SystemExit(2) from None


AI_ROOT = Path(__file__).resolve().parent
MAX_SNAPSHOT_ATTEMPTS = 3


class TrainingDataError(RuntimeError):
    """训练目录的结构或元数据无法安全统计。"""


@dataclass(frozen=True, slots=True)
class GameError:
    game_id: int
    reason: str


@dataclass(frozen=True, slots=True)
class TrainingResultReport:
    run_dir: Path
    phase: str
    completed_games: int | None
    target_games: int | None
    training_steps: int | None
    total_games: int
    retained_games: int
    red_wins: int
    black_wins: int
    draws: int
    errors: tuple[GameError, ...]

    @property
    def invalid_games(self) -> int:
        return len(self.errors)

    @property
    def classified_games(self) -> int:
        return self.red_wins + self.black_wins + self.draws


def _read_json_object(path: Path, *, required: bool) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise TrainingDataError(f"缺少文件：{path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise TrainingDataError(f"文件不是有效 UTF-8：{path}") from error
    except json.JSONDecodeError as error:
        raise TrainingDataError(f"文件不是有效 JSON：{path}（{error.msg}）") from error
    except OSError as error:
        raise TrainingDataError(f"无法读取文件：{path}（{error}）") from error
    if not isinstance(payload, dict):
        raise TrainingDataError(f"JSON 顶层必须是对象：{path}")
    return payload


def _non_negative_integer(
    payload: dict[str, Any], name: str, *, path: Path
) -> int:
    value = payload.get(name)
    if type(value) is not int or value < 0:
        raise TrainingDataError(f"{path} 中 {name} 必须是非负整数")
    return value


def _manifest(run_dir: Path) -> tuple[tuple[int, ...], int, bytes]:
    path = run_dir / "replay" / "manifest.json"
    try:
        raw = path.read_bytes()
    except FileNotFoundError as error:
        raise TrainingDataError(f"缺少文件：{path}") from error
    except OSError as error:
        raise TrainingDataError(f"无法读取文件：{path}（{error}）") from error
    try:
        payload = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise TrainingDataError(f"文件不是有效 UTF-8：{path}") from error
    except json.JSONDecodeError as error:
        raise TrainingDataError(f"文件不是有效 JSON：{path}（{error.msg}）") from error
    if not isinstance(payload, dict):
        raise TrainingDataError(f"JSON 顶层必须是对象：{path}")
    raw_games = payload.get("games")
    if not isinstance(raw_games, list) or any(
        type(game_id) is not int or game_id <= 0 for game_id in raw_games
    ):
        raise TrainingDataError(f"{path} 中 games 必须是正整数棋局 ID 列表")
    games = tuple(raw_games)
    if len(set(games)) != len(games):
        raise TrainingDataError(f"{path} 中 games 包含重复棋局 ID")
    total_games = _non_negative_integer(payload, "total_games", path=path)
    if total_games < len(games):
        raise TrainingDataError(
            f"{path} 中 total_games 不能小于当前保留棋局数"
        )
    return games, total_games, raw


def _status(run_dir: Path) -> tuple[str, int | None, int | None, int | None]:
    path = run_dir / "status.json"
    payload = _read_json_object(path, required=False)
    if payload is None:
        return "unknown", None, None, None
    phase = payload.get("phase")
    if not isinstance(phase, str) or not phase.strip():
        raise TrainingDataError(f"{path} 中 phase 必须是非空字符串")
    values = tuple(
        _non_negative_integer(payload, name, path=path)
        for name in ("completed_games", "target_games", "training_steps")
    )
    completed, target, steps = values
    if target < completed:
        raise TrainingDataError(f"{path} 中 target_games 不能小于 completed_games")
    return phase, completed, target, steps


def _classify_game(path: Path) -> str:
    try:
        with np.load(path, allow_pickle=False) as stored:
            values = np.asarray(stored["values"])
            sides = np.asarray(stored["sides"])
            plies = np.asarray(stored["plies"])
    except Exception as error:
        raise ValueError(f"无法读取 values/sides：{type(error).__name__}: {error}") from error
    if values.ndim != 1 or sides.ndim != 1 or values.size == 0:
        raise ValueError("values 和 sides 必须是等长非空一维数组")
    if values.shape != sides.shape:
        raise ValueError("values 和 sides 长度不一致")
    if (
        plies.shape != (1,)
        or not np.issubdtype(plies.dtype, np.integer)
        or int(plies[0]) < 0
        or int(plies[0]) != values.size
    ):
        raise ValueError("plies 必须是与样本数相等的单元素非负整数数组")
    if not np.all(np.isfinite(values)):
        raise ValueError("values 包含非有限数")
    if not np.all(np.isin(sides, (0, 1))):
        raise ValueError("sides 只能包含 0（红）或 1（黑）")
    red_values = np.where(sides == 0, values, -values)
    if np.allclose(red_values, 1.0):
        return "red_win"
    if np.allclose(red_values, -1.0):
        return "black_win"
    if np.allclose(red_values, 0.0):
        return "draw"
    raise ValueError("values 与 sides 无法得到一致的胜负结果")


def count_results(run_dir: Path | str) -> TrainingResultReport:
    path = Path(run_dir).expanduser().resolve()
    if not path.is_dir():
        raise TrainingDataError(f"训练目录不存在：{path}")
    manifest_path = path / "replay" / "manifest.json"
    for _attempt in range(MAX_SNAPSHOT_ATTEMPTS):
        game_ids, total_games, manifest_before = _manifest(path)
        phase, completed, target, steps = _status(path)
        counts = {"red_win": 0, "black_win": 0, "draw": 0}
        errors: list[GameError] = []
        for game_id in game_ids:
            game_path = path / "replay" / "games" / f"{game_id:012d}.npz"
            try:
                outcome = _classify_game(game_path)
            except (OSError, ValueError) as error:
                errors.append(GameError(game_id, " ".join(str(error).splitlines())))
                continue
            counts[outcome] += 1
        try:
            manifest_after = manifest_path.read_bytes()
        except OSError:
            continue
        if manifest_after != manifest_before:
            continue
        status_after = _status(path)
        if status_after != (phase, completed, target, steps):
            continue
        if completed is not None and completed != total_games:
            continue
        return TrainingResultReport(
            run_dir=path,
            phase=phase,
            completed_games=completed,
            target_games=target,
            training_steps=steps,
            total_games=total_games,
            retained_games=len(game_ids),
            red_wins=counts["red_win"],
            black_wins=counts["black_win"],
            draws=counts["draw"],
            errors=tuple(errors),
        )
    raise TrainingDataError(
        f"训练数据持续更新，{MAX_SNAPSHOT_ATTEMPTS} 次尝试仍无法取得稳定快照"
    )


def format_report(report: TrainingResultReport, observed_at: datetime) -> str:
    progress = (
        "未知"
        if report.completed_games is None or report.target_games is None
        else f"{report.completed_games}/{report.target_games}"
    )
    steps = "未知" if report.training_steps is None else str(report.training_steps)
    lines = [
        f"统计时间：{observed_at.isoformat(timespec='seconds')}",
        f"训练目录：{report.run_dir}",
        f"训练阶段：{report.phase}",
        f"训练进度：{progress}，训练步数：{steps}",
        f"历史累计：{report.total_games} 局",
        f"当前保留：{report.retained_games} 局",
        f"已分类：{report.classified_games} 局",
        f"红方胜：{report.red_wins} 局",
        f"黑方胜：{report.black_wins} 局",
        f"和棋：{report.draws} 局",
        f"异常：{report.invalid_games} 局",
    ]
    lines.extend(f"  - 棋局 {error.game_id}：{error.reason}" for error in report.errors)
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_run_dir = AI_ROOT / "AI-runs/cpu-main"
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=default_run_dir,
        help=f"训练目录（默认：{default_run_dir}）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = count_results(args.run_dir)
    except TrainingDataError as error:
        print(f"统计失败：{error}", file=sys.stderr)
        return 2
    observed_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    print(format_report(report, observed_at))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
