from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import torch

from ai.checkpoint import CheckpointManager
from ai.config import TrainingConfig
from ai.control import RunControl, RunStatus
from ai.trainer import Trainer


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的整数")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的有限数")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("必须是非负整数")
    return parsed


def _add_run_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("AI-runs/default"),
        help="训练运行目录（默认：AI-runs/default）",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai",
        description=(
            "中国象棋 AlphaZero 独立训练工具；默认训练 10000 局，每局最多 "
            "512 个完整回合（红黑各走一步），即 1024 ply。"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="新建训练")
    _add_run_dir(train)
    train.add_argument("--games", type=_positive_int, default=10_000)
    train.add_argument(
        "--full-moves",
        type=_positive_int,
        default=512,
        help="单局完整回合上限（默认 512，即 1024 ply）",
    )
    train.add_argument("--device", default="auto", help="auto、cpu、cuda 或 cuda:N")
    train.add_argument("--torch-threads", type=_positive_int, default=1)
    train.add_argument(
        "--self-play-workers",
        type=_positive_int,
        default=1,
        help="CPU 生产进程数（CPU/CUDA 模式都生效，默认 1）",
    )
    train.add_argument(
        "--parallel-games",
        type=_positive_int,
        default=16,
        help="CUDA 最多在途棋局数（默认 16）",
    )
    train.add_argument("--simulations", type=_positive_int, default=64)
    train.add_argument("--residual-blocks", type=_positive_int, default=4)
    train.add_argument("--channels", type=_positive_int, default=64)
    train.add_argument("--batch-size", type=_positive_int, default=128)
    train.add_argument("--replay-capacity-games", type=_positive_int, default=2_000)
    train.add_argument("--learning-rate", type=_positive_float, default=1e-3)
    train.add_argument("--checkpoint-interval-games", type=_positive_int, default=10)
    train.add_argument("--game-retry-limit", type=_nonnegative_int, default=2)
    train.add_argument("--seed", type=int, default=0)

    pause = commands.add_parser("pause", help="请求在安全点暂停训练")
    _add_run_dir(pause)

    resume = commands.add_parser("resume", help="从持久化 checkpoint 断点续训")
    _add_run_dir(resume)
    resume.add_argument("--device", default=None, help="覆盖保存的训练设备")
    resume.add_argument(
        "--torch-threads", type=_positive_int, default=None, help="覆盖 CPU 线程数"
    )
    resume.add_argument(
        "--self-play-workers",
        type=_positive_int,
        default=None,
        help="覆盖 CPU 生产进程数（CPU/CUDA 模式都生效）",
    )
    resume.add_argument(
        "--parallel-games",
        type=_positive_int,
        default=None,
        help="覆盖 CUDA 最多在途棋局数",
    )

    extend = commands.add_parser("extend", help="累计追加目标训练局数")
    _add_run_dir(extend)
    extend.add_argument("--games", type=_positive_int, required=True)

    status = commands.add_parser("status", help="输出稳定的 JSON 训练状态")
    _add_run_dir(status)
    return parser


def _load_checkpoint_config(run_dir: Path) -> TrainingConfig:
    """读取可恢复槽位中的配置，不预先构造可能不兼容的网络。"""
    manager = CheckpointManager(run_dir)
    errors: list[str] = []
    # CheckpointManager 在完整恢复时也按此顺序回退到另一槽；这里先取得架构。
    for slot in manager._candidate_slots():
        try:
            payload = torch.load(
                manager._slots[slot], map_location="cpu", weights_only=True
            )
            loaded = manager._validate_payload(payload)
            return replace(loaded.config, run_dir=run_dir)
        except Exception as error:  # noqa: BLE001 - 损坏槽位可能抛出多种异常
            errors.append(f"{slot}: {error}")
    detail = "; ".join(errors) if errors else "未找到任何 checkpoint 槽位"
    raise RuntimeError(f"没有可读取配置的 checkpoint；{detail}")


def _print_status(status: RunStatus) -> None:
    print(json.dumps(asdict(status), ensure_ascii=False, sort_keys=True))


def _train(args: argparse.Namespace) -> None:
    run_dir: Path = args.run_dir
    existing_entries = tuple(run_dir.iterdir()) if run_dir.exists() else ()
    manager = CheckpointManager(run_dir)
    if manager.has_checkpoint():
        saved = _load_checkpoint_config(run_dir)
        targets = [saved.target_games, args.games]
        control = RunControl(run_dir)
        if control.status_path.exists():
            targets.append(control.read_status().target_games)
        config = replace(saved, target_games=max(targets), run_dir=run_dir)
        Trainer(config).run(resume=True)
        return
    if existing_entries:
        raise RuntimeError(f"训练目录非空但没有可恢复的 checkpoint：{run_dir}")
    config = TrainingConfig(
        target_games=args.games,
        max_full_moves=args.full_moves,
        device=args.device,
        torch_threads=args.torch_threads,
        self_play_workers=args.self_play_workers,
        parallel_games=args.parallel_games,
        simulations_per_move=args.simulations,
        residual_blocks=args.residual_blocks,
        channels=args.channels,
        batch_size=args.batch_size,
        replay_capacity_games=args.replay_capacity_games,
        learning_rate=args.learning_rate,
        checkpoint_interval_games=args.checkpoint_interval_games,
        game_retry_limit=args.game_retry_limit,
        seed=args.seed,
        run_dir=args.run_dir,
    )
    Trainer(config).run()


def _resume(args: argparse.Namespace) -> None:
    config = _load_checkpoint_config(args.run_dir)
    overrides = {
        name: value
        for name, value in {
            "device": args.device,
            "torch_threads": args.torch_threads,
            "self_play_workers": args.self_play_workers,
            "parallel_games": args.parallel_games,
        }.items()
        if value is not None
    }
    Trainer(replace(config, **overrides)).run(resume=True)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as error:
        return int(error.code) if isinstance(error.code, int) else 1
    try:
        if args.command == "train":
            _train(args)
        elif args.command == "resume":
            _resume(args)
        elif args.command == "pause":
            RunControl(args.run_dir).request_pause()
        elif args.command == "extend":
            _print_status(RunControl(args.run_dir).extend(args.games))
        elif args.command == "status":
            _print_status(RunControl(args.run_dir).read_status())
        else:  # pragma: no cover - argparse 的 required 子命令保证不可达
            raise AssertionError(f"未知命令：{args.command}")
    except Exception as error:  # noqa: BLE001 - CLI 边界统一转换为稳定退出码
        print(f"错误：{error}", file=sys.stderr)
        return 1
    return 0


__all__ = ["build_parser", "main"]
