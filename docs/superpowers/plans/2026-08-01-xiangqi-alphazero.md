# 中国象棋 AlphaZero 训练系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `src/ai/` 中实现可配置的中国象棋 AlphaZero 自我对弈训练系统，默认累计训练 10,000 局、每局最多 512 个完整回合（1024 ply），支持多线程 CPU、CUDA、安全暂停、追加和断点续传。

**Architecture:** 复用 `xiangqi` 的不可变棋盘与合法走法，AI 包负责当前方视角编码、Policy/Value ResNet、PUCT MCTS、自我对弈、持久化 Replay Buffer 和训练协调。训练运行状态以原子 checkpoint 和 JSON 控制文件持久化；CPU 可并发产生棋局，CUDA 模式由主进程持有模型并执行推理和优化。

**Tech Stack:** Python 3.11、PyTorch 2.3+、NumPy 1.26+、pytest、现有 `xiangqi` 规则引擎。

---

## 文件映射

- `pyproject.toml`：增加 `ai` 可选依赖和 AI 测试依赖。
- `.gitignore`：排除 `AI-runs/` 训练产物。
- `src/ai/config.py`：不可变配置与校验。
- `src/ai/encoding.py`：15 平面局面编码、8100 动作空间与合法掩码。
- `src/ai/network.py`：ResNet、设备解析和 CPU 线程设置。
- `src/ai/mcts.py`：状态、PUCT 节点、搜索和策略生成。
- `src/ai/self_play.py`：完整单盘、自我对弈结果与 1024 ply 截止。
- `src/ai/replay.py`：按棋局原子写入的磁盘 Replay Buffer。
- `src/ai/checkpoint.py`：双槽原子 checkpoint、随机状态捕获与恢复。
- `src/ai/control.py`：状态、暂停请求、恢复与追加目标。
- `src/ai/trainer.py`：训练循环、CPU worker 调度、CUDA/CPU 优化。
- `src/ai/cli.py`、`src/ai/__main__.py`：命令行接口。
- `tests/ai/`：与上述模块一一对应的测试。
- `README.md`：安装、训练、暂停、续训和 CUDA 示例。

### Task 1: AI 依赖和强类型配置

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `src/ai/__init__.py`
- Create: `src/ai/config.py`
- Create: `tests/ai/test_config.py`

- [ ] **Step 1: 写配置失败测试**

```python
# tests/ai/test_config.py
import pytest

from ai.config import TrainingConfig


def test_defaults_are_10000_games_and_512_full_moves() -> None:
    config = TrainingConfig()
    assert config.target_games == 10_000
    assert config.max_full_moves == 512
    assert config.max_plies == 1024


def test_training_scale_can_be_changed_from_python() -> None:
    config = TrainingConfig(target_games=25, max_full_moves=7)
    assert config.target_games == 25
    assert config.max_plies == 14


@pytest.mark.parametrize("field,value", [("target_games", 0), ("max_full_moves", 0), ("self_play_workers", 0), ("torch_threads", 0)])
def test_positive_configuration_is_required(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**{field: value})
```

- [ ] **Step 2: 运行测试并确认因 `ai.config` 不存在而失败**

Run: `.venv/bin/python -m pytest tests/ai/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ai.config'`.

- [ ] **Step 3: 实现配置和依赖声明**

```python
# src/ai/config.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    target_games: int = 10_000
    max_full_moves: int = 512
    device: str = "auto"
    torch_threads: int = 1
    self_play_workers: int = 1
    simulations_per_move: int = 64
    residual_blocks: int = 4
    channels: int = 64
    batch_size: int = 128
    replay_capacity_games: int = 2_000
    learning_rate: float = 1e-3
    checkpoint_interval_games: int = 10
    seed: int = 0
    run_dir: Path = Path("AI-runs/default")

    def __post_init__(self) -> None:
        positive = {
            "target_games": self.target_games,
            "max_full_moves": self.max_full_moves,
            "torch_threads": self.torch_threads,
            "self_play_workers": self.self_play_workers,
            "simulations_per_move": self.simulations_per_move,
            "batch_size": self.batch_size,
            "replay_capacity_games": self.replay_capacity_games,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0")

    @property
    def max_plies(self) -> int:
        return self.max_full_moves * 2
```

在 `pyproject.toml` 增加：

```toml
ai = ["torch>=2.3", "numpy>=1.26"]
```

在 `.gitignore` 增加：

```gitignore
AI-runs/
```

- [ ] **Step 4: 运行测试并确认通过**

Run: `.venv/bin/python -m pytest tests/ai/test_config.py -v`
Expected: 4 parameter cases plus 2 behavior tests PASS.

- [ ] **Step 5: 中文提交**

```bash
git add pyproject.toml .gitignore src/ai/__init__.py src/ai/config.py tests/ai/test_config.py
git commit -m "功能：增加 AI 训练配置"
```

### Task 2: 局面与动作编码

**Files:**
- Create: `src/ai/encoding.py`
- Create: `tests/ai/test_encoding.py`

- [ ] **Step 1: 写编码失败测试**

```python
# tests/ai/test_encoding.py
import numpy as np

from ai.encoding import ACTION_SIZE, INPUT_CHANNELS, decode_action, encode_action, encode_board, legal_policy
from xiangqi.board import Board
from xiangqi.domain import Color
from xiangqi.rules import all_legal_moves


def test_action_round_trip_for_every_standard_legal_move() -> None:
    board = Board.standard()
    for move in all_legal_moves(board, Color.RED):
        assert decode_action(encode_action(move), board) == move


def test_black_view_is_rotated_to_current_player_perspective() -> None:
    red = encode_board(Board.standard(), Color.RED)
    black = encode_board(Board.standard(), Color.BLACK)
    assert red.shape == (INPUT_CHANNELS, 10, 9)
    assert np.array_equal(red[:14], black[:14, ::-1, ::-1])


def test_legal_policy_masks_all_illegal_logits() -> None:
    board = Board.standard()
    moves = all_legal_moves(board, Color.RED)
    probabilities = legal_policy(np.zeros(ACTION_SIZE, dtype=np.float32), moves)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.count_nonzero(probabilities) == len(moves)
```

- [ ] **Step 2: 运行测试并确认缺少编码 API**

Run: `.venv/bin/python -m pytest tests/ai/test_encoding.py -v`
Expected: FAIL importing `ai.encoding`.

- [ ] **Step 3: 实现固定编码 API**

```python
# src/ai/encoding.py
from __future__ import annotations

import numpy as np

from xiangqi.board import Board
from xiangqi.domain import Color, Coord, Move, PieceType

ACTION_SIZE = 90 * 90
INPUT_CHANNELS = 15
_KINDS = tuple(PieceType)


def _view(coord: Coord, side: Color) -> Coord:
    return coord if side is Color.RED else Coord(8 - coord.file, 9 - coord.rank)


def encode_board(board: Board, side: Color) -> np.ndarray:
    result = np.zeros((INPUT_CHANNELS, 10, 9), dtype=np.float32)
    for coord, piece in board.pieces.items():
        viewed = _view(coord, side)
        owner_offset = 0 if piece.color is side else 7
        result[owner_offset + _KINDS.index(piece.kind), viewed.rank, viewed.file] = 1.0
    result[14, :, :] = 1.0 if side is Color.RED else 0.0
    return result


def encode_action(move: Move) -> int:
    start = move.start.rank * 9 + move.start.file
    end = move.end.rank * 9 + move.end.file
    return start * 90 + end


def decode_action(index: int, board: Board) -> Move:
    start_index, end_index = divmod(index, 90)
    start = Coord(start_index % 9, start_index // 9)
    end = Coord(end_index % 9, end_index // 9)
    piece = board.at(start)
    if piece is None:
        raise ValueError("动作起点没有棋子")
    return Move(start, end, piece, board.at(end))


def legal_policy(logits: np.ndarray, moves: tuple[Move, ...]) -> np.ndarray:
    if not moves:
        raise ValueError("无合法走法时不能生成策略")
    indices = np.asarray([encode_action(move) for move in moves])
    selected = logits[indices].astype(np.float64)
    selected = np.exp(selected - selected.max())
    result = np.zeros(ACTION_SIZE, dtype=np.float32)
    result[indices] = selected / selected.sum()
    return result
```

- [ ] **Step 4: 修正黑方动作视角**

将 `encode_action`/`decode_action` 增加 `side: Color` 参数，并在黑方使用 `_view`；更新测试，对红黑双方全部合法着法验证 round trip。此修改必须在 MCTS 前完成，保证网络动作始终处于当前方统一视角。

- [ ] **Step 5: 运行编码与原有规则测试**

Run: `.venv/bin/python -m pytest tests/ai/test_encoding.py tests/test_legality_and_endings.py -v`
Expected: PASS.

- [ ] **Step 6: 中文提交**

```bash
git add src/ai/encoding.py tests/ai/test_encoding.py
git commit -m "功能：实现 AI 局面与动作编码"
```

### Task 3: Policy/Value ResNet 与设备配置

**Files:**
- Create: `src/ai/network.py`
- Create: `tests/ai/test_network.py`

- [ ] **Step 1: 写网络和设备失败测试**

```python
# tests/ai/test_network.py
import pytest
import torch

from ai.encoding import ACTION_SIZE, INPUT_CHANNELS
from ai.network import PolicyValueNetwork, configure_device


def test_network_outputs_policy_and_bounded_value() -> None:
    model = PolicyValueNetwork(channels=16, residual_blocks=1)
    policy, value = model(torch.zeros(2, INPUT_CHANNELS, 10, 9))
    assert policy.shape == (2, ACTION_SIZE)
    assert value.shape == (2, 1)
    assert torch.all(value.abs() <= 1)


def test_cpu_device_applies_requested_thread_count() -> None:
    device = configure_device("cpu", torch_threads=2)
    assert device.type == "cpu"
    assert torch.get_num_threads() == 2


def test_explicit_cuda_fails_instead_of_falling_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        configure_device("cuda", torch_threads=1)
```

- [ ] **Step 2: 运行并观察 `ai.network` 导入失败**

Run: `.venv/bin/python -m pytest tests/ai/test_network.py -v`
Expected: FAIL importing `ai.network`.

- [ ] **Step 3: 实现 ResNet 和设备解析**

```python
# src/ai/network.py
import torch
from torch import nn

from ai.encoding import ACTION_SIZE, INPUT_CHANNELS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.relu(inputs + self.body(inputs))


class PolicyValueNetwork(nn.Module):
    def __init__(self, channels: int = 64, residual_blocks: int = 4) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
            *(ResidualBlock(channels) for _ in range(residual_blocks)),
        )
        self.policy = nn.Sequential(nn.Conv2d(channels, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(180, ACTION_SIZE))
        self.value = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.ReLU(), nn.Flatten(), nn.Linear(90, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trunk = self.trunk(inputs)
        return self.policy(trunk), self.value(trunk)


def configure_device(name: str, torch_threads: int) -> torch.device:
    torch.set_num_threads(torch_threads)
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"已请求 {name}，但 CUDA 不可用")
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"不支持的设备: {name}")
    return device
```

- [ ] **Step 4: 运行 CPU 测试及条件 CUDA 烟雾测试**

Run: `.venv/bin/python -m pytest tests/ai/test_network.py -v`
Expected: PASS; add a `@pytest.mark.skipif(not torch.cuda.is_available(), ...)` test that performs one CUDA forward pass when available.

- [ ] **Step 5: 中文提交**

```bash
git add src/ai/network.py tests/ai/test_network.py
git commit -m "功能：实现策略价值网络与设备选择"
```

### Task 4: PUCT MCTS

**Files:**
- Create: `src/ai/mcts.py`
- Create: `tests/ai/test_mcts.py`

- [ ] **Step 1: 写搜索失败测试**

```python
# tests/ai/test_mcts.py
import numpy as np

from ai.mcts import MCTS, SearchState
from xiangqi.board import Board
from xiangqi.domain import Color
from xiangqi.rules import all_legal_moves


class UniformEvaluator:
    def evaluate(self, state: SearchState) -> tuple[np.ndarray, float]:
        return np.zeros(8100, dtype=np.float32), 0.25


def test_search_visits_only_legal_root_moves() -> None:
    state = SearchState(Board.standard(), Color.RED)
    policy = MCTS(UniformEvaluator(), simulations=8, c_puct=1.5, seed=3).search(state, add_noise=False)
    legal = {move for move in all_legal_moves(state.board, state.side)}
    assert set(policy) == legal
    assert sum(policy.values()) == 1.0


def test_child_value_is_negated_for_parent_perspective() -> None:
    state = SearchState(Board.standard(), Color.RED)
    search = MCTS(UniformEvaluator(), simulations=2, c_puct=1.5, seed=3)
    search.search(state, add_noise=False)
    assert any(child.value_sum < 0 for child in search.root.children.values())
```

- [ ] **Step 2: 运行并确认缺少搜索实现**

Run: `.venv/bin/python -m pytest tests/ai/test_mcts.py -v`
Expected: FAIL importing `ai.mcts`.

- [ ] **Step 3: 实现状态推进、节点和 PUCT**

```python
# src/ai/mcts.py (public contract)
@dataclass(frozen=True, slots=True)
class SearchState:
    board: Board
    side: Color

    def play(self, move: Move) -> "SearchState":
        return SearchState(self.board.move_unchecked(move.start, move.end), self.side.opponent)


class Evaluator(Protocol):
    def evaluate(self, state: SearchState) -> tuple[np.ndarray, float]: ...


@dataclass(slots=True)
class Node:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[Move, "Node"] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count
```

`MCTS.search` 必须执行指定模拟次数、用 `Q + c_puct * P * sqrt(parent_n)/(1+child_n)` 选择、终局直接返回当前方视角的 `-1`、逐层取负反传，并把根访问次数归一化为合法动作概率。根噪声使用配置的 `dirichlet_alpha` 和 `dirichlet_fraction`，仅在 `add_noise=True` 时混入先验。

- [ ] **Step 4: 增加真实一步推进和终局测试并运行**

Run: `.venv/bin/python -m pytest tests/ai/test_mcts.py tests/test_legality_and_endings.py -v`
Expected: PASS, including a constructed checkmate state whose终局价值不调用 evaluator.

- [ ] **Step 5: 中文提交**

```bash
git add src/ai/mcts.py tests/ai/test_mcts.py
git commit -m "功能：实现 PUCT 蒙特卡洛树搜索"
```

### Task 5: 自我对弈与 1024 ply 上限

**Files:**
- Create: `src/ai/self_play.py`
- Create: `tests/ai/test_self_play.py`

- [ ] **Step 1: 写上限与价值回填失败测试**

```python
# tests/ai/test_self_play.py
from ai.self_play import play_game
from xiangqi.domain import Color


def test_512_full_moves_means_exactly_1024_plies() -> None:
    result = play_game(search=AlwaysFirstLegalSearch(), max_plies=1024, initial_state=LoopingState())
    assert result.plies == 1024
    assert result.winner is None
    assert all(sample.value == 0.0 for sample in result.samples)


def test_winner_is_converted_to_each_sample_side() -> None:
    result = play_game(search=ScriptedWinSearch(), max_plies=1024)
    assert result.winner is Color.RED
    assert result.samples[0].value == 1.0
    assert result.samples[1].value == -1.0
```

- [ ] **Step 2: 运行并确认缺少 `play_game`**

Run: `.venv/bin/python -m pytest tests/ai/test_self_play.py -v`
Expected: FAIL importing `ai.self_play`.

- [ ] **Step 3: 实现单盘数据结构和循环**

```python
# src/ai/self_play.py (public contract)
@dataclass(frozen=True, slots=True)
class TrainingSample:
    state: np.ndarray
    policy_indices: np.ndarray
    policy_probabilities: np.ndarray
    side: Color
    value: float


@dataclass(frozen=True, slots=True)
class GameResult:
    samples: tuple[TrainingSample, ...]
    winner: Color | None
    plies: int
    termination: str


def play_game(search: Search, max_plies: int = 1024, *, seed: int = 0) -> GameResult:
    state = SearchState(Board.standard(), Color.RED)
    pending: list[tuple[np.ndarray, np.ndarray, np.ndarray, Color]] = []
    for ply in range(max_plies):
        position = evaluate_position(state.board, state.side)
        if position.kind is not PositionKind.ONGOING:
            return _finish(pending, position.winner, ply, position.kind.value)
        policy = search.search(state, add_noise=True)
        move = _sample(policy, temperature=1.0 if ply < 30 else 0.0, seed=seed + ply)
        pending.append(_encode_sample(state, policy))
        state = state.play(move)
    return _finish(pending, None, max_plies, "move_limit")
```

为测试引入小型 `GameAdapter` protocol，使循环局面可不依赖构造 1024 步真实棋局；生产默认 adapter 仍调用 `Board.standard`、`evaluate_position` 与 `SearchState.play`。

- [ ] **Step 4: 运行自我对弈测试**

Run: `.venv/bin/python -m pytest tests/ai/test_self_play.py -v`
Expected: PASS and proves no 1025th ply is requested.

- [ ] **Step 5: 中文提交**

```bash
git add src/ai/self_play.py tests/ai/test_self_play.py
git commit -m "功能：实现自我对弈与回合上限"
```

### Task 6: Replay Buffer 与原子 checkpoint

**Files:**
- Create: `src/ai/replay.py`
- Create: `src/ai/checkpoint.py`
- Create: `tests/ai/test_replay.py`
- Create: `tests/ai/test_checkpoint.py`

- [ ] **Step 1: 写持久化失败测试**

```python
# tests/ai/test_replay.py
def test_replay_commits_complete_games_and_evicts_oldest(tmp_path: Path) -> None:
    replay = ReplayBuffer(tmp_path, capacity_games=2)
    replay.append_game(game(1))
    replay.append_game(game(2))
    replay.append_game(game(3))
    reopened = ReplayBuffer(tmp_path, capacity_games=2)
    assert reopened.game_ids == (2, 3)


# tests/ai/test_checkpoint.py
def test_checkpoint_restores_model_optimizer_progress_and_rng(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    manager.save(model, optimizer, TrainingProgress(completed_games=7, target_games=10, training_steps=3), config)
    restored = manager.load_latest(model, optimizer)
    assert restored.progress.completed_games == 7
    assert restored.progress.target_games == 10


def test_corrupt_latest_checkpoint_falls_back_to_previous_slot(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    manager.save(model, optimizer, progress(1), config)
    manager.save(model, optimizer, progress(2), config)
    manager.latest_path.write_bytes(b"broken")
    assert manager.load_latest(model, optimizer).progress.completed_games == 1
```

- [ ] **Step 2: 运行并确认模块不存在**

Run: `.venv/bin/python -m pytest tests/ai/test_replay.py tests/ai/test_checkpoint.py -v`
Expected: FAIL importing persistence modules.

- [ ] **Step 3: 实现 Replay Buffer**

每局写 `games/<monotonic-id>.npz.tmp`，`flush`/`fsync` 后 `os.replace` 为 `.npz`；清单 `manifest.json` 同样原子替换。清单结构固定为：

```json
{"schema_version": 1, "encoding_version": 1, "action_version": 1, "next_game_id": 4, "games": [2, 3]}
```

`ReplayBuffer.sample(batch_size, rng)` 只返回训练所需 dense state、稀疏 policy 和 value，并在训练 collate 时展开 policy target，避免磁盘保存 8100 维零向量。

- [ ] **Step 4: 实现双槽 checkpoint**

`CheckpointManager.save` 在 `checkpoint-a.pt` 与 `checkpoint-b.pt` 间轮换，payload 包含 schema、配置、模型、优化器、进度、Replay 清单 hash，以及 `random.getstate()`、NumPy generator state、`torch.get_rng_state()`、可用时的 CUDA RNG。写入成功后原子更新 `latest.json`。`load_latest` 先读 latest 槽，损坏时验证并回退另一槽；架构、输入通道、动作版本不匹配时抛 `CheckpointCompatibilityError`。

- [ ] **Step 5: 运行持久化测试**

Run: `.venv/bin/python -m pytest tests/ai/test_replay.py tests/ai/test_checkpoint.py -v`
Expected: PASS, and `git status --short` contains no files below `AI-runs/`.

- [ ] **Step 6: 中文提交**

```bash
git add src/ai/replay.py src/ai/checkpoint.py tests/ai/test_replay.py tests/ai/test_checkpoint.py
git commit -m "功能：持久化训练数据与断点"
```

### Task 7: 控制状态、暂停、恢复和追加

**Files:**
- Create: `src/ai/control.py`
- Create: `tests/ai/test_control.py`

- [ ] **Step 1: 写控制失败测试**

```python
# tests/ai/test_control.py
def test_pause_request_is_observed_only_at_safe_point(tmp_path: Path) -> None:
    control = RunControl(tmp_path)
    control.request_pause()
    assert control.pause_requested()
    control.mark_paused(progress(completed_games=4, target_games=10))
    assert control.read_status().phase == "paused"


def test_extend_adds_to_cumulative_target_without_resetting_progress(tmp_path: Path) -> None:
    control = RunControl(tmp_path)
    control.write_status(RunStatus("running", completed_games=4, target_games=10, training_steps=2, device="cpu"))
    updated = control.extend(5)
    assert updated.completed_games == 4
    assert updated.target_games == 15
```

- [ ] **Step 2: 运行并确认缺少控制模块**

Run: `.venv/bin/python -m pytest tests/ai/test_control.py -v`
Expected: FAIL importing `ai.control`.

- [ ] **Step 3: 实现原子 JSON 控制文件**

```python
# src/ai/control.py (public contract)
@dataclass(frozen=True, slots=True)
class RunStatus:
    phase: Literal["new", "running", "pausing", "paused", "completed", "failed"]
    completed_games: int
    target_games: int
    training_steps: int
    device: str
    message: str = ""


class RunControl:
    def request_pause(self) -> None: ...
    def pause_requested(self) -> bool: ...
    def clear_pause(self) -> None: ...
    def read_status(self) -> RunStatus: ...
    def write_status(self, status: RunStatus) -> None: ...
    def extend(self, games: int) -> RunStatus: ...
```

所有状态更新在同目录用 `NamedTemporaryFile(delete=False)`、`flush`、`os.fsync`、`os.replace`；`extend` 要持有 `fcntl.flock`，拒绝非正数，避免训练器与命令同时覆盖目标数。

- [ ] **Step 4: 运行控制测试**

Run: `.venv/bin/python -m pytest tests/ai/test_control.py -v`
Expected: PASS, including两个并发 extend 后目标累计正确的测试。

- [ ] **Step 5: 中文提交**

```bash
git add src/ai/control.py tests/ai/test_control.py
git commit -m "功能：增加训练暂停恢复与追加控制"
```

### Task 8: 训练协调器、CPU 并发与续训

**Files:**
- Create: `src/ai/trainer.py`
- Create: `tests/ai/test_trainer.py`

- [ ] **Step 1: 写端到端协调失败测试**

```python
# tests/ai/test_trainer.py
def test_trainer_runs_configured_games_and_saves_checkpoint(tmp_path: Path) -> None:
    trainer = Trainer(tiny_config(tmp_path, target_games=2), game_factory=one_move_draw_game)
    trainer.run()
    status = RunControl(tmp_path).read_status()
    assert status.phase == "completed"
    assert status.completed_games == 2
    assert CheckpointManager(tmp_path).has_checkpoint()


def test_resume_continues_instead_of_restarting(tmp_path: Path) -> None:
    Trainer(tiny_config(tmp_path, target_games=1), game_factory=one_move_draw_game).run()
    RunControl(tmp_path).extend(2)
    resumed = Trainer(tiny_config(tmp_path, target_games=3), game_factory=one_move_draw_game)
    resumed.run(resume=True)
    assert RunControl(tmp_path).read_status().completed_games == 3


def test_cpu_worker_setting_is_used(tmp_path: Path) -> None:
    trainer = Trainer(tiny_config(tmp_path, target_games=4, self_play_workers=2), game_factory=recording_game)
    trainer.run()
    assert trainer.worker_count == 2
```

- [ ] **Step 2: 运行并确认训练器不存在**

Run: `.venv/bin/python -m pytest tests/ai/test_trainer.py -v`
Expected: FAIL importing `ai.trainer`.

- [ ] **Step 3: 实现训练批次和 loss**

```python
def train_batch(model, optimizer, states, policy_targets, value_targets, device):
    model.train()
    logits, values = model(states.to(device))
    policy_loss = -(policy_targets.to(device) * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
    value_loss = torch.nn.functional.mse_loss(values.squeeze(1), value_targets.to(device))
    loss = policy_loss + value_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(policy_loss.detach()), float(value_loss.detach())
```

- [ ] **Step 4: 实现协调循环和安全暂停点**

`Trainer.run` 顺序必须是：加载或初始化 → 配置设备/线程 → 写 running → 生成完整棋局 → 原子追加 Replay → 增加 completed_games → 可训练时更新模型 → checkpoint → 读取最新累计目标与 pause 请求。看到 pause 后先保存 checkpoint，再写 paused 并退出。正常完成时保存最终 checkpoint 和 completed。

CPU `self_play_workers > 1` 使用 `multiprocessing.get_context("spawn").Pool`；worker 接收只读 CPU `state_dict`、配置和唯一 seed，返回完整 `GameResult`，主进程是 Replay/模型/checkpoint 的唯一写入者。CUDA 模式主进程持有 GPU 模型；第一版使用主进程顺序 MCTS 推理，不复制 CUDA context 到 worker，并在状态里准确报告 `self_play_workers_effective=1`。CPU 多进程测试用 PID 集合证明实际使用多个进程，而不只检查配置值。

- [ ] **Step 5: 运行训练、暂停和恢复集成测试**

Run: `.venv/bin/python -m pytest tests/ai/test_trainer.py tests/ai/test_control.py tests/ai/test_checkpoint.py -v`
Expected: PASS, including暂停后 checkpoint 可加载、恢复后优化器步数连续、追加目标不重置完成局数。

- [ ] **Step 6: 中文提交**

```bash
git add src/ai/trainer.py tests/ai/test_trainer.py
git commit -m "功能：实现可续传的自我对弈训练循环"
```

### Task 9: 命令行和 Python 入口

**Files:**
- Create: `src/ai/cli.py`
- Create: `src/ai/__main__.py`
- Create: `tests/ai/test_cli.py`

- [ ] **Step 1: 写 CLI 失败测试**

```python
# tests/ai/test_cli.py
def test_train_defaults_and_overrides(monkeypatch, tmp_path: Path) -> None:
    captured = capture_trainer(monkeypatch)
    assert main(["train", "--run-dir", str(tmp_path), "--games", "12", "--full-moves", "5", "--device", "cpu", "--torch-threads", "3", "--self-play-workers", "2"]) == 0
    assert captured.config.target_games == 12
    assert captured.config.max_plies == 10
    assert captured.config.torch_threads == 3


def test_pause_extend_resume_and_status(tmp_path: Path, capsys) -> None:
    seed_status(tmp_path, completed=3, target=10)
    assert main(["extend", "--run-dir", str(tmp_path), "--games", "5"]) == 0
    assert main(["pause", "--run-dir", str(tmp_path)]) == 0
    assert main(["status", "--run-dir", str(tmp_path)]) == 0
    assert '"target_games": 15' in capsys.readouterr().out
```

- [ ] **Step 2: 运行并确认 CLI 缺失**

Run: `.venv/bin/python -m pytest tests/ai/test_cli.py -v`
Expected: FAIL importing `ai.cli`.

- [ ] **Step 3: 实现 argparse 子命令**

`build_parser()` 创建 `train/pause/resume/extend/status`；`train` 参数包括 `--games`（默认 10000）、`--full-moves`（默认 512）、`--device`、`--torch-threads`、`--self-play-workers`、`--simulations`、`--run-dir`。`resume` 读取 checkpoint 中的配置继续，允许设备和线程参数覆盖但拒绝网络/编码不兼容覆盖。`main()` 返回整数退出码，错误写 stderr。

```python
# src/ai/__main__.py
from ai.cli import main

raise SystemExit(main())
```

- [ ] **Step 4: 运行 CLI 测试和帮助烟雾测试**

Run: `.venv/bin/python -m pytest tests/ai/test_cli.py -v && .venv/bin/python -m ai --help`
Expected: PASS and help lists all five commands.

- [ ] **Step 5: 中文提交**

```bash
git add src/ai/cli.py src/ai/__main__.py tests/ai/test_cli.py
git commit -m "功能：增加 AI 训练命令行控制"
```

### Task 10: 文档、CPU/CUDA 烟雾验证与回归

**Files:**
- Modify: `README.md`
- Create: `tests/ai/test_smoke_training.py`

- [ ] **Step 1: 写最短真实 CPU 训练烟雾测试**

```python
# tests/ai/test_smoke_training.py
def test_real_cpu_training_produces_reloadable_checkpoint(tmp_path: Path) -> None:
    config = TrainingConfig(
        target_games=1,
        max_full_moves=1,
        simulations_per_move=1,
        channels=8,
        residual_blocks=1,
        batch_size=1,
        checkpoint_interval_games=1,
        run_dir=tmp_path,
        device="cpu",
    )
    Trainer(config).run()
    assert CheckpointManager(tmp_path).inspect().progress.completed_games == 1
```

- [ ] **Step 2: 运行测试并确认在 README/最终集成完成前暴露缺口**

Run: `.venv/bin/python -m pytest tests/ai/test_smoke_training.py -v`
Expected: 首次运行若失败，只允许修复真实训练路径；不得用 fake game 替换此烟雾测试。

- [ ] **Step 3: 补充 README 操作说明**

README 必须明确：

```bash
python -m pip install -e ".[ai,dev]"
python -m ai train
python -m ai train --games 20000 --full-moves 512 --device cpu --torch-threads 8 --self-play-workers 8
python -m ai train --device cuda:0
python -m ai pause --run-dir AI-runs/default
python -m ai resume --run-dir AI-runs/default
python -m ai extend --run-dir AI-runs/default --games 5000
python -m ai status --run-dir AI-runs/default
```

并说明 512 个完整回合等于 1024 ply、10,000 是累计默认目标、显式 CUDA 不可用会报错、10,000 局只是初始规模而非棋力保证。

- [ ] **Step 4: 运行 AI 和原项目完整验证**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -v`
Expected: all tests PASS; CUDA-only test may SKIP only when `torch.cuda.is_available()` is false.

Run: `.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check . && .venv/bin/python -m compileall -q src`
Expected: all commands exit 0 with no errors.

- [ ] **Step 5: 手工执行暂停、追加和续传烟雾链路**

Run:

```bash
.venv/bin/python -m ai train --run-dir /tmp/xiangqi-ai-smoke --games 2 --full-moves 1 --simulations 1 --device cpu --torch-threads 2 --self-play-workers 1
.venv/bin/python -m ai extend --run-dir /tmp/xiangqi-ai-smoke --games 1
.venv/bin/python -m ai resume --run-dir /tmp/xiangqi-ai-smoke
.venv/bin/python -m ai status --run-dir /tmp/xiangqi-ai-smoke
```

Expected: final JSON reports `completed_games: 3`, `target_games: 3`, `phase: completed`, and a loadable latest checkpoint.

- [ ] **Step 6: 检测 CUDA 能力并在可用时执行真实 CUDA 烟雾训练**

Run: `.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'`

If true, run:

```bash
.venv/bin/python -m ai train --run-dir /tmp/xiangqi-ai-cuda-smoke --games 1 --full-moves 1 --simulations 1 --device cuda:0
```

Expected: checkpoint tensors can be loaded with `map_location="cpu"`; if false, record CUDA runtime as unavailable and rely only on the passing device-selection tests, without claiming a real CUDA run.

- [ ] **Step 7: 中文提交**

```bash
git add README.md tests/ai/test_smoke_training.py
git commit -m "文档：补充 AI 训练与续传说明"
```

## 完成审计

- [ ] `TrainingConfig()` 现场输出 `target_games=10000`、`max_full_moves=512`、`max_plies=1024`。
- [ ] CPU 真实烟雾训练产生训练样本、执行梯度更新并保存可恢复 checkpoint。
- [ ] 两个 CPU worker 的测试记录到不同 PID，证明并发不是配置空壳。
- [ ] 显式 `cuda:0` 在 CUDA 可用机器上完成训练；不可用机器上明确失败且没有静默回退。
- [ ] pause 在完整棋局和原子写入之后生效，resume 从保存的局数/优化器/RNG 继续。
- [ ] extend 增加累计目标且不改变已完成局数。
- [ ] 1024 ply 上限测试证明从不请求第 1025 步。
- [ ] 所有现有象棋/UI/API 测试仍通过。

