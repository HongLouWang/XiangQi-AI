# Colab GPU 训练 Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成可直接在 Google Colab 顺序运行的 GPU 训练 Notebook，使源码、缓存、临时文件、日志、Replay、checkpoint 和模型全部持久化到 Google Drive 的 `MyDrive/XiangQi-AI/`。

**Architecture:** Notebook 使用纯 Python 代码单元格挂载 Drive、管理 Drive 源码、安装依赖、检查 CUDA，并通过后台 `subprocess.Popen` 启动训练。控制命令使用参数数组执行；持久化 PID 和日志允许用户查询、暂停、追加，并在 Colab 断线后从 Drive 恢复。

**Tech Stack:** Jupyter nbformat 4、Google Colab、Google Drive、Python、PyTorch CUDA、现有 `python -m ai` CLI、pytest。

---

### Task 1: 建立 Notebook 结构和静态契约测试

**Files:**
- Create: `docs/XiangQi-AI-Colab-GPU-Training.ipynb`
- Create: `tests/test_colab_gpu_notebook.py`

- [ ] **Step 1: 写 Notebook 契约失败测试**

```python
import ast
import json
from pathlib import Path


NOTEBOOK = Path("docs/XiangQi-AI-Colab-GPU-Training.ipynb")


def _load() -> dict[str, object]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_colab_notebook_is_nbformat_4_with_compilable_code() -> None:
    notebook = _load()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert notebook["metadata"]["colab"]["name"] == NOTEBOOK.name
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_all_persistent_paths_are_below_drive_root() -> None:
    source = NOTEBOOK.read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/XiangQi-AI" in source
    for required in ("source", "runs", "temp", "cache", "logs"):
        assert required in source


def test_notebook_contains_complete_gpu_training_workflow() -> None:
    source = NOTEBOOK.read_text(encoding="utf-8")
    for required in (
        "drive.mount",
        "torch.cuda.is_available",
        'DEVICE = "cuda:0"',
        "TARGET_GAMES = 10_000",
        "MAX_FULL_MOVES = 512",
        "subprocess.Popen",
        '"pause"',
        '"extend"',
        '"resume"',
        'weights_only=True',
    ):
        assert required in source


def test_notebook_never_uses_unsafe_or_destructive_operations() -> None:
    source = NOTEBOOK.read_text(encoding="utf-8")
    for forbidden in (
        "weights_only=False",
        "reset --hard",
        "checkout -f",
        "pull --force",
        "shutil.rmtree",
    ):
        assert forbidden not in source
```

- [ ] **Step 2: 运行测试并确认 Notebook 尚不存在**

Run: `.venv/bin/python -m pytest tests/test_colab_gpu_notebook.py -v`

Expected: FAIL with `FileNotFoundError` for `docs/XiangQi-AI-Colab-GPU-Training.ipynb`.

- [ ] **Step 3: 创建合法 nbformat 4 Notebook 骨架**

Notebook 顶层必须包含：

```json
{
  "cells": [],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "name": "XiangQi-AI-Colab-GPU-Training.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {"name": "python"}
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

使用 Python 标准库 `json` 生成或更新 Notebook，禁止手写无效 JSON。所有代码单元格必须是可由 `ast.parse` 编译的纯 Python，不使用 `%pip` 或 `!command` 魔法。

- [ ] **Step 4: 运行骨架测试并保留缺失工作流的预期失败**

Run: `.venv/bin/python -m pytest tests/test_colab_gpu_notebook.py -v`

Expected: nbformat/compile test PASS，路径或完整工作流测试 FAIL，证明测试仍约束实际内容。

### Task 2: 实现 Drive、源码、安装和 CUDA 单元格

**Files:**
- Modify: `docs/XiangQi-AI-Colab-GPU-Training.ipynb`
- Modify: `tests/test_colab_gpu_notebook.py`

- [ ] **Step 1: 添加说明和 Drive 挂载单元格**

代码必须使用：

```python
from google.colab import drive

drive.mount("/content/drive")
```

目录单元格定义：

```python
from pathlib import Path
import os

DRIVE_ROOT = Path("/content/drive/MyDrive/XiangQi-AI")
SOURCE_DIR = DRIVE_ROOT / "source"
RUNS_DIR = DRIVE_ROOT / "runs"
TEMP_DIR = DRIVE_ROOT / "temp"
CACHE_DIR = DRIVE_ROOT / "cache"
LOGS_DIR = DRIVE_ROOT / "logs"

for directory in (DRIVE_ROOT, RUNS_DIR, TEMP_DIR, CACHE_DIR / "pip", CACHE_DIR / "torch", LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

os.environ["TMPDIR"] = str(TEMP_DIR)
os.environ["PIP_CACHE_DIR"] = str(CACHE_DIR / "pip")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
```

- [ ] **Step 2: 添加首次克隆和安全更新单元格**

首次克隆使用参数数组：

```python
REPOSITORY_URL = "https://github.com/HongLouWang/XiangQi-AI.git"
if not (SOURCE_DIR / ".git").is_dir():
    subprocess.run(["git", "clone", REPOSITORY_URL, str(SOURCE_DIR)], check=True)
```

安全更新先执行 `git status --porcelain`；只有输出为空时才执行：

```python
subprocess.run(["git", "-C", str(SOURCE_DIR), "pull", "--ff-only"], check=True)
```

- [ ] **Step 3: 添加安装和 CUDA 强校验单元格**

安装使用：

```python
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", f"{SOURCE_DIR}[ai]"],
    check=True,
)
```

CUDA 检查使用 `nvidia-smi` 和 PyTorch；若 `torch.cuda.is_available()` 为 false，抛出中文 `RuntimeError`，不改用 CPU。

- [ ] **Step 4: 运行静态测试**

Run: `.venv/bin/python -m pytest tests/test_colab_gpu_notebook.py -v`

Expected: Drive 路径、clone、安全更新、安装和 CUDA 关键字测试 PASS；训练控制测试仍可保持失败直到下一任务。

### Task 3: 实现训练、控制、恢复和模型检查单元格

**Files:**
- Modify: `docs/XiangQi-AI-Colab-GPU-Training.ipynb`
- Modify: `tests/test_colab_gpu_notebook.py`

- [ ] **Step 1: 添加可编辑的正式训练配置**

```python
RUN_NAME = "colab-gpu"
TARGET_GAMES = 10_000
MAX_FULL_MOVES = 512
DEVICE = "cuda:0"
SIMULATIONS = 64
CHANNELS = 64
RESIDUAL_BLOCKS = 4
BATCH_SIZE = 128
CHECKPOINT_INTERVAL_GAMES = 10
GAME_RETRY_LIMIT = 2
SEED = 0
RUN_DIR = RUNS_DIR / RUN_NAME
LOG_PATH = LOGS_DIR / f"{RUN_NAME}.log"
PID_PATH = DRIVE_ROOT / "colab-training.pid"
```

- [ ] **Step 2: 添加同步 CUDA smoke 单元格**

Smoke 使用 Drive 的 `runs/colab-gpu-smoke`，通过参数数组调用当前 CLI；训练后同步调用 `status`，检查 `final_model.pt` 和 checkpoint 槽存在。

- [ ] **Step 3: 添加安全后台启动函数**

定义三个函数：`process_matches(pid: int) -> bool` 检查 PID 命令行，
`training_command(resume: bool) -> list[str]` 生成参数数组，
`start_background_training(resume: bool = False) -> int` 启动进程并返回 PID。

`process_matches` 读取 `/proc/<pid>/cmdline`，要求同时包含 `-m`、`ai` 和当前 `RUN_DIR`。`start_background_training` 在已运行时拒绝重复启动；日志以 append 模式打开并传给 `subprocess.Popen`，使用 `cwd=SOURCE_DIR` 和当前环境；PID 原子写到 Drive。

- [ ] **Step 4: 添加 status、日志、暂停、追加和恢复单元格**

控制命令统一通过：

```python
def run_ai_command(*arguments: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ai", *arguments],
        cwd=SOURCE_DIR,
        env=os.environ.copy(),
        check=True,
        text=True,
        capture_output=capture_output,
    )
```

暂停后最多轮询 30 分钟，每 10 秒查询一次状态。追加局数由单元格变量
`ADDITIONAL_GAMES = 5_000` 控制。恢复使用 `start_background_training(resume=True)`。

- [ ] **Step 5: 添加模型和断线恢复单元格**

模型验证遍历 checkpoint、final model、manifest、status 和日志；仅对存在的 `.pt`
文件执行：

```python
torch.load(path, map_location="cpu", weights_only=True)
```

最后一个 Markdown 单元格列出断线恢复顺序，并强调不要同时启动同一 `RUN_DIR`。

- [ ] **Step 6: 运行完整 Notebook 测试**

Run: `.venv/bin/python -m pytest tests/test_colab_gpu_notebook.py -v`

Expected: all tests PASS；所有代码单元格可编译，默认值、Drive 路径、安全命令和完整工作流均存在。

### Task 4: 文档链接、模拟执行验证和提交

**Files:**
- Modify: `README.md`
- Modify: `docs/AI-TRAINING-GUIDE.md`
- Modify: `tests/test_colab_gpu_notebook.py`

- [ ] **Step 1: 增加 Notebook 链接**

README 和 `docs/AI-TRAINING-GUIDE.md` 的 CUDA 章节都增加相对链接：

```markdown
[Colab GPU 训练 Notebook](XiangQi-AI-Colab-GPU-Training.ipynb)
```

README 从仓库根目录链接时使用 `docs/XiangQi-AI-Colab-GPU-Training.ipynb`。

- [ ] **Step 2: 增加本地模拟验证**

测试从 Notebook 提取所有代码单元格并编译；另外验证训练命令由列表构成、不含
`shell=True`、所有持久化目录派生自 `DRIVE_ROOT`、PID 校验包含 run dir。

Colab 专有的 Drive 挂载和真实 CUDA 训练不能在当前 macOS 环境执行，因此不把
静态测试描述为真实 Colab GPU 验证。

- [ ] **Step 3: 运行最终验证**

Run:

```bash
.venv/bin/python -m pytest tests/test_colab_gpu_notebook.py -v
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m compileall -q src tests
git diff --check
```

Expected: all commands exit 0；CUDA 相关 skip 仍只反映当前机器无 NVIDIA CUDA。

- [ ] **Step 4: 中文提交**

```bash
git add README.md docs/AI-TRAINING-GUIDE.md docs/XiangQi-AI-Colab-GPU-Training.ipynb tests/test_colab_gpu_notebook.py
git commit -m "文档：新增 Colab GPU 训练 Notebook"
```

## 完成审计

- [ ] Notebook 是合法 nbformat 4，所有代码单元格可编译。
- [ ] Google Drive 根目录固定为 `MyDrive/XiangQi-AI`。
- [ ] 源码、temp、cache、logs、Replay、checkpoint 和最终模型均位于 Drive。
- [ ] CUDA 不可用时明确停止，不回退 CPU。
- [ ] 默认配置是 10,000 局、512 完整回合和 `cuda:0`。
- [ ] 支持 smoke、后台训练、状态、日志、暂停、追加、恢复和模型安全加载。
- [ ] PID 校验防止同一运行目录重复启动。
- [ ] v1/v2 Replay 和 checkpoint 的恢复完全交给现有训练器，不在 Notebook 手工编辑文件。
- [ ] README 与完整手册均链接 Notebook。
- [ ] 未修改现有桌面程序和训练核心。
