# 中国象棋 AlphaZero AI 训练使用手册

本手册介绍如何独立训练项目中的中国象棋 AI，包括 CPU、CUDA、暂停、追加、
断点续传和模型文件管理。

## 1. 系统边界

AI 训练代码位于 `src/ai/`，采用以下 AlphaZero 风格流程：

```text
中国象棋合法局面
      ↓
Policy/Value ResNet
      ↓
PUCT MCTS 搜索
      ↓
自我对弈生成训练样本
      ↓
Replay Buffer 与梯度训练
      ↓
持久化 checkpoint 和模型
```

AI 目前是独立训练工具，只复用现有规则引擎生成合法着法并判断终局，**没有接入**
桌面 UI、controller 或 HTTP API。训练生成的每一步都会经过合法走法校验。

## 2. 环境与依赖安装

在项目根目录执行：

```bash
cd /Volumes/MobileWork/XiangQi-AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ai,dev]"
```

确认训练命令可用：

```bash
python -m ai --help
python -m ai train --help
```

如果已经进入项目的虚拟环境，后续命令中的 `python` 会自动使用 `.venv`。

## 3. 第一次最小 CPU 试跑

不要第一次就直接运行默认的 10,000 局。先用单独目录训练一盘极小测试局：

```bash
python -m ai train \
  --run-dir AI-runs/smoke \
  --games 1 \
  --full-moves 1 \
  --simulations 1 \
  --channels 8 \
  --residual-blocks 1 \
  --batch-size 1 \
  --checkpoint-interval-games 1 \
  --device cpu
```

查询结果：

```bash
python -m ai status --run-dir AI-runs/smoke
```

成功时状态中应包含类似内容：

```json
{
  "completed_games": 1,
  "phase": "completed",
  "target_games": 1,
  "training_steps": 1
}
```

再确认模型和断点已经落盘：

```bash
ls -lh AI-runs/smoke
```

该命令只验证训练、保存和恢复链路，不能用来评估 AI 棋力。

## 4. 默认训练规则

直接执行以下命令时：

```bash
python -m ai train
```

默认行为是：

- 累计训练目标：10,000 局；
- 单局上限：512 个完整回合；
- 一个完整回合是红方和黑方各走一步，因此上限为 1024 ply；
- 1024 ply 之前出现将死或困毙，立即按真实胜负结束；
- 第 1024 ply 本身造成将死，也按真实胜负结束；
- 只有完成 1024 ply 后仍未终局，才按和棋生成训练标签；
- 默认运行目录：`AI-runs/default`；
- `--games` 表示该运行目录的累计目标，不是本次额外增加的局数。

10,000 局只是默认的初始训练规模，不代表训练后一定达到某个棋力等级。实际效果
取决于网络大小、每步 MCTS 模拟数、总训练局数、硬件和训练时间。

## 5. CPU 正式训练

CPU 模式有两个不同的并行参数：

- `--torch-threads`：每个 PyTorch 进程用于神经网络计算的线程数；
- `--self-play-workers`：同时生成自我对弈棋局的独立进程数。

示例：

```bash
python -m ai train \
  --run-dir AI-runs/cpu-main \
  --games 10000 \
  --full-moves 512 \
  --device cpu \
  --torch-threads 4 \
  --self-play-workers 4 \
  --simulations 64
```

不要盲目把两个参数都设为 CPU 逻辑核心数，否则容易出现过度订阅。例如 8 个
worker、每个又使用 8 个 PyTorch 线程，理论上会争抢 64 个执行线程。推荐从
以下关系开始尝试：

```text
self_play_workers × torch_threads ≈ CPU 性能核心数
```

示例：8 核 CPU 可先试：

```bash
--self-play-workers 4 --torch-threads 2
```

想提高棋力时，优先逐步增加 `--simulations`、网络通道数、残差块和总训练局数，
但这些参数都会明显增加训练时间与内存占用。

## 6. CUDA/GPU 训练

如果本机没有 NVIDIA GPU，可以使用
[Colab GPU 训练 Notebook](XiangQi-AI-Colab-GPU-Training.ipynb)。它会把源码、
临时目录、依赖缓存、日志、Replay、checkpoint 和最终模型全部保存在 Google
Drive 的 `MyDrive/XiangQi-AI/`，并提供暂停、追加和断点恢复单元格。

先检测当前 PyTorch 是否能使用 CUDA：

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())
print("GPU 数量:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("当前 GPU:", torch.cuda.get_device_name(0))
PY
```

使用默认 CUDA 设备：

```bash
python -m ai train \
  --run-dir AI-runs/gpu-main \
  --games 10000 \
  --device cuda \
  --self-play-workers 4 \
  --parallel-games 16
```

指定第 0 张 GPU：

```bash
python -m ai train \
  --run-dir AI-runs/gpu-0 \
  --games 10000 \
  --device cuda:0
```

`--device auto` 会在 CUDA 可用时选择 CUDA，否则选择 CPU。显式指定 `cuda` 或
`cuda:N` 而 CUDA 不可用、设备编号不存在时，程序会直接报错，不会静默退回
CPU。

当前开发机器检测结果是 CUDA 不可用，因此 CUDA 的设备选择、错误分支和张量
语义已有自动化测试，但本机没有完成真实 NVIDIA GPU 烟雾训练。首次在 CUDA
机器运行时，建议仍先使用 `--games 1 --full-moves 1 --simulations 1` 做验证。

CUDA 模式下，多个 CPU 生产进程各自生成棋局和 MCTS 搜索请求，主进程
独占 CUDA context，将来自不同 worker 的请求合并后在单张 GPU 上推理。
`--self-play-workers` 是 CPU 生产进程数；`--parallel-games`（默认 16）是最多
在途棋局数，不是保证同时活跃的棋局数。有效 worker 数不会超过这个上限。

每完成一局，训练器都会立即提交 Replay 并更新状态；定期 checkpoint 仍由
`--checkpoint-interval-games` 决定。因此完成局数不再需要等待整组在途棋局
全部结束才可见。可在 `status` 的 `message` 中观察
`self_play_workers_requested`、`self_play_workers_effective`、
`parallel_games_requested`、`parallel_games_effective`、`active_games`、
`last_inference_batch_size`、`max_inference_batch_size` 和 `inference_requests`。

调参时先根据 CPU 核数增加 `--self-play-workers`，再观察实际推理 batch 和
GPU 利用率，最后才增加 `--parallel-games`。T4 等 GPU 搭配较少 CPU 核时，
合法着法生成和 MCTS 树操作可能是主要瓶颈；显存很空不代表应先扩大网络
或在途棋局上限。

Colab 正式训练应在 Notebook 单元格前台运行，以便会话持续跟踪活跃
计算。不要把长时训练作为脱离 Notebook 的后台子进程；Colab 仍可能因配额、
网络或 runtime 回收而中断，所以必须保留 Drive 中的整个运行目录以断点续传。

## 7. 暂停、恢复、追加和状态查询

以下示例假设运行目录为 `AI-runs/cpu-main`。

### 查询状态

```bash
python -m ai status --run-dir AI-runs/cpu-main
```

主要字段：

- `phase`：`running`、`paused`、`completed` 或 `failed` 等阶段；
- `completed_games`：已经完整提交的棋局数；
- `target_games`：当前累计目标；
- `training_steps`：已经执行的梯度更新次数；
- `device`：实际训练设备；
- `message`：worker/在途棋局/推理 batch 指标，或错误摘要。

### 安全暂停

```bash
python -m ai pause --run-dir AI-runs/cpu-main
```

暂停是协作式的。程序会完成当前最小安全工作单元，将完整棋局、Replay、模型、
优化器和状态写入磁盘后才进入 `paused`，因此命令发出后可能不会瞬间停止。

### 断点恢复

```bash
python -m ai resume --run-dir AI-runs/cpu-main
```

恢复时可以覆盖设备和 CPU 并行参数：

```bash
python -m ai resume \
  --run-dir AI-runs/cpu-main \
  --device cpu \
  --torch-threads 4 \
  --self-play-workers 4
```

网络架构、Replay 和累计进度仍使用 checkpoint 中保存的配置，不会从零开始。

同一运行目录存在有效 checkpoint 时再次执行 `train` 也会自动续传。损坏的
checkpoint 或非空但没有有效 checkpoint 的目录会被拒绝，不会静默覆盖。

### 追加训练局数

```bash
python -m ai extend --run-dir AI-runs/cpu-main --games 5000
```

如果当前目标是 10,000 局，该命令会把累计目标改为 15,000 局，不会清空已完成
局数、Replay、模型或优化器。

- 训练进程仍在运行：它会读取新目标并继续训练；
- 已暂停或已完成：追加后执行 `resume`。

```bash
python -m ai resume --run-dir AI-runs/cpu-main
```

## 8. 模型与训练数据持久化

每个运行目录包含：

```text
AI-runs/cpu-main/
├── checkpoint-a.pt
├── checkpoint-b.pt
├── final_model.pt
├── latest.json
├── status.json
├── control.lock
└── replay/
    ├── manifest.json
    ├── games/
    └── migration-v1-to-v2.json  # 仅在旧 Replay 迁移过程中可能短暂存在
```

文件作用：

- `checkpoint-a.pt`、`checkpoint-b.pt`：轮换保存的完整断点，包括模型、优化器、
  训练进度、配置和随机状态；最新槽损坏时可以回退上一槽；
- `latest.json`：当前优先加载的 checkpoint 槽；
- `final_model.pt`：达到当前累计目标后导出的纯模型权重；
- `status.json`：可读训练状态；
- `replay/manifest.json`：Replay 版本、累计局数和样本计数；
- `replay/games/`：按完整棋局原子保存的训练样本；
- migration sidecar：旧版 Replay 安全迁移期间使用，成功后会自动删除。

安全读取纯模型权重：

```python
import torch

weights = torch.load(
    "AI-runs/cpu-main/final_model.pt",
    map_location="cpu",
    weights_only=True,
)
print(f"读取了 {len(weights)} 个权重张量")
```

需要完整续训时必须保留整个运行目录，而不是只复制 `final_model.pt`。

## 9. 使用 Python 修改训练参数

可以绕过 CLI，直接构造不可变的 `TrainingConfig`：

```python
from pathlib import Path

from ai.config import TrainingConfig
from ai.trainer import Trainer

config = TrainingConfig(
    target_games=20_000,
    max_full_moves=512,
    device="cpu",
    torch_threads=4,
    self_play_workers=4,
    simulations_per_move=128,
    residual_blocks=6,
    channels=96,
    batch_size=256,
    replay_capacity_games=2_000,
    learning_rate=1e-3,
    checkpoint_interval_games=10,
    game_retry_limit=2,
    seed=42,
    run_dir=Path("AI-runs/python-main"),
)

Trainer(config).run()
```

主要字段：

| 字段 | 含义 | 默认值 |
|---|---|---:|
| `target_games` | 累计训练目标局数 | `10000` |
| `max_full_moves` | 单局完整回合上限 | `512` |
| `device` | `auto`、`cpu`、`cuda` 或 `cuda:N` | `auto` |
| `torch_threads` | 每个 PyTorch 进程的线程数 | `1` |
| `self_play_workers` | CPU 自我对弈生产进程数（CPU/CUDA 都生效） | `1` |
| `parallel_games` | CUDA 最多在途棋局数 | `16` |
| `simulations_per_move` | 每步 MCTS 模拟次数 | `64` |
| `residual_blocks` | ResNet 残差块数 | `4` |
| `channels` | 网络主干通道数 | `64` |
| `batch_size` | 每次梯度更新的样本数 | `128` |
| `replay_capacity_games` | Replay 最多保留的最近棋局数 | `2000` |
| `learning_rate` | Adam 学习率 | `0.001` |
| `checkpoint_interval_games` | 每多少局定期保存一次 checkpoint | `10` |
| `game_retry_limit` | 单局 worker 失败后的重试次数 | `2` |
| `seed` | 随机种子 | `0` |
| `run_dir` | 持久化运行目录 | `AI-runs/default` |

`max_plies` 是只读派生属性，始终等于 `max_full_moves × 2`。

## 10. 推荐配置模板

以下只是起点，需要根据机器资源和训练速度调整。

### 最小链路验证

```bash
python -m ai train \
  --run-dir AI-runs/smoke \
  --games 1 --full-moves 1 --simulations 1 \
  --channels 8 --residual-blocks 1 --batch-size 1 \
  --checkpoint-interval-games 1 --device cpu
```

### 8 核 CPU 起步配置

```bash
python -m ai train \
  --run-dir AI-runs/cpu-main \
  --games 10000 --full-moves 512 \
  --device cpu --torch-threads 2 --self-play-workers 4 \
  --simulations 64
```

### 单张 NVIDIA GPU 起步配置

```bash
python -m ai train \
  --run-dir AI-runs/gpu-main \
  --games 10000 --full-moves 512 \
  --device cuda:0 --self-play-workers 4 --parallel-games 16 \
  --simulations 64
```

先使 worker 数与可用 CPU 核数匹配，再根据状态中的实际推理 batch 调整
`--parallel-games`。如果显存不足，优先降低 `--batch-size`、`--channels` 或
`--residual-blocks`。

## 11. 常见问题与安全恢复

### 显式 CUDA 报“不可用”

先运行第 6 节检测命令。常见原因是机器没有 NVIDIA GPU、PyTorch 安装包不含
CUDA、驱动不可用或设备编号不存在。macOS 的 Apple GPU 不是 CUDA 设备；当前
训练器只支持 CPU 和 CUDA，不支持 MPS。

### `pause` 后进程没有立即退出

这是正常的安全暂停行为。等待当前完整棋局和 checkpoint 写入结束，然后再次用
`status` 查看是否进入 `paused`。

### 追加后状态是 `paused`

已经完成的运行追加目标后需要执行：

```bash
python -m ai resume --run-dir AI-runs/cpu-main
```

### checkpoint 或 Replay 报损坏/不兼容

不要手工删除槽位、修改 `latest.json`、`manifest.json` 或 migration sidecar 后强行
启动。先停止训练进程，再复制整个运行目录做备份，保留原始错误信息后检查磁盘
空间、文件权限和版本。双槽 checkpoint 会自动尝试上一份有效断点。

### 如何备份正在训练的模型

最稳妥的方法是先请求安全暂停：

```bash
python -m ai pause --run-dir AI-runs/cpu-main
```

确认 `phase` 为 `paused` 后，复制整个 `AI-runs/cpu-main/`。只复制
`final_model.pt` 可以用于加载权重，但不能完整续训。

### Worker 偶发失败

系统默认对同一棋局 seed 重试 2 次，并记录棋局编号和 traceback。超过限制后进入
`failed`，不会提交半盘数据，并尽力保存最近 checkpoint。处理外部原因后可从完整
运行目录恢复。

### 如何查看所有参数

```bash
python -m ai train --help
python -m ai resume --help
```

## 12. 命令速查

```bash
# 默认训练
python -m ai train

# CPU 并行训练
python -m ai train --device cpu --torch-threads 2 --self-play-workers 4

# CUDA 训练
python -m ai train --device cuda:0 --self-play-workers 4 --parallel-games 16

# 状态
python -m ai status --run-dir AI-runs/default

# 安全暂停
python -m ai pause --run-dir AI-runs/default

# 断点恢复
python -m ai resume --run-dir AI-runs/default

# 在累计目标上追加 5,000 局
python -m ai extend --run-dir AI-runs/default --games 5000
```
