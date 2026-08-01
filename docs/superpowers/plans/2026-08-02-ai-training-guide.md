# AI 训练使用手册 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增可独立完成安装、CPU/CUDA 训练、暂停、追加、续传和模型检查的中文操作手册，并从 README 提供入口。

**Architecture:** `docs/AI-TRAINING-GUIDE.md` 承载完整教程，按实际操作顺序组织；README 只增加醒目的完整手册链接，不复制大段内容。所有参数和示例以当前 CLI 帮助、`TrainingConfig` 和短程 CPU smoke 为权威来源。

**Tech Stack:** Markdown、Python 3.11+、当前 `python -m ai` CLI、PyTorch。

---

### Task 1: 编写完整中文训练手册

**Files:**
- Create: `docs/AI-TRAINING-GUIDE.md`

- [ ] **Step 1: 记录当前命令和配置来源**

Run:

```bash
.venv/bin/python -m ai --help
.venv/bin/python -m ai train --help
.venv/bin/python -m ai resume --help
.venv/bin/python - <<'PY'
from dataclasses import asdict
from ai.config import TrainingConfig
print(asdict(TrainingConfig()))
PY
```

Expected: 帮助列出 `train/pause/resume/extend/status`，默认配置输出 `target_games=10000`、`max_full_moves=512`，且字段名与手册 Python 示例一致。

- [ ] **Step 2: 创建手册并按固定结构写入完整内容**

`docs/AI-TRAINING-GUIDE.md` 使用以下目录，不省略章节：

```markdown
# 中国象棋 AlphaZero AI 训练使用手册

## 1. 系统边界
## 2. 环境与依赖安装
## 3. 第一次最小 CPU 试跑
## 4. 默认训练规则
## 5. CPU 正式训练
## 6. CUDA/GPU 训练
## 7. 暂停、恢复、追加和状态查询
## 8. 模型与训练数据持久化
## 9. 使用 Python 修改训练参数
## 10. 推荐配置模板
## 11. 常见问题与安全恢复
## 12. 命令速查
```

文档必须包含可复制的以下最小试跑：

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
python -m ai status --run-dir AI-runs/smoke
```

Python 示例使用真实字段：

```python
from pathlib import Path

from ai.config import TrainingConfig
from ai.trainer import Trainer

config = TrainingConfig(
    target_games=20_000,
    max_full_moves=512,
    device="cpu",
    torch_threads=8,
    self_play_workers=8,
    simulations_per_move=128,
    batch_size=256,
    checkpoint_interval_games=10,
    game_retry_limit=2,
    run_dir=Path("AI-runs/cpu-main"),
)
Trainer(config).run()
```

- [ ] **Step 3: 自审手册的事实与安全边界**

Run:

```bash
rg -n "10000|10,000|512|1024|torch-threads|self-play-workers|cuda|pause|resume|extend|checkpoint|final_model|weights_only|TrainingConfig" docs/AI-TRAINING-GUIDE.md
rg -n "TBD|TODO|待定|稍后补充" docs/AI-TRAINING-GUIDE.md
```

Expected: 第一条覆盖全部关键主题；第二条没有输出。文档明确当前机器未完成真实 CUDA 验证，不承诺 10,000 局后的棋力，不建议手工编辑或删除 checkpoint/manifest/sidecar。

### Task 2: README 入口与命令验证

**Files:**
- Modify: `README.md`
- Test: `docs/AI-TRAINING-GUIDE.md`

- [ ] **Step 1: 在 README 独立训练章节增加完整手册链接**

在 `## 独立 AlphaZero 训练` 的首段后增加：

```markdown
完整的安装、CPU/CUDA 配置、暂停、追加、断点续传和故障排查说明见
[AI 训练使用手册](docs/AI-TRAINING-GUIDE.md)。
```

- [ ] **Step 2: 运行真实最小 CPU smoke**

Run:

```bash
GUIDE_SMOKE_DIR=$(mktemp -d /tmp/xiangqi-guide-smoke-XXXXXX)
.venv/bin/python -m ai train \
  --run-dir "$GUIDE_SMOKE_DIR" \
  --games 1 \
  --full-moves 1 \
  --simulations 1 \
  --channels 8 \
  --residual-blocks 1 \
  --batch-size 1 \
  --checkpoint-interval-games 1 \
  --device cpu
.venv/bin/python -m ai status --run-dir "$GUIDE_SMOKE_DIR"
```

Expected: JSON 包含 `"phase": "completed"`、`"completed_games": 1`、`"training_steps": 1`，运行目录存在 `final_model.pt` 和至少一个 checkpoint 槽。

- [ ] **Step 3: 验证文档、格式和工作区范围**

Run:

```bash
git diff --check
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
git status --short
```

Expected: 前三条退出 0；Git 状态只包含 `README.md`、`docs/AI-TRAINING-GUIDE.md` 和用户原有未跟踪 `.vscode/`，不包含 `src/xiangqi`、UI、controller 或 API 文件。

- [ ] **Step 4: 中文提交**

```bash
git add README.md docs/AI-TRAINING-GUIDE.md
git commit -m "文档：新增 AI 训练使用手册"
```

## 完成审计

- [ ] README 链接可以解析到完整手册。
- [ ] 最小 CPU 命令完成真实一局训练并产生持久化模型。
- [ ] CPU 线程、CPU worker、CUDA、暂停、恢复、追加和状态命令均有示例。
- [ ] 默认 10,000 局、512 完整回合和 1024 ply 的语义无歧义。
- [ ] Python 示例字段全部存在于当前 `TrainingConfig`。
- [ ] 手册没有声称本机已经完成真实 CUDA 训练。
- [ ] `.vscode/` 保持未修改、未暂存。
