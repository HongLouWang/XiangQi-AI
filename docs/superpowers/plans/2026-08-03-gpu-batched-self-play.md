# GPU 多局同步批量自我对弈 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在单个 CUDA 进程内同步推进默认 16 盘自我对弈并批量执行 MCTS 神经网络推理，同时保留 CPU 多进程、合法走子和持久化兼容性。

**Architecture:** 将单次 MCTS 搜索拆成可暂停的 `SearchSession`，调度器每轮从每个活跃棋局取得一个叶子请求，交给 `BatchEvaluator` 一次推理后分别回传。`BatchedSelfPlay` 管理棋局槽位、补位、终局提交和独立随机种子；Trainer 的 CUDA 路径使用该调度器并在明确 OOM 时从最近安全 checkpoint 按 16、8、4、1 降级。

**Tech Stack:** Python 3.12+、PyTorch、NumPy、pytest、Jupyter nbformat 4、Google Colab CUDA

---

### Task 1: 配置、CLI 和 checkpoint 兼容

**Files:**
- Modify: `src/ai/config.py`
- Modify: `src/ai/cli.py`
- Modify: `src/ai/checkpoint.py`
- Modify: `tests/ai/test_cli.py`
- Modify: `tests/ai/test_checkpoint.py`

- [ ] **Step 1: 写配置和 CLI 失败测试**

断言 `TrainingConfig().parallel_games == 16`，非正整数被拒绝；`train --parallel-games 8` 写入配置；`resume --parallel-games 4` 覆盖 checkpoint 配置。

```python
def test_train_accepts_parallel_games(monkeypatch, tmp_path):
    captured = _capture_trainer(monkeypatch)
    assert main(["train", "--run-dir", str(tmp_path), "--parallel-games", "8"]) == 0
    assert captured.config.parallel_games == 8
```

- [ ] **Step 2: 运行确认 RED**

Run: `.venv/bin/python -m pytest tests/ai/test_cli.py -k parallel_games -v`

Expected: FAIL，字段或参数不存在。

- [ ] **Step 3: 实现最小配置与 CLI**

在 `TrainingConfig` 增加 `parallel_games: int = 16` 并纳入正整数校验；`train` 增加默认 16 的 `--parallel-games`，`resume` 增加默认 `None` 的覆盖参数，并传入 `_train`/`_resume`。

- [ ] **Step 4: 写旧 checkpoint 缺字段兼容测试并确认 RED**

构造删除 `config["parallel_games"]` 的安全 payload，期望加载后值为 16；不得放宽其他未知或缺失字段。

- [ ] **Step 5: 实现 checkpoint 配置默认迁移并验证**

仅在反序列化配置时对缺失的 `parallel_games` 注入 16，继续使用现有严格字段验证。

Run: `.venv/bin/python -m pytest tests/ai/test_cli.py tests/ai/test_checkpoint.py -q`

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add src/ai/config.py src/ai/cli.py src/ai/checkpoint.py tests/ai/test_cli.py tests/ai/test_checkpoint.py
git commit -m "功能：增加 CUDA 并行棋局配置"
```

### Task 2: 将 MCTS 拆成可批量调度的搜索会话

**Files:**
- Modify: `src/ai/mcts.py`
- Modify: `tests/ai/test_mcts.py`

- [ ] **Step 1: 写 SearchSession 状态机失败测试**

覆盖 `MCTS.start_search(state, add_noise)`、首次根评估请求、逐次叶子请求、`accept_evaluation(logits, value)`、终局叶子无需网络评估，以及完成后的 visit policy。

```python
session = search.start_search(state, add_noise=False)
request = session.next_evaluation()
assert request.state == state
session.accept_evaluation(logits, 0.25)
while not session.done:
    request = session.next_evaluation()
    session.accept_evaluation(logits_for(request.state), value_for(request.state))
assert session.policy() == search.search(state, add_noise=False)
```

- [ ] **Step 2: 运行确认 RED**

Run: `.venv/bin/python -m pytest tests/ai/test_mcts.py -k search_session -v`

Expected: FAIL，API 不存在。

- [ ] **Step 3: 实现 `EvaluationRequest` 与 `SearchSession`**

`EvaluationRequest` 只保存 `state`、`node`、`path`、`legal_moves`；`SearchSession` 每次最多暴露一个待评估叶子。终局叶子直接用 `-1.0` 回传。根噪声只在根首次扩展后加入一次。

- [ ] **Step 4: 用会话 API 重写同步 `MCTS.search`**

旧 `search` 循环调用 `next_evaluation()` 和现有 evaluator，保持公开行为不变，避免维护两套搜索算法。

- [ ] **Step 5: 运行 MCTS 全部测试**

Run: `.venv/bin/python -m pytest tests/ai/test_mcts.py -q`

Expected: PASS，旧测试结果不变。

- [ ] **Step 6: 提交**

```bash
git add src/ai/mcts.py tests/ai/test_mcts.py
git commit -m "重构：支持可批量调度的 MCTS 会话"
```

### Task 3: 实现一次 CUDA batch 的评估接口

**Files:**
- Modify: `src/ai/network.py`
- Modify: `tests/ai/test_network.py`

- [ ] **Step 1: 写批量与逐局等价失败测试**

创建固定权重小网络和多个 `SearchState`，比较 `TorchEvaluator.evaluate_many(states)` 与逐个 `evaluate` 的 logits/value，使用明确的 `torch.testing.assert_close` 容差。

- [ ] **Step 2: 运行确认 RED**

Run: `.venv/bin/python -m pytest tests/ai/test_network.py -k evaluate_many -v`

Expected: FAIL，方法不存在。

- [ ] **Step 3: 实现单次批量前向**

编码所有状态后用 `np.stack`/`torch.from_numpy` 形成 `[N,C,10,9]`，一次移动到目标 device 并在 `torch.inference_mode()` 中前向；返回与输入顺序一致的 logits 数组和 float value。

- [ ] **Step 4: 让 `evaluate` 委托单元素批量接口**

保持现有协议和数值校验，避免两套推理路径漂移。

- [ ] **Step 5: 验证并提交**

Run: `.venv/bin/python -m pytest tests/ai/test_network.py -q`

Expected: PASS。

```bash
git add src/ai/network.py tests/ai/test_network.py
git commit -m "功能：增加神经网络批量局面评估"
```

### Task 4: 新建多局同步自我对弈调度器

**Files:**
- Create: `src/ai/batched_self_play.py`
- Create: `tests/ai/test_batched_self_play.py`
- Read: `src/ai/self_play.py`

- [ ] **Step 1: 写槽位隔离与批量映射失败测试**

使用假规则局面和记录型 batch evaluator，启动 3 个槽位，断言每次评估 batch 按槽位编号稳定排序，返回结果只更新对应树，历史和 seed 互不共享。

- [ ] **Step 2: 写终局补位与目标停止失败测试**

让不同槽位在不同 ply 终局；断言已完成棋局按 game number 返回，未暂停时补位，达到请求数量后不再创建棋局。

- [ ] **Step 3: 写合法性与上限失败测试**

对真实初始棋盘运行小网络、`simulations=1`、小 ply 上限，断言历史中每步当时合法、将死立即结束、上限结果为和棋。

- [ ] **Step 4: 实现 `GameSlot` 和 `BatchedSelfPlay.generate(count, parallel_games)`**

`GameSlot` 封装 board/side/history/ply/session/RNG/game_number；调度器负责收集 `EvaluationRequest`、调用一次 `evaluate_many`、回传结果、完成走子、终局封装 `GameResult` 和补位。

- [ ] **Step 5: 验证调度器**

Run: `.venv/bin/python -m pytest tests/ai/test_batched_self_play.py -q`

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add src/ai/batched_self_play.py tests/ai/test_batched_self_play.py
git commit -m "功能：实现 CUDA 多局同步自我对弈"
```

### Task 5: 接入 Trainer 并保持 CPU 路径不变

**Files:**
- Modify: `src/ai/trainer.py`
- Modify: `tests/ai/test_trainer.py`

- [ ] **Step 1: 写设备路由失败测试**

CPU 配置继续调用 `_generate_games` 和 multiprocessing pool；CUDA 配置调用 `BatchedSelfPlay.generate`，且 `worker_count == 1`、有效棋局数等于配置值。

- [ ] **Step 2: 写暂停安全边界失败测试**

模拟批量调度器返回完整棋局；断言暂停后不请求下一批、当前返回棋局逐局原子提交、checkpoint 保存后标记 paused。

- [ ] **Step 3: 实现 CUDA 分支**

Trainer 初始化时保存 `cuda_mode`、`parallel_games_requested/effective` 和 batch 统计。CPU while-loop 保持现状；CUDA while-loop按剩余目标调用批量调度器并复用 `_commit_game`。

- [ ] **Step 4: 扩展状态 message**

报告 `self_play_workers_effective=1`、请求/有效并行度、最近 batch 和 OOM 次数；CPU 输出保持兼容并增加不误报 CUDA 批量字段的测试。

- [ ] **Step 5: 验证 Trainer**

Run: `.venv/bin/python -m pytest tests/ai/test_trainer.py -q`

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add src/ai/trainer.py tests/ai/test_trainer.py
git commit -m "功能：接入 CUDA 批量自我对弈训练"
```

### Task 6: OOM 降级与安全恢复

**Files:**
- Modify: `src/ai/trainer.py`
- Modify: `tests/ai/test_trainer.py`

- [ ] **Step 1: 写候选序列和错误分类失败测试**

断言 16 得到 `[16,8,4,1]`，10 得到 `[10,5,2,1]`；只有 `torch.OutOfMemoryError` 或明确 CUDA OOM 才降级，普通 RuntimeError 直接失败。

- [ ] **Step 2: 写不重复 Replay 的失败测试**

先成功提交若干完整棋局，再让下一批 OOM；恢复后断言 manifest game number 唯一、已完成计数不回退且未重复训练相同 Replay 条目。

- [ ] **Step 3: 实现降级循环**

OOM 时释放批量调度器与临时 Tensor、`torch.cuda.empty_cache()`，增加计数、选择下一并行度并调用现有严格 checkpoint/Replay 恢复边界。并行度 1 OOM 写 failed 并重新抛出。

- [ ] **Step 4: 验证失败与恢复路径**

Run: `.venv/bin/python -m pytest tests/ai/test_trainer.py -k 'oom or parallel_games' -v`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/ai/trainer.py tests/ai/test_trainer.py
git commit -m "功能：支持 CUDA OOM 自动降级"
```

### Task 7: 更新 Colab Notebook 与训练文档

**Files:**
- Modify: `docs/XiangQi-AI-Colab-GPU-Training.ipynb`
- Modify: `tests/test_colab_gpu_notebook.py`
- Modify: `docs/AI-TRAINING-GUIDE.md`
- Modify: `README.md`

- [ ] **Step 1: 写 Notebook 参数与命令失败测试**

断言 `PARALLEL_GAMES = 16`、新训练命令和 resume 命令都包含 `--parallel-games`，状态说明区分 worker 进程和并行棋局。

- [ ] **Step 2: 运行确认 RED**

Run: `.venv/bin/python -m pytest tests/test_colab_gpu_notebook.py -k parallel_games -v`

Expected: FAIL。

- [ ] **Step 3: 更新 Notebook**

参数单元加入 `PARALLEL_GAMES = 16`；train/resume argv 显式传参；状态单元显示有效并行度；增加单槽位与 16 槽位基准说明以及 `nvidia-smi` 观察命令。保持 Drive 外置锁、安全清理和无执行输出。

- [ ] **Step 4: 更新文档**

说明 CPU 的 `self_play_workers` 是进程数，CUDA 的 `parallel_games` 是单进程并行棋局数；记录 OOM 降级、调优和 checkpoint 丢失窗口。

- [ ] **Step 5: 验证并提交**

Run: `.venv/bin/python -m pytest tests/test_colab_gpu_notebook.py -q`

Expected: PASS。

```bash
git add docs/XiangQi-AI-Colab-GPU-Training.ipynb tests/test_colab_gpu_notebook.py docs/AI-TRAINING-GUIDE.md README.md
git commit -m "文档：增加 Colab GPU 批量训练配置"
```

### Task 8: 全量验证与 Colab 交付

**Files:**
- Verify: `src/ai/`
- Verify: `tests/ai/`
- Verify: `docs/XiangQi-AI-Colab-GPU-Training.ipynb`

- [ ] **Step 1: 运行 AI 专项**

Run: `.venv/bin/python -m pytest tests/ai tests/test_colab_gpu_notebook.py -q`

Expected: PASS。

- [ ] **Step 2: 运行静态验证**

Run: `.venv/bin/python -m ruff check .`

Expected: `All checks passed!`

Run: `.venv/bin/python -m ruff format --check src/ai tests/ai tests/test_colab_gpu_notebook.py`

Expected: PASS。

Run: `.venv/bin/python -m compileall -q src tests && git diff --check`

Expected: exit 0。

- [ ] **Step 3: 运行仓库全量测试**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q`

Expected: 报告准确通过、跳过和任何既有时序失败，不得把专项结果冒充全量通过。

- [ ] **Step 4: 原位覆盖 Drive Notebook 并回读**

通过 Google Drive `update_file` 更新现有文件 ID `1bSAvSfcmNjhgt99VvpM606Clov49_y5A`，保持名称和父目录；回读确认 `PARALLEL_GAMES = 16`、`--parallel-games`、外置锁及空 outputs。

- [ ] **Step 5: 在 Colab 执行真实 CUDA smoke 与基准**

先用 `parallel_games=1` 完成小规模基线，再用自动 16 完成相同参数运行；记录局/小时、GPU 利用率、峰值显存、平均 batch、有效并行度和 OOM 次数。真实 CUDA 未执行前只报告本地调度验证，不宣称 A100 加速结果。
