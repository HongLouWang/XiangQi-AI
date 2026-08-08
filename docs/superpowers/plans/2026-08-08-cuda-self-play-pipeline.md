# CUDA 多进程自我对弈流水线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用多个 CPU spawn worker 为单个主进程 CUDA 模型生产 MCTS 推理请求，逐局持久化结果和进度，并把 Colab 正式训练改为前台运行。

**Architecture:** 新模块 `ai.cuda_pipeline` 让 worker 持有棋局/MCTS，通过请求队列向主进程的推理 broker 请求网络评估；主进程合批推理、路由响应并以迭代器逐局接收结果。`Trainer` 继续独占 GPU、Replay、优化器、状态和 checkpoint，Notebook 直接以前台 CLI 运行训练。

**Tech Stack:** Python 3.11+、`multiprocessing` spawn、PyTorch CUDA、NumPy、pytest、Jupyter nbformat JSON、Ruff。

---

## 文件结构

- Create: `src/ai/cuda_pipeline.py` — 进程间消息、远程 evaluator、worker 入口、GPU broker 和流水线生命周期。
- Modify: `src/ai/trainer.py` — CUDA 路径接入流水线、逐局提交、状态指标、日志、暂停和 OOM 处理。
- Modify: `src/ai/config.py` — 保持配置格式，明确 CUDA 模式下 worker 也生效；不新增 checkpoint 字段。
- Modify: `src/ai/cli.py` — 帮助文本反映 CUDA worker 语义。
- Delete: `src/ai/batched_self_play.py` — 新流水线验收后移除单进程同步调度器。
- Create: `tests/ai/test_cuda_pipeline.py` — IPC、合批、路由、真实 spawn PID、逐局交付和清理测试。
- Modify: `tests/ai/test_trainer.py` — Trainer CUDA 流水线、暂停、追加、失败和状态测试。
- Delete: `tests/ai/test_batched_self_play.py` — 被新流水线行为测试替代。
- Modify: `docs/XiangQi-AI-Colab-GPU-Training.ipynb` — 动态 worker 配置和前台训练/恢复。
- Modify: `tests/test_colab_gpu_notebook.py` — Notebook 静态与受控执行契约。
- Modify: `docs/AI-TRAINING-GUIDE.md` — 参数、前台运行、指标和现实性能限制。
- Modify: `README.md` — CUDA 命令示例和简短语义说明。

### Task 1: 定义可测试的 IPC 协议和远程 evaluator

**Files:**
- Create: `src/ai/cuda_pipeline.py`
- Create: `tests/ai/test_cuda_pipeline.py`

- [ ] **Step 1: 写请求路由的失败测试**

```python
def test_remote_evaluator_routes_response_by_worker_and_request_id():
    requests = queue.Queue()
    responses = queue.Queue()
    evaluator = RemoteEvaluator(3, requests, responses)
    state = SearchState(Board.standard(), Color.RED)
    expected = np.zeros(ACTION_SIZE, dtype=np.float32)

    responses.put(InferenceResponse(3, 1, expected, 0.25))
    policy, value = evaluator.evaluate(state)

    request = requests.get_nowait()
    assert (request.worker_id, request.request_id, request.state) == (3, 1, state)
    assert np.array_equal(policy, expected)
    assert value == 0.25
```

- [ ] **Step 2: 运行测试并确认 RED**

Run: `.venv/bin/python -m pytest tests/ai/test_cuda_pipeline.py::test_remote_evaluator_routes_response_by_worker_and_request_id -v`

Expected: FAIL because `ai.cuda_pipeline` does not exist.

- [ ] **Step 3: 实现最小消息协议和 evaluator**

在 `src/ai/cuda_pipeline.py` 增加冻结 dataclass：

```python
@dataclass(frozen=True, slots=True)
class InferenceRequest:
    worker_id: int
    request_id: int
    state: SearchState

@dataclass(frozen=True, slots=True)
class InferenceResponse:
    worker_id: int
    request_id: int
    policy: NDArray[np.float32]
    value: float

class RemoteEvaluator:
    def __init__(self, worker_id, request_queue, response_queue):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.next_request_id = 1

    def evaluate(self, state: SearchState):
        request_id = self.next_request_id
        self.next_request_id += 1
        self.request_queue.put(InferenceRequest(self.worker_id, request_id, state))
        response = self.response_queue.get()
        if (response.worker_id, response.request_id) != (self.worker_id, request_id):
            raise RuntimeError("推理响应与请求不匹配")
        return response.policy, response.value
```

队列类型用小型 `Protocol` 表达，以便单元测试使用标准 `queue.Queue`、生产使用
`multiprocessing.Queue`。

- [ ] **Step 4: 运行聚焦测试并确认 GREEN**

Run: `.venv/bin/python -m pytest tests/ai/test_cuda_pipeline.py -v`

Expected: PASS.

- [ ] **Step 5: 提交**

```bash
git add src/ai/cuda_pipeline.py tests/ai/test_cuda_pipeline.py
git commit -m "功能：定义 CUDA 自我对弈通信协议"
```

### Task 2: 实现 GPU 推理 broker 的跨 worker 合批和精确路由

**Files:**
- Modify: `src/ai/cuda_pipeline.py`
- Modify: `tests/ai/test_cuda_pipeline.py`

- [ ] **Step 1: 写合批与不串线的失败测试**

```python
def test_broker_batches_available_requests_and_routes_each_response():
    evaluator = RecordingBatchEvaluator()
    requests = queue.Queue()
    responses = {1: queue.Queue(), 2: queue.Queue()}
    requests.put(InferenceRequest(1, 7, STATE_RED))
    requests.put(InferenceRequest(2, 4, STATE_BLACK))
    broker = CudaInferenceBroker(evaluator, requests, responses, max_batch_size=8)

    assert broker.serve_one_batch(first_request_timeout=0.01) == 2

    assert evaluator.batch_sizes == [2]
    assert responses[1].get_nowait().request_id == 7
    assert responses[2].get_nowait().request_id == 4
    assert broker.inference_requests == 2
    assert broker.max_inference_batch_size == 2
```

另加测试：没有请求时返回 0；非法 evaluator 输出数量时报错；batch 上限为 1 时不多取。

- [ ] **Step 2: 运行并确认 RED**

Run: `.venv/bin/python -m pytest tests/ai/test_cuda_pipeline.py -k broker -v`

Expected: FAIL because `CudaInferenceBroker` is missing.

- [ ] **Step 3: 实现 broker**

`serve_one_batch()` 阻塞取得第一个请求后，以 `get_nowait()` 收集至
`max_batch_size`；调用一次 `evaluate_many()`，验证 `(N, ACTION_SIZE)` 和 `(N,)`
输出，然后按原请求 ID 写入各 worker 的专用响应队列。维护
`last_inference_batch_size`、`max_inference_batch_size` 和 `inference_requests`。

- [ ] **Step 4: 运行并确认 GREEN**

Run: `.venv/bin/python -m pytest tests/ai/test_cuda_pipeline.py -k broker -v`

Expected: all selected tests PASS.

- [ ] **Step 5: 提交**

```bash
git add src/ai/cuda_pipeline.py tests/ai/test_cuda_pipeline.py
git commit -m "功能：实现单 GPU 推理请求合批"
```

### Task 3: 实现真实 spawn worker 和逐局完成事件

**Files:**
- Modify: `src/ai/cuda_pipeline.py`
- Modify: `tests/ai/test_cuda_pipeline.py`

- [ ] **Step 1: 写真实进程与逐局交付的失败测试**

用顶层、可 pickle 的极小 `GameFactory`/搜索替身运行两个 worker，断言：

```python
with CudaSelfPlayPipeline(..., worker_count=2, max_active_games=2) as pipeline:
    completed = list(pipeline.generate(game_numbers=(1, 2)))

assert {item.game_number for item in completed} == {1, 2}
assert len({item.pid for item in completed}) == 2
assert all(item.pid != os.getpid() for item in completed)
assert pipeline.active_games == 0
assert not any(process.is_alive() for process in pipeline.processes)
```

再增加顺序测试：第 2 局被阻塞时，第 1 局完成事件已经从迭代器 yield，而不是等待
两局组成列表。

- [ ] **Step 2: 运行并确认 RED**

Run: `.venv/bin/python -m pytest tests/ai/test_cuda_pipeline.py -k 'spawn or yields_each' -v`

Expected: FAIL because pipeline and worker entry do not exist.

- [ ] **Step 3: 实现 worker、任务和完成协议**

增加 `SelfPlayTask`、`GameCompleted`、`WorkerFailed` 消息。worker 入口创建
`RemoteEvaluator`、`MCTS` 并调用现有 `play_game`；每个 worker 一次只领取一局，
完成后立即发送事件再领取下一局。Pipeline 使用 spawn context 创建每个 worker 的
响应队列、共享请求/结果/任务队列，并在主循环中交替：

1. broker 处理已有推理请求；
2. 读取完成/失败事件；
3. 对空闲 worker 补任务；
4. yield 每个成功结果。

关闭协议先发送 sentinel、join 有界等待，再 terminate 仍存活的精确子进程。

- [ ] **Step 4: 运行并确认 GREEN**

Run: `.venv/bin/python -m pytest tests/ai/test_cuda_pipeline.py -v`

Expected: all tests PASS and pytest exits without hanging processes.

- [ ] **Step 5: 提交**

```bash
git add src/ai/cuda_pipeline.py tests/ai/test_cuda_pipeline.py
git commit -m "功能：并行生成 CUDA 自我对弈棋局"
```

### Task 4: 接入 Trainer、逐局持久化和实时指标

**Files:**
- Modify: `src/ai/trainer.py`
- Modify: `tests/ai/test_trainer.py`
- Delete: `src/ai/batched_self_play.py`
- Delete: `tests/ai/test_batched_self_play.py`

- [ ] **Step 1: 写 Trainer 失败测试**

覆盖四个独立行为：

1. CUDA 模式的 `worker_count == min(self_play_workers, parallel_games)` 且状态同时报告请求值和有效值；
2. fake pipeline 先 yield 一局、第二局尚未完成时，Replay 和 status 已经为 1；
3. 每完成一局输出 flush 进度日志并更新 `active_games`、batch 和请求计数；
4. `game_factory is not None` 的 CUDA 测试路径仍能使用可控替身，不要求真实 CUDA。

关键断言示例：

```python
first_commit_seen = False
def after_first_yield():
    nonlocal first_commit_seen
    first_commit_seen = trainer.replay.total_games == 1

trainer.run()
assert first_commit_seen
assert "self_play_workers_requested=4" in status.message
assert "parallel_games_effective=4" in status.message
assert "inference_requests=" in status.message
```

- [ ] **Step 2: 运行并确认 RED**

Run: `.venv/bin/python -m pytest tests/ai/test_trainer.py -k 'cuda or inference or progress' -v`

Expected: old assertions about effective worker 1 fail and pipeline behavior is absent.

- [ ] **Step 3: 最小接入流水线**

移除 `BatchedSelfPlay` import 和 `_generate_cuda_games()`。CUDA 运行创建一个长期存活
的 `CudaSelfPlayPipeline`，有效 worker/活动棋局为
`min(config.self_play_workers, config.parallel_games)`。Trainer 消费迭代器，每个
`GameCompleted` 立即 `_commit_game()`、checkpoint 判定和暂停判定，并从 pipeline
同步指标写 status。

进度日志使用模块 LOGGER，格式固定为：

```text
self-play completed=<n>/<target> plies=<plies> termination=<kind> active=<n> last_batch=<n> max_batch=<n> requests=<n>
```

完成接入后删除旧同步调度器及其测试，避免维护两套 CUDA 生产路径。

- [ ] **Step 4: 运行并确认 GREEN**

Run: `.venv/bin/python -m pytest tests/ai/test_trainer.py tests/ai/test_cuda_pipeline.py -v`

Expected: all selected tests PASS.

- [ ] **Step 5: 提交**

```bash
git add src/ai/trainer.py src/ai/cuda_pipeline.py tests/ai/test_trainer.py tests/ai/test_cuda_pipeline.py
git add -u src/ai/batched_self_play.py tests/ai/test_batched_self_play.py
git commit -m "功能：接入 CUDA 多进程自我对弈流水线"
```

### Task 5: 暂停、追加、重试、OOM 和生命周期安全

**Files:**
- Modify: `src/ai/cuda_pipeline.py`
- Modify: `src/ai/trainer.py`
- Modify: `tests/ai/test_cuda_pipeline.py`
- Modify: `tests/ai/test_trainer.py`

- [ ] **Step 1: 写异常路径失败测试**

分别测试：

- 一局失败后用同一 game number/seed 重试，超过 `game_retry_limit` 抛出包含 traceback 的错误；
- 暂停后不补新任务，但在途棋局完成、逐局提交、保存 checkpoint 并最终 `paused`；
- 运行中 extend 后继续补充新游戏编号；
- broker 抛一次 `torch.OutOfMemoryError` 后并行上限按候选减半并继续；
- 主循环异常和 `KeyboardInterrupt` 都关闭 worker，且已提交 Replay 保留。

- [ ] **Step 2: 运行并确认 RED**

Run: `.venv/bin/python -m pytest tests/ai/test_cuda_pipeline.py tests/ai/test_trainer.py -k 'retry or pause or extend or oom or interrupt or cleanup' -v`

Expected: one or more new tests FAIL for missing lifecycle behavior.

- [ ] **Step 3: 实现安全控制**

Pipeline 增加 `stop_refilling()` 和有效上限降级；Trainer 捕获 `KeyboardInterrupt` 时
先请求不再补任务、关闭流水线、保存 checkpoint、写 `paused` 后返回。普通异常写
`failed` 并重新抛出。OOM 只减少后续活动任务/推理 batch 上限，调用
`torch.cuda.empty_cache()`，不得删除或回滚 Replay。

- [ ] **Step 4: 运行并确认 GREEN**

Run: `.venv/bin/python -m pytest tests/ai/test_cuda_pipeline.py tests/ai/test_trainer.py -v`

Expected: all selected tests PASS without leaked processes.

- [ ] **Step 5: 提交**

```bash
git add src/ai/cuda_pipeline.py src/ai/trainer.py tests/ai/test_cuda_pipeline.py tests/ai/test_trainer.py
git commit -m "修复：完善 CUDA 流水线暂停与异常恢复"
```

### Task 6: 更新 CLI 语义和训练文档

**Files:**
- Modify: `src/ai/cli.py`
- Modify: `tests/ai/test_cli.py`
- Modify: `README.md`
- Modify: `docs/AI-TRAINING-GUIDE.md`

- [ ] **Step 1: 写帮助文本失败测试**

```python
def test_train_help_explains_cuda_worker_and_parallel_game_limits(capsys):
    with pytest.raises(SystemExit):
        main(["train", "--help"])
    output = capsys.readouterr().out
    assert "CPU 生产进程" in output
    assert "最多在途棋局" in output
```

- [ ] **Step 2: 运行并确认 RED**

Run: `.venv/bin/python -m pytest tests/ai/test_cli.py -k help -v`

Expected: FAIL because current help does not describe the new CUDA semantics.

- [ ] **Step 3: 更新帮助和文档**

CLI 明确 `--self-play-workers` 在 CPU/CUDA 都是 CPU 生产进程数，
`--parallel-games` 是 CUDA 最多在途局数。README 示例增加
`--self-play-workers 4 --parallel-games 16`。训练指南说明指标、逐局 checkpoint
窗口、前台 Colab、T4 CPU 限制和调参顺序：先按 CPU 核数调 worker，再观察 batch，
最后才增加并行局数。

- [ ] **Step 4: 运行并确认 GREEN**

Run: `.venv/bin/python -m pytest tests/ai/test_cli.py -v`

Expected: all tests PASS.

- [ ] **Step 5: 提交**

```bash
git add src/ai/cli.py tests/ai/test_cli.py README.md docs/AI-TRAINING-GUIDE.md
git commit -m "文档：说明 CUDA 多进程训练参数"
```

### Task 7: 将 Colab 正式训练改为前台运行

**Files:**
- Modify: `docs/XiangQi-AI-Colab-GPU-Training.ipynb`
- Modify: `tests/test_colab_gpu_notebook.py`

- [ ] **Step 1: 修改 Notebook 契约测试并确认旧实现失败**

测试要求：

```python
assert "SELF_PLAY_WORKERS = max(1, min(8, os.cpu_count() or 1))" in source
assert '"--self-play-workers"' in training_command_source
assert "run_foreground_training" in source
assert "subprocess.Popen" not in formal_training_cell
assert "subprocess.run" in formal_training_cell
assert "KeyboardInterrupt" in foreground_helper
```

同时删除“正式训练必须包含后台 Popen”的旧断言，保留锁、状态、暂停、追加、恢复和
Drive 持久化安全契约。

- [ ] **Step 2: 运行并确认 RED**

Run: `.venv/bin/python -m pytest tests/test_colab_gpu_notebook.py -k 'workflow or foreground or worker' -v`

Expected: FAIL because Notebook still launches a background process.

- [ ] **Step 3: 修改 Notebook**

配置单元打印 CPU 数、requested/effective worker 和并行局数。正式训练及恢复单元用
前台 `subprocess.run(training_command(...), check=False)`，stdout/stderr 通过 Python
实时 tee 到输出和 Drive 日志；保留外置单运行锁，并用 `finally` 只释放本会话 token。
用户停止单元格时提示 checkpoint/status 结果，不声称 Colab runtime 回收时能执行
清理。清空所有执行计数和 outputs 后保存。

- [ ] **Step 4: 验证 Notebook**

Run: `.venv/bin/python -m pytest tests/test_colab_gpu_notebook.py -v`

Expected: all tests PASS.

Run: `.venv/bin/python -m json.tool docs/XiangQi-AI-Colab-GPU-Training.ipynb >/dev/null`

Expected: exit 0.

- [ ] **Step 5: 提交**

```bash
git add docs/XiangQi-AI-Colab-GPU-Training.ipynb tests/test_colab_gpu_notebook.py
git commit -m "功能：改进 Colab 前台 CUDA 训练"
```

### Task 8: 全量回归和交付验证

**Files:**
- Verify: `src/ai/`
- Verify: `tests/`
- Verify: `docs/XiangQi-AI-Colab-GPU-Training.ipynb`

- [ ] **Step 1: 运行 AI 与 Notebook 聚焦测试**

Run: `.venv/bin/python -m pytest tests/ai tests/test_colab_gpu_notebook.py tests/test_count_training_results.py -q`

Expected: 0 failures.

- [ ] **Step 2: 运行完整测试套件**

Run: `.venv/bin/python -m pytest -q`

Expected: 0 failures; CUDA-only tests may skip when no NVIDIA GPU exists.

- [ ] **Step 3: 运行静态验证**

Run: `.venv/bin/python -m ruff check .`

Expected: exit 0.

Run: `.venv/bin/python -m ruff format --check .`

Expected: exit 0.

Run: `.venv/bin/python -m compileall -q src tests`

Expected: exit 0.

Run: `git diff --check`

Expected: exit 0.

- [ ] **Step 4: 核对需求清单**

确认：多 PID 证据、单 GPU owner、跨 worker batch、逐局 Replay/status、暂停/追加/
恢复、512 完整回合、合法走子、10,000 局默认、前台 Colab、Drive 持久化均有源码
路径和测试证据；本机未执行真实 CUDA 时明确记录该限制。

- [ ] **Step 5: 如有最终机械修正则提交**

```bash
git add src tests README.md docs
git commit -m "测试：验证 CUDA 自我对弈流水线"
```

仅在全量验证产生必要修正时创建该提交；没有修正时不创建空提交。

## 后续部署（需单独获得上传授权）

本计划完成只保证本地代码和 Notebook。上传服务器、推送 GitHub、覆盖 Google Drive
中的既有 Notebook/源码 ZIP 都是外部状态修改，必须在用户明确要求后执行，并分别
回读 hash/commit/Notebook 关键单元验证。
