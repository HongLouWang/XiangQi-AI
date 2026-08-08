# CUDA 多进程自我对弈流水线设计

## 目标

将当前“单个 Python 进程同步推进多局”的 CUDA 自我对弈改为“多个 CPU
生产进程 + 主进程单 GPU 批量推理”的流水线。解决 Colab 实测中训练进程只占用
约一个 CPU 核、T4 GPU 利用率约 7%、第一组 16 局结束前状态长期为 0 的问题。

本次修改同时让 Colab 正式训练在前台单元格运行，避免仅依赖后台子进程；Replay、
checkpoint、最终模型和控制文件仍持久化在 Google Drive。

## 已确认范围

- 中国象棋规则、合法走子、将死判定、512 个完整回合（1024 ply）上限不变。
- 默认目标仍为 10,000 局，并允许通过 CLI 或 Notebook 修改、追加。
- CPU 训练继续支持 `--self-play-workers` 多进程。
- CUDA 只由训练主进程持有；子进程不得创建 CUDA context 或复制 GPU 模型。
- 每完成一局立即原子写入 Replay 并更新 `status.json`，不等待整组棋局结束。
- 继续支持暂停、追加目标和 checkpoint 断点续传。
- 不接入现有 PySide6/FastAPI 对弈程序。

## 方案比较

### 方案 A：多进程 CPU 生产者 + 主进程 GPU 推理代理（采用）

每个 CPU worker 独立持有棋局、MCTS 树和规则状态。MCTS 需要神经网络评估时，
worker 把不可变的 `SearchState` 发给主进程并等待结果。主进程把多个 worker 的请求
合成 batch，在唯一 CUDA 模型上推理，再把对应结果发回。

优点是能绕过 Python GIL、保持单 CUDA context，并可逐局交付结果。代价是每次叶子
评估都有进程间通信；实际加速受 Colab CPU 核数限制。

### 方案 B：主进程线程池推进规则

实现较简单，但 Python 规则和 MCTS 计算受 GIL 限制，不能解决实测中的单核瓶颈，
不采用。

### 方案 C：将规则和树搜索改写为原生扩展

长期性能上限更高，但范围远大于本次修复，还会提高构建、Colab 安装和跨平台维护
成本，不纳入本次工作。

## 架构

新增独立的 CUDA 自我对弈流水线模块，职责分成三部分：

1. `CudaSelfPlayWorker`：在 spawn 子进程内执行完整棋局和 MCTS；通过远程 evaluator
   请求推理；只使用 CPU。
2. `CudaInferenceBroker`：运行在训练主进程，收集带有 worker/request ID 的评估
   请求；达到批量上限或短暂合并窗口后调用 `TorchEvaluator.evaluate_many`；把结果
   精确路由回请求者。
3. `CudaSelfPlayPipeline`：维护活动任务、失败重试、完成事件、暂停时停止补充新局，
   并以迭代器形式在每局结束时立即向 `Trainer` 交付结果。

主进程仍是 Replay、模型优化器、状态和 checkpoint 的唯一写入者。GPU 推理和模型
训练都在同一主进程串行执行，避免模型被并发读写。

## 参数语义

- `self_play_workers`：CPU 生产进程数。CPU 和 CUDA 模式都生效。
- `parallel_games`：CUDA 流水线最多同时在途的棋局数；有效值为
  `min(parallel_games, self_play_workers)`，因为每个阻塞式 worker 同时推进一局。
- `inference_batch_size`：单次 GPU 推理请求上限，默认沿用有效并行棋局数，不新增
  用户必须配置的 CLI 参数。
- 推理合并窗口采用很短的固定上限，仅用于吸收几乎同时到达的请求；不能长期等待
  满 batch，以免低并发死锁。

Colab 默认根据可见 CPU 数设置 worker：`max(1, min(8, os.cpu_count() or 1))`；
`PARALLEL_GAMES` 继续默认 16。Notebook 明确打印请求值和受 CPU 核数约束后的有效值。
超额进程不能创造 CPU 算力，因此默认不盲目启动 16 个进程。

## 数据流与一致性

1. Trainer 按全局游戏编号生成确定性 seed 并派发任务。
2. worker 使用现有 `play_game`、`MCTS` 和合法走子实现。
3. worker 的 evaluator 发送 `SearchState`，主进程批量推理后返回 logits/value。
4. worker 返回完整 `GameResult` 或结构化错误。
5. Pipeline 收到完成事件后立即 yield；Trainer 立即调用 `_commit_game`。
6. `_commit_game` 原子追加 Replay、更新完成局数和状态，并在样本充足时训练一步。
7. Trainer 再补充一局保持流水线占用，直到达到实时目标或收到暂停请求。

游戏编号和 seed 在重试时保持不变。结果允许乱序完成，但 Replay 提交使用完成顺序；
状态中的 `completed_games` 以已成功持久化的实际局数为准。checkpoint 继续以 Replay
manifest 为恢复依据，因此 Colab 终止后不会把已原子提交的棋局重复计数。

## 暂停、追加和终止

- 每交付一局后读取实时目标和暂停标记。
- 暂停时不再派发新棋局；已经在途的棋局允许完成并逐局提交，随后保存 checkpoint
  并标记 `paused`。
- 追加目标由现有 `extend` 修改状态；流水线下一次补任务时读取新目标。
- worker 异常沿用 `game_retry_limit`。超过限制时停止所有 worker、写 `failed` 状态并
  尝试保存 checkpoint。
- 主进程退出、异常或键盘中断时必须关闭队列并有界等待 worker；超时后终止残留
  worker，不能留下操作同一运行目录的孤儿进程。
- CUDA OOM 时先清空缓存并降低有效在途棋局/批量上限；已完成 Replay 不回滚。

## 状态与日志

CUDA 状态消息至少包含：

- `self_play_workers_requested` / `self_play_workers_effective`；
- `parallel_games_requested` / `parallel_games_effective`；
- `active_games`、`finished_games_in_session`；
- `last_inference_batch_size`、`max_inference_batch_size`；
- `inference_requests`、`oom_downgrades`；
- 最近一次局完成时间或会话启动后的已运行秒数。

每完成一局输出一条可 flush 的进度日志，包含总完成局数、目标、该局 ply/终局原因、
活动棋局数和推理 batch 指标。这样即使第一局很长，也可以通过周期性心跳看到活动
棋局数和累计推理请求，不再把空日志误判为卡死。

## Colab Notebook

正式训练单元格使用 `subprocess.run` 或直接调用 CLI，并保持单元格前台占用。用户可
用 Colab 的停止按钮触发键盘中断；训练器负责关闭流水线并保存安全 checkpoint。

Notebook 保留独立的状态、暂停、追加和恢复单元格。断线后 Drive 中的数据继续存在，
重新连接后先检查状态，再执行恢复。原后台启动函数不再作为推荐正式路径，避免给出
“后台进程能抵抗 Colab runtime 回收”的错误预期。

## 测试与验收

- 单元测试证明多个 spawn worker PID 实际参与 CUDA 自我对弈，而不是只修改状态文本。
- 测试推理请求可跨 worker 合批、结果按 request ID 返回且不会串线。
- 测试一局完成就被 Trainer 提交，另一局未完成时 `completed_games` 已增加。
- 测试暂停停止补任务、在途任务安全收尾、追加目标继续派发。
- 测试 worker 失败重试、超限失败、清理无孤儿进程和 OOM 降级。
- Notebook 静态测试验证前台训练、动态 worker 默认值、Drive 路径以及恢复命令。
- 运行 AI 聚焦测试、Notebook 测试、完整 pytest、Ruff、格式检查和 Python 编译检查。
- 在无 NVIDIA GPU 的本机只声明调度与 CPU 模拟验证；真实 T4/A100 利用率和局/小时
  必须在 Colab 实测后报告。

## 非目标和现实限制

- 不承诺固定加速倍数。T4 Colab 常见 CPU 核数较少，GPU 利用率仍可能受规则计算
  限制；该架构消除人为的单进程限制，但不能超过分配到的 CPU 算力。
- 本次不实现分布式多机训练、多个 GPU、CUDA 图、原生规则扩展或 replay 服务化。
- 不改变模型结构、动作编码、checkpoint 权重格式或合法走子规则。
