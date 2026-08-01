# AI 训练使用手册设计

## 目标

新增独立中文手册 `docs/AI-TRAINING-GUIDE.md`，让第一次使用项目的人能够完成依赖安装、CPU/CUDA 训练、暂停、追加、断点续传、状态查询和模型文件确认，无需阅读 AI 源码。

README 保留当前快速说明，并增加指向完整手册的链接。现有桌面程序、`src/xiangqi`、UI、controller 和 API 均不修改。

## 读者

手册面向两类使用者：

- 希望先用最小配置验证训练链路的普通用户；
- 希望设置 CPU 并行度、CUDA 设备、网络规模和训练目标的进阶用户。

默认读者能够使用终端，但不要求了解 AlphaZero、PyTorch 或项目内部架构。

## 内容结构

正式手册按实际操作顺序组织：

1. 训练系统与现有桌面程序的边界。
2. Python 环境和 `.[ai,dev]` 依赖安装。
3. 一局最小 CPU 试跑及结果检查。
4. 默认 10,000 局和 512 完整回合（1024 ply）的含义。
5. 正式 CPU 训练、PyTorch 线程和自我对弈进程设置。
6. CUDA 可用性检测、`cuda`/`cuda:N` 用法与不可用错误。
7. `train`、`pause`、`resume`、`extend`、`status` 完整工作流。
8. `AI-runs/<name>/` 中 Replay、双槽 checkpoint、状态文件和 `final_model.pt` 的作用。
9. 使用 `TrainingConfig` 从 Python 修改局数、回合、网络、MCTS、batch、重试和保存间隔。
10. 常见错误、恢复原则和避免覆盖训练目录的注意事项。
11. 推荐的最小试跑、CPU 正式训练和单 GPU 配置模板。

## 命令准确性

所有命令以当前 `python -m ai --help`、各子命令 `--help` 和源码配置字段为准。手册不记录尚不存在的参数，不声称当前无 CUDA 的机器已经完成真实 GPU 验证。

最小试跑使用独立运行目录和极小网络，避免用户误启动默认 10,000 局：

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

手册会说明该命令只是验证链路，不能用来评估棋力。

## 持久化和恢复说明

手册明确区分：

- `checkpoint-a.pt` / `checkpoint-b.pt`：用于完整续训；
- `final_model.pt`：当前累计目标完成后的纯模型权重；
- `replay/`：完整棋局训练样本；
- `status.json`：可读训练状态。

暂停必须等待安全点；`extend` 修改累计目标而不是立即新增一次独立任务；已完成或暂停的运行追加后需要执行 `resume`。同一运行目录存在有效 checkpoint 时再次执行 `train` 会安全续传。

## 错误处理和安全边界

手册覆盖以下情况：

- 显式 CUDA 不可用时直接报错；
- checkpoint 或 Replay 损坏时不得删除文件后强行重开；
- 非空目录没有有效 checkpoint 时程序拒绝覆盖；
- 可以复制整个运行目录做备份；
- 不手工编辑 manifest、checkpoint、migration sidecar 或状态文件；
- 模型文件使用 `torch.load(..., weights_only=True)` 加载。

## 验收标准

- 新用户可以只阅读手册完成最小 CPU 试跑。
- 文档包含 CPU 多线程、多进程和 CUDA 命令。
- 文档包含暂停、恢复、追加和断点续传的可复制命令。
- 文档解释默认 10,000 局、512 完整回合和累计目标语义。
- 文档展示 Python API 配置示例，字段与当前 `TrainingConfig` 一致。
- 文档解释所有持久化文件及安全恢复原则。
- README 链接到完整手册。
- 文档中的 CLI 命令通过当前帮助输出或短程 smoke 验证。
- 不修改或接入现有桌面程序，也不触碰用户的 `.vscode/`。
