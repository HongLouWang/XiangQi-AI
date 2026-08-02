# Colab Drive Notebook 首次训练锁修复设计

## 目标

修复 Google Drive 上 `XiangQi-AI-Colab-GPU-Training.ipynb` 的首次正式训练失败问题，并使用本地干净版本整体覆盖 Drive 上的现有 Notebook，同时保持原 Drive 文件 ID 和链接不变。

## 根因

Notebook 在启动训练进程前将 `.colab-training.lock` 建在新的 `RUN_DIR` 中。训练 CLI 会拒绝“非空且没有 checkpoint”的首次训练目录，因此每次后台首次启动都会在生成 `status.json` 前失败。提前运行暂停命令还会在该目录留下 `control.lock` 和 `pause.json`。

## 设计

1. 将 Notebook 自身的跨 Colab 会话锁迁移到 `MyDrive/XiangQi-AI/locks/<RUN_NAME>.lock`，使它不属于训练器的运行目录。
2. 锁的临时 metadata 和陈旧锁回收目录也放在 `locks/` 下，保证锁操作期间 `RUN_DIR` 仍为空。
3. 增加一次性安全恢复单元。仅当以下条件全部成立时，允许删除失败启动留下的已知控制文件：
   - 没有匹配当前 `RUN_DIR` 的存活训练进程；
   - 不存在 `status.json`；
   - 不存在 checkpoint 或最终模型；
   - 旧锁目录只包含预期的 `owner.json`。
4. 状态查询使用非抛出式 subprocess 调用。无论状态命令是否成功，都显示退出码、stderr 和最近日志。
5. 使用本地干净 Notebook 整体覆盖 Drive 文件，清除 Drive 版本中的运行输出和用户临时改单元；通过 Drive `files.update` 保留原文件 ID、位置和共享状态。

## 测试与验收

- 先新增失败测试，证明首次启动前 `RUN_DIR` 不包含 Notebook 锁或锁临时文件。
- 覆盖外置锁的竞争、陈旧锁回收、失败清理和所有权保护。
- 验证安全恢复单元在存在训练产物或存活进程时拒绝清理。
- 验证状态查询在退出码 1 时仍输出 stderr 和日志。
- 所有 Notebook 代码单元可通过 AST 编译，专项测试、Ruff、compileall 和 diff-check 通过。
- 覆盖后从 Drive 重新读取同一文件 ID，确认 MIME 类型、修改时间及修复后的关键代码。

## 非目标

- 不修改 AlphaZero 训练核心、棋规、UI 或 API。
- 不删除任何 checkpoint、Replay、最终模型或有效状态文件。
- 不创建第二份 Drive Notebook。
