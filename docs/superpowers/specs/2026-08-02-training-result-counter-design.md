# 训练棋局胜负统计脚本设计

## 目标

新增一个可直接在服务器 AI 目录运行的独立 Python 文件
`src/ai/count_training_results.py`。脚本只读检查指定 AlphaZero 训练目录，统计已经完整写入
Replay 的红胜、黑胜与和棋局数，同时显示训练进度和无法解析的数据文件。

## 使用方式

默认统计服务器当前主训练目录：

```bash
cd /XiangQi-AI/src/ai
../../.venv/bin/python count_training_results.py
```

也允许显式指定其他训练目录：

```bash
../../.venv/bin/python count_training_results.py \
  --run-dir AI-runs/cpu-main
```

`--run-dir` 的默认值为脚本同级的 `AI-runs/cpu-main`。脚本不依赖当前工作目录来
解析默认路径，而是以脚本所在的 AI 目录为基准。

## 数据来源与判定

脚本读取以下文件，不写入、不锁定也不修改任何训练数据：

- `<run-dir>/status.json`：当前阶段、已完成局数、目标局数和训练步数；
- `<run-dir>/replay/manifest.json`：累计提交局数以及当前保留的棋局 ID；
- `<run-dir>/replay/games/<12 位棋局 ID>.npz`：每局的 `values`、`sides` 和
  `plies` 数组。

每个样本的 `sides` 中 `0` 表示红方、`1` 表示黑方；`values` 是该样本行棋方
视角的最终价值。脚本把价值统一换算为红方视角：全部为 `1` 判定红胜，全部为
`-1` 判定黑胜，全部为 `0` 判定和棋。数值不一致、字段缺失或文件损坏的棋局计入
“异常”，并在明细中列出棋局 ID 和原因。

统计口径为 manifest 当前列出的 Replay 棋局。若 Replay 容量未来发生淘汰，脚本会
同时输出 manifest 的历史累计局数和当前可分类局数，避免把保留窗口误报为全部历史
胜负。

## 输出与错误处理

正常输出为简洁中文文本，至少包括：统计时间、训练目录、训练阶段、完成进度、历史
累计局数、当前可分类局数、红胜、黑胜、和棋和异常数。训练仍在运行时，这只是执行
瞬间的只读快照。

以下情况以非零退出码结束并输出明确错误：训练目录不存在、manifest 不存在或不是
合法 JSON、manifest 结构无效，以及缺少运行所需的 NumPy。单局 NPZ 解析失败不会
中止整个统计，而是计入异常，保证其他完整棋局仍能统计。

## 测试与部署

测试先构造最小临时训练目录和 NPZ 数据，覆盖红胜、黑胜、和棋、异常文件、路径解析
及 CLI 输出。测试先失败，再实现最小代码使其通过。

本地测试通过后，只上传 `src/ai/count_training_results.py` 到服务器
`/XiangQi-AI/src/ai/count_training_results.py`，不上传其他工作区文件、不重启训练进程。
随后使用服务器现有 `.venv/bin/python` 运行脚本，并将现场统计结果与 `status.json`
和 manifest 的局数进行一致性核对。
