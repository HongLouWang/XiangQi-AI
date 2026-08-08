# 中国象棋

一个可在 macOS 上运行的 Python 中国象棋桌面游戏。界面使用 PySide6，
棋盘始终红方在下、黑方在上。默认红黑双方均由玩家操作，当前版本不接入 AI。

## 独立 AlphaZero 训练

`src/ai` 是独立的中国象棋 AlphaZero 风格训练工具，采用 Policy/Value
ResNet、PUCT MCTS 和自我对弈循环。它只复用规则引擎来生成合法着法和判断
将军、将死、困毙；目前**没有接入**桌面 UI、controller 或 HTTP API。训练中的
每一步都从规则引擎返回的合法着法中选择；若在上限前将死，对局立即按将死
结束。

完整的安装、CPU/CUDA 配置、暂停、追加、断点续传和故障排查说明见
[AI 训练使用手册](docs/AI-TRAINING-GUIDE.md)。需要在 Google Colab 使用 NVIDIA
GPU，并把源码、临时文件和全部训练产物持久化到 Google Drive 时，可直接打开
[Colab GPU 训练 Notebook](docs/XiangQi-AI-Colab-GPU-Training.ipynb)。

安装训练和开发依赖：

```bash
python -m pip install -e ".[ai,dev]"
```

默认命令累计训练 10,000 局，每局最多 512 个完整回合。一个完整回合指红方和
黑方各走一步，因此上限是 1024 ply；到达上限且尚未自然终局时按和棋生成训练
标签。10,000 局只是可修改的初始训练规模，不是棋力保证：

```bash
python -m ai train
python -m ai train --games 20000 --full-moves 512 --device cpu \
  --torch-threads 8 --self-play-workers 8
python -m ai train --device cuda:0 \
  --self-play-workers 4 --parallel-games 16
```

`--games` 是该运行目录的累计目标局数，不是每次额外新增的局数。CPU 模式可用
`--torch-threads` 设置每个 PyTorch 进程的计算线程数，用
`--self-play-workers` 设置 CPU 自我对弈生产进程数，在 CPU 和 CUDA 模式都
生效。GPU 模式接受 `cuda` 或 `cuda:N`；CUDA 由多个 CPU 生产进程发送
MCTS 请求，主进程在单张 GPU 上合批推理。`--parallel-games` 是 CUDA 最多在途
棋局数，默认 16，实际数量还受 worker 数和当前工作量限制。显式选择 CUDA 而当前 PyTorch/CUDA 不可用时
会直接报错，不会静默退回 CPU。也可以在 Python 中构造 `TrainingConfig` 修改局数、回合上限、网络
尺寸、MCTS 模拟次数、batch 和持久化间隔等参数。

默认运行目录为 `AI-runs/default`。Replay 数据、状态和模型均会持久化到该目录：

- `replay/`：按完整棋局原子写入的训练样本及 manifest；
- `checkpoint-a.pt`、`checkpoint-b.pt`、`latest.json`：安全原子写入、轮换保存
  的完整断点，包含模型、优化器、训练进度和随机数状态；
- `final_model.pt`：完成当前累计目标后导出的纯模型权重，可用
  `torch.load(..., map_location="cpu", weights_only=True)` 安全加载；
- `status.json`：可供其他进程读取的训练状态。

在已有 checkpoint 的同一目录再次执行 `train` 会自动断点续传，不会清空已有
模型或 Replay。暂停只会在一局完整提交并安全保存之后生效；恢复、追加累计目标
和查看状态的命令如下：

```bash
python -m ai pause --run-dir AI-runs/default
python -m ai resume --run-dir AI-runs/default
python -m ai extend --run-dir AI-runs/default --games 5000
python -m ai status --run-dir AI-runs/default
```

`extend --games 5000` 会在当前累计目标上再增加 5,000 局，保留已完成局数、
Replay、模型和优化器状态；若训练进程仍在运行，它会读取新增目标继续训练，若已
暂停或完成，则随后执行 `resume`。

## 功能

- 标准中国象棋合法性、将军、将死和困毙判断，每一步走棋后立即检查。
- 开局时可选“中国棋规（2020）”或“亚洲棋规（2003）”，包含长将、长杀、
  长捉/长打和重复局面的责任判定。
- 左键选择棋子并显示全部合法落点；上一手起点和落子棋子持续高亮。
- 双方提和与回应、无限次悔棋、终局后悔棋。
- JSON 完整棋谱与中文纵线棋谱的导入、导出和回放，可从回放位置继续。
- 同进程 Python 控制、回调，以及仅限本机的 HTTP/WebSocket 控制接口。

## macOS 安装与启动

需要 Python 3.11 或更高版本。建议在项目目录创建独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m xiangqi
```

默认同时启动桌面窗口与 `127.0.0.1:8765` 上的 API。可以更换端口或完全
关闭 API：

```bash
python -m xiangqi --port 9000
python -m xiangqi --no-api
```

命令行故意不提供 `--host`：服务只能绑定 `127.0.0.1`，不会直接暴露到
局域网或互联网。API 启用时可在
`http://127.0.0.1:8765/docs` 查看交互文档。

## 棋谱

“导出”按扩展名生成 `.json` 或 `.txt`。JSON 保存规则模式、玩家、初始局面、
完整着法和裁决信息，适合无损恢复；文本格式保存“炮二平五”一类中文纵线
记谱。导入会先在临时棋局中完整校验，失败不会替换当前对局。

回放提供首步、前一步、播放/暂停、后一步、末步和速度控制。“从此继续”会
截断游标后的着法并创建新分支。

## Python 控制与 callback

规则引擎和控制器可脱离 UI 使用。坐标以左上角为 `(0, 0)`，横向为 `file`，
纵向为 `rank`：

```python
from xiangqi.controller import GameController
from xiangqi.domain import Color, Coord

controller = GameController.new()
controller.register_callback(
    lambda event: print(event.kind, event.move, event.checkmate, event.stalemate)
)

lease = controller.claim_side(Color.RED, "example-python")
controller.make_move(Coord(0, 6), Coord(0, 5), actor=lease.token)
controller.release_side(Color.RED, "example-python", lease.token)
```

`get_state()` 查询完整状态，`get_legal_moves()` 返回合法着法。程序可以申请
红方、黑方或依次申请双方的控制权；未申请的一方保持人工控制。callback 异常
会被控制器记录，不会撤销已经完成的合法着法。

## HTTP 与 WebSocket

HTTP 查询不需要控制权：

```bash
curl http://127.0.0.1:8765/state
curl http://127.0.0.1:8765/legal-moves
```

写命令包含唯一 `request_id`、调用方 `controller_id` 和当前
`expected_version`。先申请一方控制权并保存响应中的 `token`：

```bash
curl -X POST http://127.0.0.1:8765/control/red/claim \
  -H 'content-type: application/json' \
  -d '{"request_id":"claim-1","controller_id":"client-1","expected_version":0}'
```

随后将 `token` 放入 `/move` 等命令。WebSocket 地址为
`ws://127.0.0.1:8765/ws`，连接后会收到 `ready`，对局变化会产生 `event`；
也可发送带 `"command":"move"` 等字段的同构控制命令。局面版本过期的写入会被
拒绝，多个观察者可以同时连接，但每方同一时刻只能有一个外部控制者。

## 构建 macOS `.app`

在 macOS 虚拟环境中安装 PyInstaller 并构建：

```bash
python -m pip install "pyinstaller>=6.10"
python -m PyInstaller --clean --noconfirm xiangqi.spec
open "dist/中国象棋.app"
```

生成的窗口应用名为“中国象棋”，bundle identifier 为
`com.xiangqi.desktop`，不会打开终端窗口。首次从 Finder 打开未签名的本地
构建时，macOS 可能要求在“系统设置 → 隐私与安全性”中确认。

## 测试

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -v
python -m ruff check .
python -m ruff format --check .
python -m compileall -q src
```
