# 中国象棋

一个可在 macOS 上运行的 Python 中国象棋桌面游戏。界面使用 PySide6，
棋盘始终红方在下、黑方在上。默认红黑双方均由玩家操作，当前版本不接入 AI。

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
