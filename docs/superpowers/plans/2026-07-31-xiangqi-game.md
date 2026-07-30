# 中国象棋桌面游戏实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个采用 PySide6 图形界面、支持完整中国象棋规则、两套长打裁决、双格式棋谱、回放、无限悔棋及本机程序控制接口的双人游戏。

**Architecture:** 纯 Python 规则引擎生成不可歧义的局面与着法，由 `GameController` 统一承接 UI、同进程程序和网络命令。PySide6 只负责交互和显示，FastAPI 只负责协议适配，JSON/中文棋谱和中国/亚洲棋例分别通过独立策略模块接入。

**Tech Stack:** Python 3.11+、PySide6、FastAPI、Uvicorn、Pydantic 2、pytest、pytest-qt、httpx、Ruff、PyInstaller（macOS `.app`）

---

## 文件结构

```text
pyproject.toml                         项目元数据、依赖与工具配置
README.md                              安装、启动、Python/API 使用说明
xiangqi.spec                           macOS `.app` 的 PyInstaller 构建配置
src/xiangqi/__init__.py                包版本与公共入口
src/xiangqi/__main__.py                `python -m xiangqi` 启动入口
src/xiangqi/domain.py                  棋子、坐标、着法、状态、事件值对象
src/xiangqi/board.py                   棋盘、标准开局、序列化和局面哈希
src/xiangqi/rules.py                   基础走法、攻击、合法性和终局判断
src/xiangqi/adjudication.py            中国/亚洲长打分类和裁决策略
src/xiangqi/record.py                  JSON 棋谱模型、读写和原子导出
src/xiangqi/notation.py                中文纵线记谱生成、解析和重放
src/xiangqi/controller.py              事务、悔棋、提和、回调和控制权
src/xiangqi/api.py                     FastAPI HTTP/WebSocket 适配
src/xiangqi/ui/board_widget.py         棋盘绘制、点击和高亮
src/xiangqi/ui/dialogs.py              新局与玩家设置对话框
src/xiangqi/ui/main_window.py          主窗口、着法表、操作和回放
src/xiangqi/app.py                     QApplication、API 线程和关闭流程
tests/conftest.py                      共享局面和 Qt fixture
tests/test_board.py                    棋盘与值对象测试
tests/test_piece_rules.py              各棋子伪合法走法测试
tests/test_legality_and_endings.py      自陷将军、将死、困毙测试
tests/test_adjudication.py             中国/亚洲棋例测试
tests/test_record.py                   JSON 棋谱与原子导入导出测试
tests/test_notation.py                 中文记谱生成、解析和重放测试
tests/test_controller.py               悔棋、提和、回调和控制权测试
tests/test_api.py                      HTTP/WebSocket 与并发版本测试
tests/ui/test_board_widget.py          选择、落点和持续高亮测试
tests/ui/test_main_window.py           操作、回放和关闭测试
```

### Task 1: 项目骨架与领域值对象

**Files:**
- Create: `pyproject.toml`
- Create: `src/xiangqi/__init__.py`
- Create: `src/xiangqi/domain.py`
- Create: `tests/test_board.py`

- [ ] **Step 1: 写领域对象失败测试**

```python
from xiangqi.domain import Color, Coord, Move, Piece, PieceType


def test_coord_rejects_outside_board() -> None:
    for file, rank in [(-1, 0), (9, 0), (0, -1), (0, 10)]:
        try:
            Coord(file, rank)
        except ValueError:
            pass
        else:
            raise AssertionError("越界坐标必须被拒绝")


def test_move_is_serializable_without_ui_types() -> None:
    move = Move(
        start=Coord(1, 7),
        end=Coord(1, 0),
        piece=Piece(Color.RED, PieceType.CANNON),
    )
    assert move.to_dict()["start"] == [1, 7]
```

- [ ] **Step 2: 运行测试并确认因包不存在而失败**

Run: `python -m pytest tests/test_board.py -v`
Expected: FAIL，包含 `ModuleNotFoundError: No module named 'xiangqi'`

- [ ] **Step 3: 创建配置和最小领域模型**

`pyproject.toml` 定义 src-layout 包、Python `>=3.11`，运行依赖
`PySide6>=6.7`、`fastapi>=0.115`、`uvicorn>=0.30`、`pydantic>=2.8`，
开发依赖 `pytest>=8.2`、`pytest-qt>=4.4`、`httpx>=0.27`、`ruff>=0.6`、
`pyinstaller>=6.10`。
`domain.py` 使用冻结 dataclass 和字符串枚举定义：

```python
class Color(StrEnum):
    RED = "red"
    BLACK = "black"

    @property
    def opponent(self) -> "Color":
        return Color.BLACK if self is Color.RED else Color.RED


class PieceType(StrEnum):
    GENERAL = "general"
    ADVISOR = "advisor"
    ELEPHANT = "elephant"
    HORSE = "horse"
    ROOK = "rook"
    CANNON = "cannon"
    PAWN = "pawn"


@dataclass(frozen=True, slots=True)
class Coord:
    file: int
    rank: int

    def __post_init__(self) -> None:
        if not 0 <= self.file < 9 or not 0 <= self.rank < 10:
            raise ValueError(f"坐标越界: ({self.file}, {self.rank})")


@dataclass(frozen=True, slots=True)
class Piece:
    color: Color
    kind: PieceType


@dataclass(frozen=True, slots=True)
class Move:
    start: Coord
    end: Coord
    piece: Piece
    captured: Piece | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "start": [self.start.file, self.start.rank],
            "end": [self.end.file, self.end.rank],
            "piece": {"color": self.piece.color, "kind": self.piece.kind},
            "captured": None
            if self.captured is None
            else {
                "color": self.captured.color,
                "kind": self.captured.kind,
            },
        }
```

- [ ] **Step 4: 运行测试并确认通过**

Run: `python -m pytest tests/test_board.py -v`
Expected: `2 passed`

- [ ] **Step 5: 提交**

```bash
git add pyproject.toml src/xiangqi/__init__.py src/xiangqi/domain.py tests/test_board.py
git commit -m "构建：初始化象棋项目与领域模型"
```

### Task 2: 棋盘与标准初始局面

**Files:**
- Create: `src/xiangqi/board.py`
- Modify: `tests/test_board.py`

- [ ] **Step 1: 写棋盘失败测试**

```python
from xiangqi.board import Board
from xiangqi.domain import Color, Coord, PieceType


def test_standard_board_has_32_pieces_and_generals() -> None:
    board = Board.standard()
    assert len(board.pieces) == 32
    assert board.at(Coord(4, 9)).kind is PieceType.GENERAL
    assert board.at(Coord(4, 9)).color is Color.RED
    assert board.at(Coord(4, 0)).color is Color.BLACK


def test_apply_returns_new_board_and_preserves_original() -> None:
    board = Board.standard()
    next_board = board.move_unchecked(Coord(0, 6), Coord(0, 5))
    assert board.at(Coord(0, 6)) is not None
    assert next_board.at(Coord(0, 6)) is None
    assert next_board.at(Coord(0, 5)).kind is PieceType.PAWN
```

- [ ] **Step 2: 运行新增测试并确认失败**

Run: `python -m pytest tests/test_board.py -v`
Expected: FAIL，包含 `No module named 'xiangqi.board'`

- [ ] **Step 3: 实现不可变棋盘**

`Board` 将 `Mapping[Coord, Piece]` 复制到只读映射，提供 `standard()`、`empty()`、
`at()`、`place()`、`remove()`、`move_unchecked()`、`to_fen()`、`from_fen()` 和
确定性的 `position_key(side_to_move)`。标准开局坐标固定黑方 rank 0、红方
rank 9，红兵 rank 6、黑卒 rank 3。

- [ ] **Step 4: 运行棋盘测试**

Run: `python -m pytest tests/test_board.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/xiangqi/board.py tests/test_board.py
git commit -m "功能：实现象棋棋盘与标准开局"
```

### Task 3: 各棋子的伪合法走法

**Files:**
- Create: `src/xiangqi/rules.py`
- Create: `tests/test_piece_rules.py`

- [ ] **Step 1: 写覆盖七类棋子的参数化失败测试**

测试用 `Board.empty().place(...)` 构造最小局面，逐项断言：

```python
@pytest.mark.parametrize(
    ("piece", "start", "legal", "illegal"),
    [
        (Piece(Color.RED, PieceType.ROOK), Coord(4, 5), Coord(4, 1), Coord(5, 4)),
        (Piece(Color.RED, PieceType.HORSE), Coord(4, 5), Coord(6, 4), Coord(4, 3)),
        (Piece(Color.RED, PieceType.ELEPHANT), Coord(4, 9), Coord(2, 7), Coord(6, 5)),
        (Piece(Color.RED, PieceType.ADVISOR), Coord(4, 9), Coord(3, 8), Coord(2, 7)),
        (Piece(Color.RED, PieceType.GENERAL), Coord(4, 9), Coord(4, 8), Coord(4, 7)),
        (Piece(Color.RED, PieceType.CANNON), Coord(1, 7), Coord(1, 4), Coord(2, 6)),
        (Piece(Color.RED, PieceType.PAWN), Coord(4, 6), Coord(4, 5), Coord(3, 6)),
    ],
)
def test_piece_pseudo_moves(piece, start, legal, illegal) -> None:
    board = Board.empty().place(start, piece)
    moves = set(pseudo_legal_destinations(board, start))
    assert legal in moves
    assert illegal not in moves
```

另写独立测试覆盖蹩马腿、塞象眼、炮隔一子吃子、炮不能隔子走空位、过河兵可
横走和未过河兵不可横走。

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_piece_rules.py -v`
Expected: FAIL，包含缺少 `pseudo_legal_destinations`

- [ ] **Step 3: 实现伪合法走法生成器**

在 `rules.py` 为每种棋子建立一个私有生成器，公共函数只分派并过滤越界、己方
占位。车和炮沿四条射线扫描；马检查马腿；象检查象眼和本方半场；将、士检查
本方九宫；兵按颜色和是否过河选择方向。

- [ ] **Step 4: 运行走法测试和已有测试**

Run: `python -m pytest tests/test_piece_rules.py tests/test_board.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/xiangqi/rules.py tests/test_piece_rules.py
git commit -m "功能：实现全部棋子的基础走法"
```

### Task 4: 将军、合法走法、将死与困毙

**Files:**
- Modify: `src/xiangqi/rules.py`
- Create: `tests/test_legality_and_endings.py`

- [ ] **Step 1: 写合法性和终局失败测试**

构造可读的 FEN fixture 并断言：

```python
def test_flying_generals_make_exposing_move_illegal() -> None:
    board = Board.from_fen("4k4/9/9/9/4R4/9/9/9/9/4K4 w")
    assert Coord(3, 5) not in legal_destinations(board, Coord(4, 4), Color.RED)


def test_checkmate_is_win_for_attacker() -> None:
    result = evaluate_position(checkmate_board(), Color.BLACK)
    assert result.kind is PositionKind.CHECKMATE
    assert result.winner is Color.RED


def test_stalemate_is_loss_for_side_without_moves() -> None:
    result = evaluate_position(stalemate_board(), Color.BLACK)
    assert result.kind is PositionKind.STALEMATE
    assert result.winner is Color.RED
```

再覆盖攻击检测、自陷将军、吃将不作为普通合法着法、被将方必须应将。

- [ ] **Step 2: 运行并确认失败**

Run: `python -m pytest tests/test_legality_and_endings.py -v`
Expected: FAIL，缺少合法性和终局 API

- [ ] **Step 3: 实现完整合法性与终局**

新增 `is_square_attacked()`、`is_in_check()`、`legal_destinations()`、
`all_legal_moves()` 和 `evaluate_position()`。合法走法先生成伪合法候选，
再模拟落子并排除己方将帅受攻；攻击检测显式处理将帅照面，避免递归调用合法
走法。

- [ ] **Step 4: 运行规则测试**

Run: `python -m pytest tests/test_piece_rules.py tests/test_legality_and_endings.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/xiangqi/rules.py tests/test_legality_and_endings.py
git commit -m "功能：实现将军与终局判断"
```

### Task 5: 中国棋规与亚洲棋规长打裁决

**Files:**
- Create: `src/xiangqi/adjudication.py`
- Create: `tests/test_adjudication.py`

- [ ] **Step 1: 写两套规则的失败测试**

使用固定 FEN 和完整着法序列定义表驱动案例：

```python
@pytest.mark.parametrize("ruleset", [Ruleset.CHINESE_2020, Ruleset.ASIAN_2003])
def test_single_side_perpetual_check_assigns_responsibility(ruleset) -> None:
    result = adjudicate_fixture("single_perpetual_check", ruleset)
    assert result.kind is AdjudicationKind.MUST_CHANGE
    assert result.responsible is Color.RED
    assert result.cycle_start == 0
    assert all(tag is MoveNature.CHECK for tag in result.responsible_natures)


def test_chinese_and_asian_mutual_attack_follow_separate_tables() -> None:
    chinese = adjudicate_fixture("mutual_mixed_attack", Ruleset.CHINESE_2020)
    asian = adjudicate_fixture("mutual_mixed_attack", Ruleset.ASIAN_2003)
    assert chinese.rule_reference.startswith("中国棋规")
    assert asian.rule_reference.startswith("亚洲棋规")
```

fixture 表必须包含单方长将、长杀、长捉车、长捉无根子、帅兵允许长捉、
一将一捉、双方同类禁止着法、双方不同责任和允许着法。

- [ ] **Step 2: 运行并确认失败**

Run: `python -m pytest tests/test_adjudication.py -v`
Expected: FAIL，缺少 `xiangqi.adjudication`

- [ ] **Step 3: 实现着法性质分析和策略**

定义：

```python
class RuleAdjudicator(Protocol):
    def evaluate(self, history: Sequence[PositionFrame]) -> Adjudication: ...


class Chinese2020Adjudicator:
    ruleset = Ruleset.CHINESE_2020


class Asian2003Adjudicator:
    ruleset = Ruleset.ASIAN_2003
```

`PositionFrame` 保存走前/走后 key、着法、攻击映射、被攻击子的根和交换价值。
先识别最短重复循环，再将每方循环着法分类为将、杀、捉、兑、献、拦、跟、
闲，最后由各策略自己的责任表裁决。结果始终带 `cycle_start`、
`responsible`、`move_natures` 和 `rule_reference`。

- [ ] **Step 4: 运行棋例与全部规则测试**

Run: `python -m pytest tests/test_adjudication.py tests/test_legality_and_endings.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/xiangqi/adjudication.py tests/test_adjudication.py
git commit -m "功能：实现中国与亚洲长打裁决"
```

### Task 6: JSON 棋谱与中文纵线记谱

**Files:**
- Create: `src/xiangqi/record.py`
- Create: `src/xiangqi/notation.py`
- Create: `tests/test_record.py`
- Create: `tests/test_notation.py`

- [ ] **Step 1: 写 JSON 往返与导入回滚失败测试**

```python
def test_json_record_round_trip_preserves_rules_and_moves(tmp_path) -> None:
    record = sample_record(ruleset=Ruleset.ASIAN_2003)
    path = tmp_path / "game.xqjson"
    export_json(record, path)
    assert load_json(path) == record


def test_invalid_import_does_not_replace_current_record(tmp_path) -> None:
    target = tmp_path / "broken.xqjson"
    target.write_text('{"format_version": 999}', encoding="utf-8")
    current = sample_record()
    with pytest.raises(RecordError):
        load_and_validate(target)
    assert current == sample_record()
```

- [ ] **Step 2: 写中文记谱生成解析失败测试**

```python
@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (Coord(1, 7), Coord(4, 7), "炮二平五"),
        (Coord(1, 0), Coord(2, 2), "马８进７"),
    ],
)
def test_format_chinese_notation(start, end, expected) -> None:
    assert (
        format_move(Board.standard(), Move.from_board(Board.standard(), start, end))
        == expected
    )


def test_parse_text_reports_exact_line_for_illegal_move() -> None:
    with pytest.raises(NotationError, match="第 2 行"):
        replay_text("炮二平五\n帅五进三")
```

- [ ] **Step 3: 运行并确认失败**

Run: `python -m pytest tests/test_record.py tests/test_notation.py -v`
Expected: FAIL，缺少 record/notation 模块

- [ ] **Step 4: 实现 Pydantic 棋谱模型和原子写入**

定义版本化 `GameRecord`、`MoveRecord`、`PlayerRecord`、`ResultRecord`，
`load_json()` 后必须从初始 FEN 重放每一步并比对局面摘要。`export_json()` 在
目标目录创建命名临时文件，flush 和 fsync 后用 `os.replace()` 原子替换。

- [ ] **Step 5: 实现中文记谱**

按行棋方视角处理一至九与１至９，支持进、退、平以及同一路前/中/后消歧。
解析时枚举当前方全部合法着法，以生成的标准记谱精确匹配输入；零匹配报非法，
多匹配报歧义。文本导入固定从标准局面开始。

- [ ] **Step 6: 运行棋谱测试**

Run: `python -m pytest tests/test_record.py tests/test_notation.py -v`
Expected: 全部 PASS

- [ ] **Step 7: 提交**

```bash
git add src/xiangqi/record.py src/xiangqi/notation.py tests/test_record.py tests/test_notation.py
git commit -m "功能：支持JSON与中文棋谱"
```

### Task 7: GameController、无限悔棋、和棋与回调

**Files:**
- Create: `src/xiangqi/controller.py`
- Create: `tests/test_controller.py`

- [ ] **Step 1: 写控制器失败测试**

```python
def test_undo_is_unlimited_and_terminal_game_can_resume() -> None:
    controller = GameController.from_record(checkmate_in_one_record())
    controller.make_move(Coord(4, 1), Coord(4, 0))
    assert controller.state.result is not None
    controller.undo()
    assert controller.state.result is None
    controller.undo(controller.state.ply)
    assert controller.state.ply == 0


def test_draw_requires_other_side_and_rejection_resumes() -> None:
    controller = GameController.new()
    controller.offer_draw(Color.RED)
    with pytest.raises(ControlError):
        controller.respond_draw(Color.RED, True)
    controller.respond_draw(Color.BLACK, False)
    assert controller.state.pending_draw is None


def test_callback_error_does_not_rollback_move() -> None:
    controller = GameController.new()
    controller.register_callback(
        lambda event: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    controller.make_move(Coord(0, 6), Coord(0, 5))
    assert controller.state.ply == 1
    assert len(controller.callback_errors) == 1
```

另测控制权冲突、机器控制任意一方或双方、导入事务、悔棋清除提和、悔棋后
新走法截断分支、回放游标不修改棋谱。

- [ ] **Step 2: 运行并确认失败**

Run: `python -m pytest tests/test_controller.py -v`
Expected: FAIL，缺少 `GameController`

- [ ] **Step 3: 实现控制器状态机**

控制器持有 `GameRecord`、当前 `Board`、行动方、状态栈、局面版本、回放游标、
控制权和 callback 列表。所有写操作经同一 `RLock` 串行化；走棋在临时状态
完成合法性、终局、裁决和记录构造后一次提交。每个成功写操作递增版本并发送
不可变 `GameEvent`。

- [ ] **Step 4: 运行控制器和规则回归测试**

Run: `python -m pytest tests/test_controller.py tests/test_adjudication.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/xiangqi/controller.py tests/test_controller.py
git commit -m "功能：实现统一对局控制器"
```

### Task 8: HTTP 与 WebSocket 本机接口

**Files:**
- Create: `src/xiangqi/api.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: 写 HTTP 失败测试**

```python
def test_http_move_requires_matching_version_and_control(test_client) -> None:
    claim = test_client.post(
        "/v1/control/red/claim", json={"controller_id": "bot"}
    ).json()
    stale = test_client.post(
        "/v1/moves",
        json={
            "request_id": "r1",
            "controller_id": "bot",
            "token": claim["token"],
            "expected_version": 999,
            "from": [0, 6],
            "to": [0, 5],
        },
    )
    assert stale.status_code == 409
    assert stale.json()["code"] == "stale_position"
```

- [ ] **Step 2: 写 WebSocket 失败测试**

```python
def test_websocket_observer_receives_move_event(test_client, claimed_red) -> None:
    with test_client.websocket_connect("/v1/events") as ws:
        post_red_move(test_client, claimed_red)
        event = ws.receive_json()
        assert event["type"] == "move_completed"
        assert event["move"]["from"] == [0, 6]
```

- [ ] **Step 3: 运行并确认失败**

Run: `python -m pytest tests/test_api.py -v`
Expected: FAIL，缺少 `create_api`

- [ ] **Step 4: 实现协议适配**

创建 `create_api(controller) -> FastAPI`。HTTP 路由统一捕获领域异常并返回
`{"code", "message", "details"}`。控制权令牌由 `secrets.token_urlsafe(32)`
生成且只存内存。WebSocket 连接拥有独立 asyncio 队列，慢观察者只丢弃旧的
非终局状态事件，不阻塞控制器；请求和响应均携带 `request_id`。

- [ ] **Step 5: 运行 API 测试**

Run: `python -m pytest tests/test_api.py -v`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add src/xiangqi/api.py tests/test_api.py
git commit -m "功能：提供本机HTTP与WebSocket接口"
```

### Task 9: PySide6 棋盘控件与高亮

**Files:**
- Create: `src/xiangqi/ui/__init__.py`
- Create: `src/xiangqi/ui/board_widget.py`
- Create: `tests/ui/test_board_widget.py`

- [ ] **Step 1: 写 Qt 交互失败测试**

```python
def test_left_click_selects_piece_and_shows_all_legal_targets(
    qtbot, controller
) -> None:
    widget = BoardWidget(controller)
    qtbot.addWidget(widget)
    qtbot.mouseClick(
        widget, Qt.MouseButton.LeftButton, pos=widget.point_for(Coord(0, 6))
    )
    assert widget.selected == Coord(0, 6)
    assert widget.legal_targets == {Coord(0, 5)}


def test_opponent_last_piece_stays_highlighted_after_move(qtbot, controller) -> None:
    widget = BoardWidget(controller)
    controller.make_move(Coord(0, 6), Coord(0, 5))
    controller.make_move(Coord(0, 3), Coord(0, 4))
    assert widget.last_move == (Coord(0, 3), Coord(0, 4))
    assert widget.highlighted_piece == Coord(0, 4)
```

另测点击空白取消选择、选中与最后着颜色不同、非法落点不走棋、缩放后坐标命中。

- [ ] **Step 2: 使用 offscreen 平台运行并确认失败**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_board_widget.py -v`
Expected: FAIL，缺少 `BoardWidget`

- [ ] **Step 3: 实现矢量棋盘控件**

继承 `QWidget`，`paintEvent()` 按当前短边计算格距和边距，绘制九路十线、楚河
汉界、九宫斜线、圆形棋子和中文棋名。定义独立颜色绘制合法落点、选中棋子、
最后着起点和最后着终点。`mousePressEvent()` 只处理左键，将像素映射到最近
交点并通过控制器获取合法走法。

- [ ] **Step 4: 运行棋盘 UI 测试**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_board_widget.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/xiangqi/ui/__init__.py src/xiangqi/ui/board_widget.py tests/ui/test_board_widget.py
git commit -m "界面：实现可交互象棋棋盘"
```

### Task 10: 主窗口、玩家设置、文件操作与回放

**Files:**
- Create: `src/xiangqi/ui/dialogs.py`
- Create: `src/xiangqi/ui/main_window.py`
- Create: `tests/ui/test_main_window.py`

- [ ] **Step 1: 写主窗口失败测试**

```python
def test_new_game_rejects_duplicate_colors(qtbot) -> None:
    dialog = NewGameDialog()
    qtbot.addWidget(dialog)
    dialog.player1_color.setCurrentData(Color.RED)
    dialog.player2_color.setCurrentData(Color.RED)
    assert not dialog.can_accept()


def test_replay_cursor_does_not_modify_record(qtbot, loaded_window) -> None:
    original = loaded_window.controller.record.model_copy(deep=True)
    loaded_window.enter_replay()
    loaded_window.replay_next()
    loaded_window.replay_previous()
    assert loaded_window.controller.record == original
```

另测规则模式锁定、悔棋、提和同意/拒绝、首步/前一步/播放/后一步/末步、
播放速度、从此继续、导入错误不替换棋局和终局后恢复。

- [ ] **Step 2: 运行并确认失败**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_main_window.py -v`
Expected: FAIL，缺少主窗口和对话框

- [ ] **Step 3: 实现新局对话框**

提供玩家姓名、互斥颜色、人工/Python/网络方式和
`chinese_2020`/`asian_2003` 规则选择。开始后规则下拉框只读，直到创建新局。

- [ ] **Step 4: 实现主窗口**

使用 splitter 布局玩家面板、`BoardWidget`、着法/状态面板；底部按钮绑定控制器。
文件对话框按扩展名选择 JSON 或中文格式。回放用 `QTimer`，退出和窗口关闭时
必须停止。提和通过非阻塞对话框让另一方确认。

- [ ] **Step 5: 运行主窗口 UI 测试**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_main_window.py -v`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add src/xiangqi/ui/dialogs.py src/xiangqi/ui/main_window.py tests/ui/test_main_window.py
git commit -m "界面：完成主窗口与棋谱回放"
```

### Task 11: 应用启动、API 生命周期与文档

**Files:**
- Create: `src/xiangqi/app.py`
- Create: `src/xiangqi/__main__.py`
- Modify: `src/xiangqi/__init__.py`
- Create: `xiangqi.spec`
- Create: `README.md`
- Modify: `tests/ui/test_main_window.py`

- [ ] **Step 1: 写关闭生命周期失败测试**

```python
def test_close_stops_replay_and_api_thread(qtbot, running_app) -> None:
    running_app.window.start_replay()
    running_app.window.close()
    assert not running_app.window.replay_timer.isActive()
    assert running_app.api_thread.wait(3000)
```

- [ ] **Step 2: 运行并确认失败**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_main_window.py::test_close_stops_replay_and_api_thread -v`
Expected: FAIL，缺少应用生命周期实现

- [ ] **Step 3: 实现启动和关闭**

`app.py` 创建一个控制器，在 `QThread` 中启动 uvicorn，强制 host 为
`127.0.0.1`。窗口关闭信号设置 `server.should_exit = True` 并等待线程；
`__main__.py` 调用 `raise SystemExit(run())`。命令行只允许配置端口和关闭 API，
不允许配置非本机 host。实现不得使用 Windows/Linux 专用 API。

- [ ] **Step 4: 创建 macOS 应用构建配置**

`xiangqi.spec` 使用 `BUNDLE` 生成名为“中国象棋”的窗口应用，收集 PySide6
平台插件、FastAPI/Pydantic/Uvicorn 隐式导入，并把 bundle identifier 设置为
`com.xiangqi.desktop`。应用入口为 `src/xiangqi/__main__.py`，控制台窗口关闭。

- [ ] **Step 5: 编写 README**

给出虚拟环境安装、`python -m xiangqi` 启动、规则模式说明、JSON/中文棋谱、
Python callback/控制示例、HTTP/WebSocket 示例、默认本机安全边界和测试命令。
另给出 `python -m PyInstaller --clean --noconfirm xiangqi.spec` 和从 Finder
启动 `dist/中国象棋.app` 的 macOS 步骤。

- [ ] **Step 6: 运行生命周期测试和启动冒烟测试**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_main_window.py -v`
Expected: 全部 PASS

Run:

```bash
QT_QPA_PLATFORM=offscreen python - <<'PY'
import os
import subprocess
import sys
import time

env = os.environ.copy()
process = subprocess.Popen(
    [sys.executable, "-m", "xiangqi", "--no-api"],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)
time.sleep(5)
assert process.poll() is None, process.communicate()
process.terminate()
stdout, stderr = process.communicate(timeout=3)
assert "Traceback" not in stderr, stderr
PY
```

Expected: 进程持续运行 5 秒且 stderr 中没有 traceback

- [ ] **Step 7: 提交**

```bash
git add src/xiangqi/app.py src/xiangqi/__main__.py src/xiangqi/__init__.py xiangqi.spec README.md tests/ui/test_main_window.py
git commit -m "功能：完成macOS应用启动与使用文档"
```

### Task 12: 全量验证与实际窗口验收

**Files:**
- Modify: only files required by failures found during verification

- [ ] **Step 1: 运行完整测试集**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest -v`
Expected: 0 failed、0 error

- [ ] **Step 2: 运行静态检查**

Run: `python -m ruff check .`
Expected: `All checks passed!`

Run: `python -m ruff format --check .`
Expected: 所有文件已格式化

- [ ] **Step 3: 验证软件包构建**

Run: `python -m build`
Expected: 生成 wheel 和 sdist，命令退出码 0

- [ ] **Step 4: 构建并启动 macOS 应用包**

Run: `python -m PyInstaller --clean --noconfirm xiangqi.spec`
Expected: 生成 `dist/中国象棋.app`，命令退出码 0

Run: `open "dist/中国象棋.app"`
Expected: macOS 显示中国象棋主窗口，能够完成一步走棋并正常退出

- [ ] **Step 5: 实际窗口人工验收**

Run: `python -m xiangqi`

依次验证：创建中国棋规新局；左键选子与全部合法落点；红黑各走一步后确认
对方最后棋子持续高亮；连续悔棋到开局；提和拒绝和同意；JSON 与中文棋谱各
导出、导入；播放回放并从中间继续；终局后悔棋；Python callback；HTTP 控制
红方、WebSocket 观察事件。记录每项结果。上述流程必须在 macOS 实际窗口中
执行，不以 offscreen 测试代替。

- [ ] **Step 6: 检查最终工作树和需求覆盖**

Run: `git status --short`
Expected: 仅出现验收修复或无输出

逐条对照
`docs/superpowers/specs/2026-07-31-xiangqi-game-design.md`，为每一项要求指出
自动化测试或人工验收证据；任何缺少证据的要求都补测或补验。

- [ ] **Step 7: 提交验收修复**

仅在验证产生代码变更时执行：

```bash
git add -u
git commit -m "修复：解决完整验收发现的问题"
```
