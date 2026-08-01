# 训练棋局胜负统计脚本实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建一个可在服务器项目根目录独立运行的只读 Python 脚本，准确统计当前 Replay 中的红胜、黑胜、和棋和异常棋局。

**Architecture:** 根目录脚本同时提供可测试的纯函数和 CLI 入口。它从 manifest 获取一致的棋局 ID 快照，逐个只读加载 NPZ，将行棋方视角的 value 换算成红方视角后分类；单局损坏被隔离为异常，目录或 manifest 级错误则令 CLI 非零退出。

**Tech Stack:** Python 3.11+、NumPy、argparse、pytest、JSON、pathlib

---

## 文件结构

- 新建 `src/ai/count_training_results.py`：路径解析、JSON/NPZ 读取、棋局分类、中文报告和 CLI。
- 新建 `tests/test_count_training_results.py`：临时训练目录夹具、分类、错误隔离、默认路径和 CLI 测试。

### Task 1: 核心统计与错误隔离

**Files:**
- Create: `src/ai/count_training_results.py`
- Test: `tests/test_count_training_results.py`

- [ ] **Step 1: 写红胜、黑胜、和棋与异常文件的失败测试**

在测试中用 `np.savez_compressed` 创建四个棋局文件：红方视角分别为全 `1`、全
`-1`、全 `0`，以及缺少 `sides` 的损坏文件。写入包含这四个 ID 的 manifest，调用：

```python
report = count_results(run_dir)
assert report.red_wins == 1
assert report.black_wins == 1
assert report.draws == 1
assert report.invalid_games == 1
assert report.classified_games == 3
assert report.retained_games == 4
assert report.total_games == 4
assert report.errors[0].game_id == 4
```

- [ ] **Step 2: 运行测试并确认因模块缺失而失败**

Run: `.venv/bin/python -m pytest tests/test_count_training_results.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'count_training_results'`。

- [ ] **Step 3: 实现最小统计模型与分类函数**

在脚本中定义不可变数据类型：

```python
@dataclass(frozen=True, slots=True)
class GameError:
    game_id: int
    reason: str

@dataclass(frozen=True, slots=True)
class TrainingResultReport:
    run_dir: Path
    phase: str
    completed_games: int | None
    target_games: int | None
    training_steps: int | None
    total_games: int
    retained_games: int
    red_wins: int
    black_wins: int
    draws: int
    errors: tuple[GameError, ...]

    @property
    def invalid_games(self) -> int:
        return len(self.errors)

    @property
    def classified_games(self) -> int:
        return self.red_wins + self.black_wins + self.draws
```

实现 `_classify_game(path)`：验证 `values`、`sides` 为等长非空一维数组，验证
`sides` 仅为 `0/1`，计算 `red_values = np.where(sides == 0, values, -values)`，使用
`np.allclose` 分类为 `red_win`、`black_win` 或 `draw`，其余抛出 `ValueError`。

实现 `count_results(run_dir)`：读取 `replay/manifest.json`，验证 `games` 是非布尔整数
列表、`total_games` 是不小于保留局数的非负整数；可选读取 `status.json`；逐局读取
`replay/games/{game_id:012d}.npz`。单局异常转为 `GameError`，继续处理其他棋局。

- [ ] **Step 4: 运行核心测试并确认通过**

Run: `.venv/bin/python -m pytest tests/test_count_training_results.py -v`

Expected: 分类测试 PASS，且无警告。

- [ ] **Step 5: 增加 manifest 级失败测试**

添加目录不存在、manifest 缺失、非法 JSON、重复或非法棋局 ID、`total_games` 小于
保留局数的参数化测试，断言 `count_results` 抛出带目标路径或字段名的
`TrainingDataError`。

- [ ] **Step 6: 运行测试并确认新增测试先失败**

Run: `.venv/bin/python -m pytest tests/test_count_training_results.py -v`

Expected: 新增验证场景 FAIL，因为当前实现尚未提供完整的 `TrainingDataError` 信息。

- [ ] **Step 7: 实现 manifest 和 status 验证**

定义：

```python
class TrainingDataError(RuntimeError):
    pass
```

用 `_read_json_object(path, *, required)` 统一处理文件缺失、UTF-8、JSON 和顶层对象
验证；manifest 错误包装为 `TrainingDataError`。`status.json` 不存在时阶段显示
`unknown`、三个进度字段显示 `None`；存在但字段不合法时同样报目录级错误，避免输出
误导性进度。

- [ ] **Step 8: 运行核心测试并确认全部通过**

Run: `.venv/bin/python -m pytest tests/test_count_training_results.py -v`

Expected: 全部 PASS。

- [ ] **Step 9: 提交核心统计**

```bash
git add src/ai/count_training_results.py tests/test_count_training_results.py
git commit -m "功能：新增训练棋局胜负统计"
```

### Task 2: CLI、完整验证与服务器部署

**Files:**
- Modify: `src/ai/count_training_results.py`
- Modify: `tests/test_count_training_results.py`

- [ ] **Step 1: 写默认路径、中文输出和错误退出测试**

用 monkeypatch 替换模块级 `AI_ROOT` 为临时 AI 目录，断言无参数时解析
`AI-runs/cpu-main`。调用 `main([...])` 并检查：

```python
assert exit_code == 0
assert "红方胜：1 局" in captured.out
assert "黑方胜：1 局" in captured.out
assert "和棋：1 局" in captured.out
assert "异常：1 局" in captured.out
```

再传入不存在的 `--run-dir`，断言退出码为 `2`、错误写入 stderr 且不输出 traceback。

- [ ] **Step 2: 运行 CLI 测试并确认失败**

Run: `.venv/bin/python -m pytest tests/test_count_training_results.py -v`

Expected: CLI 测试 FAIL，因为 `main` 和格式化输出尚未实现。

- [ ] **Step 3: 实现 CLI 和中文报告**

定义 `AI_ROOT = Path(__file__).resolve().parent`；argparse 的 `--run-dir` 默认值为
`AI_ROOT / "AI-runs/cpu-main"`，相对的用户参数按当前工作目录解析。实现
`format_report(report, observed_at)`，输出北京时间 ISO 时间、绝对训练目录、阶段、
`completed/target` 进度、历史累计、当前保留、已分类及胜负和异常；异常不为空时逐行
列出棋局 ID 与单行原因。`main()` 捕获 `TrainingDataError`，向 stderr 输出
`统计失败：...` 并返回 `2`。

- [ ] **Step 4: 运行目标测试和完整测试套件**

Run: `.venv/bin/python -m pytest tests/test_count_training_results.py -v`

Expected: 全部 PASS。

Run: `.venv/bin/python -m pytest -q`

Expected: 全部项目测试 PASS；如有既存失败，准确记录完整数量并确认目标测试仍通过。

- [ ] **Step 5: 静态验证和本地真实目录试运行**

```bash
.venv/bin/python -m py_compile src/ai/count_training_results.py
git diff --check
.venv/bin/python src/ai/count_training_results.py \
  --run-dir src/ai/AI-runs/cpu-main
```

Expected: 编译与 diff 检查退出码为 `0`；本地目录不存在时只允许得到受控的
`统计失败` 和退出码 `2`，不得出现 traceback。

- [ ] **Step 6: 提交 CLI 与测试**

```bash
git add src/ai/count_training_results.py tests/test_count_training_results.py
git commit -m "完善：增加训练胜负统计命令行输出"
```

- [ ] **Step 7: 只上传独立脚本**

使用项目已有 SFTP 凭据，仅将本地 `src/ai/count_training_results.py` 上传到
`/XiangQi-AI/src/ai/count_training_results.py`。上传前后分别计算本地与远端 SHA-256，要求
两者完全一致；不上传 `.vscode`、测试、规格或计划，不重启训练进程。

- [ ] **Step 8: 在服务器只读运行并核对快照**

```bash
cd /XiangQi-AI
.venv/bin/python src/ai/count_training_results.py \
  --run-dir src/ai/AI-runs/cpu-main
```

Expected: 退出码 `0`，胜负和异常之和等于当前保留棋局数；manifest 历史累计局数与
输出一致；`status.json` 的完成局数与输出一致。记录执行时间，因为训练仍在继续。
