# Colab Drive Notebook 首次训练锁修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复首次后台训练因 Notebook 锁污染 `RUN_DIR` 而失败的问题，并将经过验证的干净 Notebook 原位覆盖到 Google Drive。

**Architecture:** Notebook 的跨会话互斥锁、锁 metadata 临时文件和陈旧锁回收目录统一迁移到 `DRIVE_ROOT / "locks"`，训练器的 `RUN_DIR` 只保存训练产物。增加受保护的一次性恢复函数和不抛出式状态诊断单元，随后通过 Drive `files.update` 保持原文件 ID 覆盖，并重新读取验证。

**Tech Stack:** Python 3.12+、Jupyter nbformat 4 JSON、pytest、subprocess、pathlib、Google Drive connector

---

### Task 1: 用失败测试固定首次训练目录为空的要求

**Files:**
- Modify: `tests/test_colab_gpu_notebook.py`
- Test: `tests/test_colab_gpu_notebook.py`

- [ ] **Step 1: 新增锁路径隔离测试**

新增测试解析 Notebook 配置与函数单元，断言 `LOCKS_DIR` 位于 `DRIVE_ROOT / "locks"`，`LOCK_DIR` 不以 `RUN_DIR` 为父目录，锁 metadata 临时文件和 stale 目录均不写入 `RUN_DIR`。

```python
def test_training_lock_never_uses_run_dir(notebook_cells):
    source = notebook_cells.combined_source
    assert 'LOCKS_DIR = DRIVE_ROOT / "locks"' in source
    assert 'LOCK_DIR = LOCKS_DIR / f"{RUN_NAME}.lock"' in source
    assert 'temporary = LOCKS_DIR / f".owner-{token}.tmp"' in source
    assert 'stale_dir = LOCKS_DIR / f".{RUN_NAME}.stale-' in source
    assert 'LOCK_DIR = RUN_DIR / ".colab-training.lock"' not in source
```

- [ ] **Step 2: 运行测试确认 RED**

Run: `.venv/bin/python -m pytest tests/test_colab_gpu_notebook.py::test_training_lock_never_uses_run_dir -v`

Expected: FAIL，指出现有 Notebook 仍包含 `LOCK_DIR = RUN_DIR / ".colab-training.lock"`。

- [ ] **Step 3: 提交仅包含测试的 RED 检查点**

```bash
git add tests/test_colab_gpu_notebook.py
git commit -m "测试：覆盖 Colab 首次训练目录隔离"
```

### Task 2: 迁移锁路径并保持并发安全

**Files:**
- Modify: `docs/XiangQi-AI-Colab-GPU-Training.ipynb`
- Test: `tests/test_colab_gpu_notebook.py`

- [ ] **Step 1: 修改持久化目录和参数单元**

在目录单元创建 `LOCKS_DIR`，在参数单元使用外置锁：

```python
LOCKS_DIR = DRIVE_ROOT / "locks"
LOCKS_DIR.mkdir(parents=True, exist_ok=True)
LOCK_DIR = LOCKS_DIR / f"{RUN_NAME}.lock"
```

- [ ] **Step 2: 将锁临时路径和陈旧锁路径迁出 RUN_DIR**

```python
temporary = LOCKS_DIR / f".owner-{RUN_NAME}-{token}.tmp"
stale_dir = LOCKS_DIR / f".{RUN_NAME}.stale-{uuid.uuid4().hex}"
```

保留原子 `mkdir`、token 所有权校验、启动宽限、精确 PID argv 检查、进程失败终止与陈旧锁原子 rename 逻辑。

- [ ] **Step 3: 运行锁相关测试确认 GREEN**

Run: `.venv/bin/python -m pytest tests/test_colab_gpu_notebook.py -k 'lock or metadata or process' -v`

Expected: PASS。

- [ ] **Step 4: 提交锁迁移**

```bash
git add docs/XiangQi-AI-Colab-GPU-Training.ipynb tests/test_colab_gpu_notebook.py
git commit -m "修复：隔离 Colab 训练锁目录"
```

### Task 3: 增加安全恢复与可诊断状态查询

**Files:**
- Modify: `docs/XiangQi-AI-Colab-GPU-Training.ipynb`
- Modify: `tests/test_colab_gpu_notebook.py`

- [ ] **Step 1: 先增加安全恢复的失败测试**

测试从 Notebook 提取 `cleanup_failed_initial_run()`，验证存在匹配进程、`status.json`、任一 checkpoint、最终模型或旧锁未知文件时均抛出 `RuntimeError`；只有目录包含 `.colab-training.lock/owner.json`、`control.lock` 和 `pause.json` 时才清理。

```python
with pytest.raises(RuntimeError):
    cleanup_failed_initial_run(process_is_live=True)
assert checkpoint_path.exists()
```

- [ ] **Step 2: 运行安全恢复测试确认 RED**

Run: `.venv/bin/python -m pytest tests/test_colab_gpu_notebook.py -k cleanup_failed_initial_run -v`

Expected: FAIL，因为函数尚不存在。

- [ ] **Step 3: 实现受保护的一次性恢复函数和独立单元**

函数必须先检查进程和训练产物，再逐一 `unlink` 已知文件并用非递归 `rmdir` 删除旧锁；不得使用 `rm -rf`、`shutil.rmtree` 或 glob 删除。

```python
def cleanup_failed_initial_run() -> None:
    saved_pid = _saved_pid()
    if saved_pid is not None and process_matches(saved_pid):
        raise RuntimeError(f"训练进程仍在运行，PID={saved_pid}")
    protected = (RUN_DIR / "status.json", RUN_DIR / "checkpoint-a.pt",
                 RUN_DIR / "checkpoint-b.pt", RUN_DIR / "final_model.pt")
    if any(path.exists() for path in protected):
        raise RuntimeError("检测到有效训练数据，拒绝清理")
```

- [ ] **Step 4: 将状态查询改为不抛出并总是显示日志**

```python
status_result = subprocess.run(
    [sys.executable, "-m", "ai", "status", "--run-dir", str(RUN_DIR)],
    cwd=SOURCE_DIR,
    env=os.environ.copy(),
    text=True,
    capture_output=True,
    check=False,
)
print("退出码:", status_result.returncode)
print(status_result.stdout or status_result.stderr)
```

- [ ] **Step 5: 运行专项测试确认 GREEN 并提交**

Run: `.venv/bin/python -m pytest tests/test_colab_gpu_notebook.py -v`

Expected: 所有专项测试 PASS。

```bash
git add docs/XiangQi-AI-Colab-GPU-Training.ipynb tests/test_colab_gpu_notebook.py
git commit -m "修复：增加 Colab 失败启动安全恢复"
```

### Task 4: 清理 Notebook 输出并完成本地验证

**Files:**
- Modify: `docs/XiangQi-AI-Colab-GPU-Training.ipynb`
- Test: `tests/test_colab_gpu_notebook.py`

- [ ] **Step 1: 清空所有执行输出和计数**

对每个代码单元设置：

```python
cell["execution_count"] = None
cell["outputs"] = []
```

- [ ] **Step 2: 验证 Notebook 与仓库**

Run: `.venv/bin/python -m pytest tests/test_colab_gpu_notebook.py -v`

Expected: PASS。

Run: `.venv/bin/python -m ruff check .`

Expected: `All checks passed!`

Run: `.venv/bin/python -m compileall -q src tests && git diff --check`

Expected: exit 0。

- [ ] **Step 3: 提交干净 Notebook**

```bash
git add docs/XiangQi-AI-Colab-GPU-Training.ipynb tests/test_colab_gpu_notebook.py
git commit -m "文档：清理 Colab Notebook 执行输出"
```

### Task 5: 原位覆盖 Google Drive 并回读验证

**Files:**
- Read: `docs/XiangQi-AI-Colab-GPU-Training.ipynb`
- Update Drive file ID: `1bSAvSfcmNjhgt99VvpM606Clov49_y5A`

- [ ] **Step 1: 计算本地文件哈希和大小**

Run: `shasum -a 256 docs/XiangQi-AI-Colab-GPU-Training.ipynb && wc -c docs/XiangQi-AI-Colab-GPU-Training.ipynb`

Expected: 输出非空 SHA-256 和字节数。

- [ ] **Step 2: 使用 Drive update_file 原位替换字节**

传入现有 `fileId=1bSAvSfcmNjhgt99VvpM606Clov49_y5A`、本地 Notebook 绝对路径和 `application/x-ipynb+json`；不传 `addParents`、`removeParents` 或新名称。

- [ ] **Step 3: 从 Drive 回读同一文件 ID**

使用 Drive fetch 的 `download_raw_file=True`，确认返回 ID 未变化、MIME 类型为 `application/x-ipynb+json`、修改时间更新，并检查回读 JSON 包含外置 `LOCKS_DIR` 且不包含旧的 `LOCK_DIR = RUN_DIR`。

- [ ] **Step 4: 最终报告**

报告 Drive 文件链接、同一文件 ID、本地专项验证结果和真实 CUDA 仍由 Colab 环境执行的边界。
