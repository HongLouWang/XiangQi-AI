import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

NOTEBOOK = Path("docs/XiangQi-AI-Colab-GPU-Training.ipynb")


def _load() -> dict[str, object]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source() -> str:
    notebook = _load()
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def _code_cell_containing(marker: str) -> str:
    notebook = _load()
    matches = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and marker in "".join(cell["source"])
    ]
    assert len(matches) == 1
    return matches[0]


def _training_namespace(tmp_path: Path) -> dict[str, object]:
    run_dir = tmp_path / "runs" / "colab-gpu"
    namespace: dict[str, object] = {
        "Path": Path,
        "os": os,
        "subprocess": subprocess,
        "sys": sys,
        "time": time,
        "json": json,
        "uuid": __import__("uuid"),
        "RUN_DIR": run_dir,
        "RUNS_DIR": tmp_path / "runs",
        "SOURCE_DIR": tmp_path / "source",
        "DRIVE_ROOT": tmp_path,
        "LOGS_DIR": tmp_path / "logs",
        "LOG_PATH": tmp_path / "logs" / "colab-gpu.log",
        "PID_PATH": tmp_path / "colab-training.pid",
        "LOCK_DIR": run_dir / ".colab-training.lock",
        "LOCK_STARTUP_GRACE_SECONDS": 120,
        "TARGET_GAMES": 10_000,
        "MAX_FULL_MOVES": 512,
        "DEVICE": "cuda:0",
        "SIMULATIONS": 64,
        "CHANNELS": 64,
        "RESIDUAL_BLOCKS": 4,
        "BATCH_SIZE": 128,
        "CHECKPOINT_INTERVAL_GAMES": 10,
        "GAME_RETRY_LIMIT": 2,
        "SEED": 0,
    }
    exec(  # noqa: S102 - 行为测试需要执行 Notebook 中受控的本地代码单元
        _code_cell_containing("def process_matches"), namespace
    )
    return namespace


def test_colab_notebook_is_nbformat_4_with_compilable_code() -> None:
    notebook = _load()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert notebook["metadata"]["colab"]["name"] == NOTEBOOK.name
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_all_persistent_paths_are_below_drive_root() -> None:
    source = _source()
    assert 'DRIVE_ROOT = Path("/content/drive/MyDrive/XiangQi-AI")' in source
    for expression in (
        'SOURCE_DIR = DRIVE_ROOT / "source"',
        'RUNS_DIR = DRIVE_ROOT / "runs"',
        'TEMP_DIR = DRIVE_ROOT / "temp"',
        'CACHE_DIR = DRIVE_ROOT / "cache"',
        'LOGS_DIR = DRIVE_ROOT / "logs"',
        'PID_PATH = DRIVE_ROOT / "colab-training.pid"',
    ):
        assert expression in source
    assert "RUN_DIR = RUNS_DIR / RUN_NAME" in source
    assert 'LOG_PATH = LOGS_DIR / f"{RUN_NAME}.log"' in source


def test_notebook_contains_drive_source_install_and_cuda_checks() -> None:
    source = _source()
    for required in (
        "drive.mount",
        'git", "clone',
        'status", "--short',
        'pull", "--ff-only',
        'pip", "install", "-e"',
        "nvidia-smi",
        "torch.cuda.is_available",
    ):
        assert required in source
    assert "如果没有可用的 NVIDIA CUDA GPU" in source


def test_notebook_contains_complete_gpu_training_workflow() -> None:
    source = _source()
    for required in (
        'DEVICE = "cuda:0"',
        "TARGET_GAMES = 10_000",
        "MAX_FULL_MOVES = 512",
        "subprocess.Popen",
        "def process_matches",
        "def training_command",
        "def start_background_training",
        "def run_ai_command",
        '"pause"',
        '"extend"',
        '"resume"',
        "weights_only=True",
        "ADDITIONAL_GAMES = 5_000",
    ):
        assert required in source


def test_commands_are_argument_lists_and_pid_check_is_run_specific() -> None:
    source = _source()
    assert "shell=True" not in source
    assert 'Path(f"/proc/{pid}/cmdline")' in source
    assert "str(RUN_DIR).encode()" in source
    assert '"-m" in parts and "ai" in parts' in source
    assert "training_command(resume=True)" in source


def test_training_lock_atomically_rejects_a_competing_session(tmp_path: Path) -> None:
    namespace = _training_namespace(tmp_path)
    token = namespace["acquire_training_lock"]()
    with pytest.raises(RuntimeError, match="锁|启动"):
        namespace["acquire_training_lock"]()
    metadata = json.loads((namespace["LOCK_DIR"] / "owner.json").read_text())
    assert metadata["token"] == token
    assert metadata["run_dir"] == str(namespace["RUN_DIR"])


def test_training_lock_safely_recovers_a_stale_owner(tmp_path: Path) -> None:
    namespace = _training_namespace(tmp_path)
    lock_dir = namespace["LOCK_DIR"]
    lock_dir.mkdir(parents=True)
    (lock_dir / "owner.json").write_text(
        json.dumps(
            {
                "token": "old-token",
                "run_dir": str(namespace["RUN_DIR"]),
                "pid": 987654,
                "created_at": 0,
            }
        )
    )
    namespace["process_matches"] = lambda pid: False
    new_token = namespace["acquire_training_lock"]()
    metadata = json.loads((lock_dir / "owner.json").read_text())
    assert metadata["token"] == new_token
    assert metadata["token"] != "old-token"


def test_popen_failure_releases_only_its_own_training_lock(
    tmp_path: Path,
) -> None:
    namespace = _training_namespace(tmp_path)
    lock_existed_before_popen = False

    class FailingPopen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            nonlocal lock_existed_before_popen
            lock_existed_before_popen = namespace["LOCK_DIR"].is_dir()
            raise OSError("cannot start")

    namespace["subprocess"] = SimpleNamespace(
        Popen=FailingPopen,
        STDOUT=subprocess.STDOUT,
        run=subprocess.run,
        CompletedProcess=subprocess.CompletedProcess,
    )
    with pytest.raises(OSError, match="cannot start"):
        namespace["start_background_training"]()
    assert lock_existed_before_popen
    assert not namespace["LOCK_DIR"].exists()


def test_metadata_failure_stops_spawned_process_before_unlocking(
    tmp_path: Path,
) -> None:
    namespace = _training_namespace(tmp_path)

    class StartedProcess:
        pid = 4321
        terminated = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: int) -> int:
            assert timeout == 30
            return 0

    process = StartedProcess()
    namespace["subprocess"] = SimpleNamespace(
        Popen=lambda *args, **kwargs: process,
        STDOUT=subprocess.STDOUT,
        run=subprocess.run,
        CompletedProcess=subprocess.CompletedProcess,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    original_write = namespace["_write_lock_metadata"]
    writes = 0

    def fail_second_metadata_write(token: str, pid: int | None) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("drive write failed")
        original_write(token, pid)

    namespace["_write_lock_metadata"] = fail_second_metadata_write
    with pytest.raises(OSError, match="drive write failed"):
        namespace["start_background_training"]()
    assert process.terminated
    assert not namespace["LOCK_DIR"].exists()


def test_training_argv_and_notebook_setup_order() -> None:
    notebook = _load()
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    status_index = next(
        index for index, cell in enumerate(code_cells) if '"rev-parse"' in cell
    )
    pull_index = next(
        index for index, cell in enumerate(code_cells) if '"pull", "--ff-only"' in cell
    )
    cuda_index = next(
        index
        for index, cell in enumerate(code_cells)
        if "torch.cuda.mem_get_info" in cell
    )
    config_index = next(
        index
        for index, cell in enumerate(code_cells)
        if "TARGET_GAMES = 10_000" in cell
    )
    assert status_index < pull_index < cuda_index < config_index

    namespace = _training_namespace(Path("/tmp/notebook-contract"))
    new_command = namespace["training_command"](resume=False)
    resume_command = namespace["training_command"](resume=True)
    assert new_command[:4] == [sys.executable, "-m", "ai", "train"]
    assert resume_command[:4] == [sys.executable, "-m", "ai", "resume"]
    assert "--run-dir" in new_command and "--run-dir" in resume_command


def test_smoke_cell_is_idempotent_for_complete_artifacts() -> None:
    smoke_source = _code_cell_containing("SMOKE_RUN_DIR")
    assert "SMOKE_FINAL_MODEL" in smoke_source
    assert "SMOKE_CHECKPOINTS" in smoke_source
    assert "Smoke 已完成，跳过重复训练" in smoke_source
    assert "存在不完整数据" in smoke_source


def test_notebook_never_uses_unsafe_or_destructive_operations() -> None:
    source = _source()
    for forbidden in (
        "weights_only=False",
        "reset --hard",
        "checkout -f",
        "pull --force",
        "shutil.rmtree",
        "%pip",
        "!git",
    ):
        assert forbidden not in source


def test_user_guides_link_to_colab_notebook() -> None:
    notebook_name = "XiangQi-AI-Colab-GPU-Training.ipynb"
    assert f"docs/{notebook_name}" in Path("README.md").read_text(encoding="utf-8")
    assert notebook_name in Path("docs/AI-TRAINING-GUIDE.md").read_text(
        encoding="utf-8"
    )
