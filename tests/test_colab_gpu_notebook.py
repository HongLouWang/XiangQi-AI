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
        "LOCKS_DIR": tmp_path / "locks",
        "LOG_PATH": tmp_path / "logs" / "colab-gpu.log",
        "PID_PATH": tmp_path / "colab-training.pid",
        "LOCK_DIR": tmp_path / "locks" / "colab-gpu.lock",
        "RUN_NAME": "colab-gpu",
        "LOCK_STARTUP_GRACE_SECONDS": 120,
        "TARGET_GAMES": 10_000,
        "MAX_FULL_MOVES": 512,
        "DEVICE": "cuda:0",
        "PARALLEL_GAMES": 16,
        "SIMULATIONS": 64,
        "CHANNELS": 64,
        "RESIDUAL_BLOCKS": 4,
        "BATCH_SIZE": 128,
        "CHECKPOINT_INTERVAL_GAMES": 10,
        "GAME_RETRY_LIMIT": 2,
        "SEED": 0,
    }
    namespace["LOCKS_DIR"].mkdir(parents=True, exist_ok=True)
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
        "PARALLEL_GAMES = 16",
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


def test_colab_training_commands_pass_parallel_games() -> None:
    namespace = _training_namespace(Path("/tmp/notebook-parallel-contract"))
    namespace["PARALLEL_GAMES"] = 16
    new_command = namespace["training_command"](resume=False)
    resume_command = namespace["training_command"](resume=True)
    assert new_command[new_command.index("--parallel-games") + 1] == "16"
    assert resume_command[resume_command.index("--parallel-games") + 1] == "16"


def test_commands_are_argument_lists_and_pid_check_is_run_specific() -> None:
    source = _source()
    assert "shell=True" not in source
    assert 'Path(f"/proc/{pid}/cmdline")' in source
    assert "str(RUN_DIR).encode()" in source
    assert '"-m" in parts and "ai" in parts' in source
    assert "training_command(resume=True)" in source


def test_training_lock_never_uses_run_dir() -> None:
    source = _source()
    assert 'LOCKS_DIR = DRIVE_ROOT / "locks"' in source
    assert 'LOCK_DIR = LOCKS_DIR / f"{RUN_NAME}.lock"' in source
    assert 'temporary = LOCKS_DIR / f".owner-{RUN_NAME}-{token}.tmp"' in source
    assert 'stale_dir = LOCKS_DIR / f".{RUN_NAME}.stale-' in source
    assert 'LOCK_DIR = RUN_DIR / ".colab-training.lock"' not in source


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


@pytest.mark.parametrize("created_at", [None, "not-a-timestamp"])
def test_training_lock_uses_mtime_when_created_at_is_invalid(
    tmp_path: Path, created_at: object
) -> None:
    namespace = _training_namespace(tmp_path)
    lock_dir = namespace["LOCK_DIR"]
    lock_dir.mkdir(parents=True)
    (lock_dir / "owner.json").write_text(
        json.dumps(
            {
                "token": "old-token",
                "run_dir": str(namespace["RUN_DIR"]),
                "pid": 987654,
                "created_at": created_at,
            }
        )
    )
    old_time = time.time() - 1_000
    os.utime(lock_dir, (old_time, old_time))
    namespace["process_matches"] = lambda pid: False
    assert namespace["acquire_training_lock"]() != "old-token"


def test_initial_metadata_failure_removes_own_empty_lock_and_allows_retry(
    tmp_path: Path,
) -> None:
    namespace = _training_namespace(tmp_path)
    original_write = namespace["_write_lock_metadata"]
    namespace["_write_lock_metadata"] = lambda token, pid: (_ for _ in ()).throw(
        OSError("drive unavailable")
    )
    with pytest.raises(OSError, match="drive unavailable"):
        namespace["acquire_training_lock"]()
    assert not namespace["LOCK_DIR"].exists()
    namespace["_write_lock_metadata"] = original_write
    assert namespace["acquire_training_lock"]()


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


def test_early_process_exit_releases_training_lock(tmp_path: Path) -> None:
    namespace = _training_namespace(tmp_path)

    class ExitedProcess:
        pid = 4321

        def poll(self) -> int:
            return 1

    namespace["subprocess"] = SimpleNamespace(
        Popen=lambda *args, **kwargs: ExitedProcess(),
        STDOUT=subprocess.STDOUT,
        run=subprocess.run,
        CompletedProcess=subprocess.CompletedProcess,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    namespace["time"] = SimpleNamespace(
        time=time.time, sleep=lambda seconds: None, monotonic=time.monotonic
    )
    with pytest.raises(RuntimeError, match="启动失败"):
        namespace["start_background_training"]()
    assert not namespace["LOCK_DIR"].exists()


def test_live_pid_and_fresh_startup_lock_are_never_reclaimed(tmp_path: Path) -> None:
    namespace = _training_namespace(tmp_path)
    token = namespace["acquire_training_lock"]()
    namespace["_write_lock_metadata"](token, pid=4321)
    namespace["process_matches"] = lambda pid: pid == 4321
    with pytest.raises(RuntimeError, match="PID=4321"):
        namespace["acquire_training_lock"]()
    assert namespace["LOCK_DIR"].is_dir()

    namespace["process_matches"] = lambda pid: False
    namespace["_write_lock_metadata"](token, pid=None)
    with pytest.raises(RuntimeError, match="正在启动"):
        namespace["acquire_training_lock"]()
    assert namespace["LOCK_DIR"].is_dir()


def test_cleanup_failed_initial_run_removes_only_known_control_files(
    tmp_path: Path,
) -> None:
    namespace = _training_namespace(tmp_path)
    run_dir = namespace["RUN_DIR"]
    run_dir.mkdir(parents=True)
    (run_dir / "pause.json").write_text("{}")
    (run_dir / "control.lock").write_text("")
    legacy_lock = run_dir / ".colab-training.lock"
    legacy_lock.mkdir()
    (legacy_lock / "owner.json").write_text("{}")
    namespace["PID_PATH"].write_text("987654\n")
    namespace["process_matches"] = lambda pid: False

    namespace["cleanup_failed_initial_run"]()

    assert run_dir.is_dir()
    assert not any(run_dir.iterdir())
    assert not namespace["PID_PATH"].exists()


@pytest.mark.parametrize(
    "protected_name",
    ["status.json", "checkpoint-a.pt", "checkpoint-b.pt", "final_model.pt"],
)
def test_cleanup_failed_initial_run_preserves_training_artifacts(
    tmp_path: Path, protected_name: str
) -> None:
    namespace = _training_namespace(tmp_path)
    run_dir = namespace["RUN_DIR"]
    run_dir.mkdir(parents=True)
    protected = run_dir / protected_name
    protected.write_text("data")

    with pytest.raises(RuntimeError, match="有效训练数据"):
        namespace["cleanup_failed_initial_run"]()

    assert protected.exists()


def test_cleanup_failed_initial_run_rejects_live_process_and_unknown_lock_file(
    tmp_path: Path,
) -> None:
    namespace = _training_namespace(tmp_path)
    namespace["PID_PATH"].write_text("4321\n")
    namespace["process_matches"] = lambda pid: pid == 4321
    with pytest.raises(RuntimeError, match="PID=4321"):
        namespace["cleanup_failed_initial_run"]()

    namespace["process_matches"] = lambda pid: False
    legacy_lock = namespace["RUN_DIR"] / ".colab-training.lock"
    legacy_lock.mkdir(parents=True)
    unknown = legacy_lock / "unexpected.bin"
    unknown.write_bytes(b"keep")
    with pytest.raises(RuntimeError, match="未知文件"):
        namespace["cleanup_failed_initial_run"]()
    assert unknown.exists()


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


def test_pull_cell_rechecks_status_and_refuses_new_dirty_state() -> None:
    pull_source = _code_cell_containing('"pull", "--ff-only"')
    calls: list[list[str]] = []

    def fake_run(
        arguments: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        if "status" in arguments:
            return subprocess.CompletedProcess(arguments, 0, stdout=" M README.md\n")
        raise AssertionError("dirty source must never be pulled")

    namespace = {
        "SOURCE_DIR": Path("/drive/source"),
        "source_status": "",
        "subprocess": SimpleNamespace(run=fake_run),
    }
    exec(pull_source, namespace)  # noqa: S102 - 执行受控 Notebook 单元验证实际行为
    assert len(calls) == 1
    assert calls[0][-2:] == ["status", "--short"]


def test_smoke_cell_is_idempotent_for_complete_artifacts() -> None:
    smoke_source = _code_cell_containing("SMOKE_RUN_DIR")
    assert "SMOKE_FINAL_MODEL" in smoke_source
    assert "SMOKE_CHECKPOINTS" in smoke_source
    assert "Smoke 已完成，跳过重复训练" in smoke_source
    assert "存在不完整数据" in smoke_source


def test_notebook_explains_full_move_limit_and_legal_endings() -> None:
    source = _source()
    assert "512 个完整回合" in source
    assert "1024 ply" in source
    assert "到达上限" in source and "和棋" in source
    assert "合法" in source and "将死" in source and "立即结束" in source


def test_status_cell_reports_command_failure_before_reading_logs() -> None:
    source = _code_cell_containing("log_lines[-100:]")
    assert "subprocess.run(" in source
    assert "check=False" in source
    assert "status_result.returncode" in source
    assert "status_result.stderr" in source


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
