import ast
import json
from pathlib import Path

NOTEBOOK = Path("docs/XiangQi-AI-Colab-GPU-Training.ipynb")


def _load() -> dict[str, object]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source() -> str:
    notebook = _load()
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


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
        'status", "--porcelain',
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
