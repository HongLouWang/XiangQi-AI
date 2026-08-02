# Colab GPU 训练 Notebook 设计

## 目标

新增 `docs/XiangQi-AI-Colab-GPU-Training.ipynb`，用于在 Google Colab 上通过
NVIDIA CUDA 训练独立的中国象棋 AlphaZero AI。源码、临时文件、缓存、日志、
Replay、checkpoint 和最终模型全部持久化到 Google Drive 的
`MyDrive/XiangQi-AI/` 目录。

Notebook 不接入桌面 UI、controller 或 HTTP API，也不修改训练核心逻辑。

## Drive 目录

Notebook 使用固定的 Drive 根目录：

```text
/content/drive/MyDrive/XiangQi-AI/
├── source/                  # GitHub 源码
├── runs/colab-gpu/          # Replay、状态、checkpoint、最终模型
├── temp/                    # Python/系统临时文件
├── cache/
│   ├── pip/                 # pip 下载缓存
│   └── torch/               # PyTorch 缓存
├── logs/
│   └── colab-gpu.log        # 后台训练日志
└── colab-training.pid       # 当前后台训练 PID
```

Notebook 挂载 Drive 后设置：

```python
os.environ["TMPDIR"] = str(DRIVE_ROOT / "temp")
os.environ["PIP_CACHE_DIR"] = str(DRIVE_ROOT / "cache" / "pip")
os.environ["TORCH_HOME"] = str(DRIVE_ROOT / "cache" / "torch")
```

## 源码获取和更新

首次运行时，将公开仓库
`https://github.com/HongLouWang/XiangQi-AI.git` 克隆到 Drive 的 `source/`。

已存在源码时默认不自动执行破坏性更新。Notebook 提供单独的“检查与安全更新源码”
单元格：

- 先显示分支、commit 和 `git status --short`；
- 工作区干净时执行 `git pull --ff-only`；
- 工作区存在修改时拒绝更新，并提示用户自行处理；
- 不运行 reset、checkout 或删除操作。

每次 Colab 运行时从 Drive 源码执行 editable 安装：

```bash
python -m pip install -e "/content/drive/MyDrive/XiangQi-AI/source[ai]"
```

## Notebook 单元格顺序

Notebook 依次提供以下可独立重跑的单元格：

1. 使用说明和 Colab GPU 运行时提示。
2. 挂载 Google Drive。
3. 创建目录并设置临时目录/缓存环境变量。
4. 克隆或检查 Drive 源码。
5. 可选的安全源码更新。
6. 安装项目 AI 依赖。
7. 检测 PyTorch、CUDA、GPU 名称和显存。
8. 编辑训练配置。
9. 执行一局极小 CUDA smoke。
10. 后台启动或自动续传正式训练。
11. 查询训练状态。
12. 查看日志尾部。
13. 安全暂停。
14. 追加累计训练局数。
15. 恢复训练。
16. 检查后台进程。
17. 验证 checkpoint 和 `final_model.pt`。
18. Colab 断线后的恢复步骤。

## 默认训练配置

配置单元格使用普通 Python 常量，用户可以直接修改：

```python
RUN_NAME = "colab-gpu"
TARGET_GAMES = 10_000
MAX_FULL_MOVES = 512
DEVICE = "cuda:0"
SIMULATIONS = 64
CHANNELS = 64
RESIDUAL_BLOCKS = 4
BATCH_SIZE = 128
CHECKPOINT_INTERVAL_GAMES = 10
GAME_RETRY_LIMIT = 2
SEED = 0
```

`RUN_DIR` 始终派生为 Drive 下的 `runs/<RUN_NAME>`。Notebook 明确说明 512 个完整
回合等于 1024 ply，只有仍未终局的上限局才判和。

## CUDA 检查

GPU 检测单元格打印：

- `nvidia-smi`；
- PyTorch 版本；
- `torch.cuda.is_available()`；
- CUDA 设备数量、名称和显存；
- 当前训练设备。

若 CUDA 不可用，单元格立即抛出带中文说明的 `RuntimeError`，提示在 Colab 菜单中
选择 GPU 运行时。正式训练单元格不会静默改用 CPU。

## Smoke 训练

正式训练前使用独立目录 `runs/colab-gpu-smoke` 执行：

```bash
python -m ai train \
  --games 1 \
  --full-moves 1 \
  --simulations 1 \
  --channels 8 \
  --residual-blocks 1 \
  --batch-size 1 \
  --checkpoint-interval-games 1 \
  --device cuda:0
```

Smoke 完成后读取状态，并检查 checkpoint 和 `final_model.pt`。Notebook 不自动
删除 smoke 目录，以满足所有训练产物保留在 Drive 的要求。

## 后台训练和控制

正式训练通过 `subprocess.Popen` 后台启动，stdout/stderr 追加到 Drive 日志。
PID 写入 `colab-training.pid`。启动前：

1. 若 PID 文件存在，检查该 PID 是否仍属于本仓库的 `python -m ai` 进程；
2. 同一进程仍运行时拒绝重复启动；
3. PID 已失效时允许覆盖 PID 文件；
4. 运行目录存在有效 checkpoint 时，`train` 自动断点续传。

启动命令使用 Notebook 配置拼接参数，不使用 shell 字符串执行，避免路径或参数
转义问题。

控制单元格通过同步 `subprocess.run([...], check=True)` 调用：

- `python -m ai status --run-dir ...`
- `python -m ai pause --run-dir ...`
- `python -m ai extend --run-dir ... --games N`
- `python -m ai resume --run-dir ...`

`resume` 也以后台进程启动，否则单元格会一直占用。暂停后轮询 `status`，直到
`paused`、`completed` 或 `failed`，并设置最大等待时间。

## Colab 断线恢复

Colab runtime 断线时后台进程可能被终止，但 Drive 数据保留。重新连接后：

1. 从头运行挂载、目录、源码、安装、CUDA 检查和配置单元格；
2. 查询 `status`；
3. 检查旧 PID 是否仍有效；
4. 运行“恢复训练”单元格；
5. 训练器从 Drive 最近有效 checkpoint 和 Replay 继续。

Notebook 明确提示 Colab 突然断线可能发生在 checkpoint 间隔内；训练系统会协调
Replay 前进状态并重新保存一致 checkpoint。

## 模型验证

模型检查单元格使用安全加载：

```python
torch.load(path, map_location="cpu", weights_only=True)
```

输出 checkpoint 槽、`final_model.pt`、Replay manifest、状态文件和日志的存在性与
大小。只有达到当前累计目标后才要求 `final_model.pt` 存在；训练中以双槽
checkpoint 为准。

## 安全边界

- 不在 Notebook 内保存 Google 凭据或 GitHub token；
- 不使用 `weights_only=False`；
- 不自动删除 Drive 文件；
- 不使用 `git reset --hard`、强制 checkout 或强制 pull；
- 不允许同一运行目录同时启动两个训练器；
- 不承诺 Colab 免费运行时可连续完成 10,000 局；
- Drive I/O 比 Colab 本地盘慢，这是方案 B 的明确取舍。

## 验收标准

- `.ipynb` 是合法的 nbformat 4 JSON，可被 Jupyter/Colab 打开；
- 所有代码单元格可以按顺序重复执行；
- 源码、临时文件、缓存、日志、Replay 和模型路径都在 Drive 的
  `MyDrive/XiangQi-AI/` 下；
- 默认配置为 CUDA、10,000 局和 512 完整回合；
- 包含 CUDA 强校验、smoke、后台训练、状态、暂停、追加、恢复和模型验证；
- 断线后可以从 Drive checkpoint 继续；
- README 和完整训练手册链接到 Notebook；
- Notebook 不修改训练核心和现有桌面程序。
