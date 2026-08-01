from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    target_games: int = 10_000
    max_full_moves: int = 512
    device: str = "auto"
    torch_threads: int = 1
    self_play_workers: int = 1
    simulations_per_move: int = 64
    residual_blocks: int = 4
    channels: int = 64
    batch_size: int = 128
    replay_capacity_games: int = 2_000
    learning_rate: float = 1e-3
    checkpoint_interval_games: int = 10
    seed: int = 0
    run_dir: Path = Path("AI-runs/default")

    def __post_init__(self) -> None:
        positive = {
            "target_games": self.target_games,
            "max_full_moves": self.max_full_moves,
            "torch_threads": self.torch_threads,
            "self_play_workers": self.self_play_workers,
            "simulations_per_move": self.simulations_per_move,
            "batch_size": self.batch_size,
            "replay_capacity_games": self.replay_capacity_games,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0")

    @property
    def max_plies(self) -> int:
        return self.max_full_moves * 2
