import pytest

from ai.config import TrainingConfig


def test_defaults_are_10000_games_and_512_full_moves() -> None:
    config = TrainingConfig()
    assert config.target_games == 10_000
    assert config.max_full_moves == 512
    assert config.max_plies == 1024


def test_training_scale_can_be_changed_from_python() -> None:
    config = TrainingConfig(target_games=25, max_full_moves=7)
    assert config.target_games == 25
    assert config.max_plies == 14


@pytest.mark.parametrize(
    "field",
    [
        "target_games",
        "max_full_moves",
        "torch_threads",
        "self_play_workers",
        "simulations_per_move",
        "residual_blocks",
        "channels",
        "batch_size",
        "replay_capacity_games",
        "checkpoint_interval_games",
    ],
)
@pytest.mark.parametrize("value", [0, 1.5, True])
def test_positive_integer_configuration_is_required(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**{field: value})
