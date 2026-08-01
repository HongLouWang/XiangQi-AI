import pytest

from ai.config import TrainingConfig


def test_defaults_are_10000_games_and_512_full_moves() -> None:
    config = TrainingConfig()
    assert config.target_games == 10_000
    assert config.max_full_moves == 512
    assert config.max_plies == 1024
    assert config.game_retry_limit == 2


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


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_game_retry_limit_is_a_nonnegative_integer(value: object) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(game_retry_limit=value)  # type: ignore[arg-type]


def test_game_retry_limit_can_disable_retries() -> None:
    assert TrainingConfig(game_retry_limit=0).game_retry_limit == 0
