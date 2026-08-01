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
    "field,value",
    [
        ("target_games", 0),
        ("max_full_moves", 0),
        ("self_play_workers", 0),
        ("torch_threads", 0),
    ],
)
def test_positive_configuration_is_required(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**{field: value})
