from __future__ import annotations

import numpy as np

from ai.batched_self_play import BatchedSelfPlay
from ai.encoding import ACTION_SIZE


class RecordingEvaluator:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def evaluate_many(self, states):
        self.batch_sizes.append(len(states))
        return (
            np.zeros((len(states), ACTION_SIZE), dtype=np.float32),
            np.zeros(len(states), dtype=np.float32),
        )


def test_two_games_share_batched_network_calls_and_remain_independent() -> None:
    evaluator = RecordingEvaluator()
    scheduler = BatchedSelfPlay(
        evaluator,
        simulations=1,
        max_plies=1,
        seed=7,
    )

    games = scheduler.generate(count=2, parallel_games=2)

    assert len(games) == 2
    assert evaluator.batch_sizes == [2, 2]
    assert all(game.plies == 1 and game.termination == "move_limit" for game in games)
    assert games[0].samples is not games[1].samples


def test_scheduler_refills_slots_until_requested_game_count() -> None:
    evaluator = RecordingEvaluator()
    scheduler = BatchedSelfPlay(evaluator, simulations=1, max_plies=1, seed=11)

    games = scheduler.generate(count=3, parallel_games=2)

    assert len(games) == 3
    assert scheduler.last_batch_size in {1, 2}
