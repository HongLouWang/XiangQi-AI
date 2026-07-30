"""Repetition and prohibited-move adjudication.

This module deliberately keeps position-cycle detection separate from each
ruleset's responsibility table.  The first increment covers checks and quiet
moves; chase, mate-threat, exchange and sacrifice classification are added by
extending :class:`MoveNature`, without changing stored position frames.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from xiangqi.board import Board
from xiangqi.domain import Color, Move
from xiangqi.rules import is_in_check


class Ruleset(StrEnum):
    CHINESE_2020 = "chinese_2020"
    ASIAN_2003 = "asian_2003"


class MoveNature(StrEnum):
    CHECK = "check"
    KILL = "kill"
    CHASE = "chase"
    EXCHANGE = "exchange"
    SACRIFICE = "sacrifice"
    BLOCK = "block"
    FOLLOW = "follow"
    IDLE = "idle"


class AdjudicationKind(StrEnum):
    MUST_CHANGE = "must_change"
    LOSS = "loss"
    DRAW = "draw"
    NO_DECISION = "no_decision"


@dataclass(frozen=True, slots=True)
class PositionFrame:
    """One actual transition, including enough state to reclassify it later."""

    board_before: Board
    board_after: Board
    side: Color
    move: Move
    before_key: str
    after_key: str
    nature: MoveNature

    @classmethod
    def from_transition(
        cls,
        board_before: Board,
        side: Color,
        move: Move,
        board_after: Board,
    ) -> PositionFrame:
        """Build and validate a frame from a real board transition."""
        if move.piece.color is not side:
            raise ValueError("着法棋子与行动方不一致")
        if board_before.at(move.start) != move.piece:
            raise ValueError("走前局面的起点棋子与着法不一致")
        if board_before.at(move.end) != move.captured:
            raise ValueError("走前局面的目标棋子与吃子记录不一致")
        expected = board_before.move_unchecked(move.start, move.end)
        if expected.pieces != board_after.pieces:
            raise ValueError("走后局面不是所给着法的结果")

        nature = (
            MoveNature.CHECK
            if is_in_check(board_after, side.opponent)
            else MoveNature.IDLE
        )
        return cls(
            board_before=board_before,
            board_after=board_after,
            side=side,
            move=move,
            before_key=board_before.position_key(side),
            after_key=board_after.position_key(side.opponent),
            nature=nature,
        )


@dataclass(frozen=True, slots=True)
class Adjudication:
    kind: AdjudicationKind
    ruleset: Ruleset
    cycle_start: int | None
    responsible: Color | None
    move_natures: tuple[MoveNature, ...]
    responsible_natures: tuple[MoveNature, ...]
    rule_reference: str
    reason: str


class RuleAdjudicator(Protocol):
    ruleset: Ruleset

    def evaluate(self, history: Sequence[PositionFrame]) -> Adjudication: ...


@dataclass(frozen=True, slots=True)
class _Cycle:
    start: int
    frames: tuple[PositionFrame, ...]


def _find_threefold_cycle(history: Sequence[PositionFrame]) -> _Cycle | None:
    """Return the shortest latest cycle whose boundary position occurs 3 times."""
    if not history:
        return None
    keys = [history[0].before_key, *(frame.after_key for frame in history)]
    end = len(keys) - 1
    candidates: list[tuple[int, int]] = []
    for cycle_length in range(1, (end // 2) + 1):
        middle = end - cycle_length
        start = end - 2 * cycle_length
        if keys[start] == keys[middle] == keys[end]:
            candidates.append((cycle_length, start))
    if not candidates:
        return None
    cycle_length, start = min(candidates)
    # Both repetitions matter when determining whether a side continually used
    # a prohibited kind of move.
    return _Cycle(start, tuple(history[start : start + 2 * cycle_length]))


def _single_perpetual_checker(cycle: _Cycle) -> Color | None:
    by_side = {
        color: tuple(frame.nature for frame in cycle.frames if frame.side is color)
        for color in Color
    }
    checking = [
        color
        for color, natures in by_side.items()
        if natures and all(nature is MoveNature.CHECK for nature in natures)
    ]
    return checking[0] if len(checking) == 1 else None


def _no_decision(ruleset: Ruleset, reference: str) -> Adjudication:
    return Adjudication(
        kind=AdjudicationKind.NO_DECISION,
        ruleset=ruleset,
        cycle_start=None,
        responsible=None,
        move_natures=(),
        responsible_natures=(),
        rule_reference=reference,
        reason="尚未形成三次相同局面的循环",
    )


def _result_for_cycle(
    cycle: _Cycle,
    ruleset: Ruleset,
    reference: str,
    perpetual_check_reason: str,
    repetition_reason: str,
) -> Adjudication:
    natures = tuple(frame.nature for frame in cycle.frames)
    responsible = _single_perpetual_checker(cycle)
    if responsible is not None:
        responsible_natures = tuple(
            frame.nature for frame in cycle.frames if frame.side is responsible
        )
        return Adjudication(
            kind=AdjudicationKind.MUST_CHANGE,
            ruleset=ruleset,
            cycle_start=cycle.start,
            responsible=responsible,
            move_natures=natures,
            responsible_natures=responsible_natures,
            rule_reference=reference,
            reason=perpetual_check_reason,
        )
    return Adjudication(
        kind=AdjudicationKind.DRAW,
        ruleset=ruleset,
        cycle_start=cycle.start,
        responsible=None,
        move_natures=natures,
        responsible_natures=(),
        rule_reference=reference,
        reason=repetition_reason,
    )


class Chinese2020Adjudicator:
    ruleset = Ruleset.CHINESE_2020
    _reference = "中国棋规2020 第24至26条（棋例术语、循环与禁止着法）"

    def evaluate(self, history: Sequence[PositionFrame]) -> Adjudication:
        cycle = _find_threefold_cycle(history)
        if cycle is None:
            return _no_decision(self.ruleset, self._reference)
        return self._evaluate_cycle(cycle)

    def _evaluate_cycle(self, cycle: _Cycle) -> Adjudication:
        return _result_for_cycle(
            cycle,
            self.ruleset,
            self._reference,
            "单方在循环中每着将军，依中国棋规须由长将方变着",
            "三次相同局面且基础分类中双方均无禁止着法，判和",
        )


class Asian2003Adjudicator:
    ruleset = Ruleset.ASIAN_2003
    _reference = "亚洲棋规2003 循环局面及禁止着法条款"

    def evaluate(self, history: Sequence[PositionFrame]) -> Adjudication:
        cycle = _find_threefold_cycle(history)
        if cycle is None:
            return _no_decision(self.ruleset, self._reference)
        return self._evaluate_cycle(cycle)

    def _evaluate_cycle(self, cycle: _Cycle) -> Adjudication:
        return _result_for_cycle(
            cycle,
            self.ruleset,
            self._reference,
            "单方连续将军构成长将，依亚洲棋规须由长将方变着",
            "同一局面出现三次且基础分类中双方均无禁止着法，判和",
        )
