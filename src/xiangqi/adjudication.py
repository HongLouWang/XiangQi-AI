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
from xiangqi.domain import Color, Coord, Move, Piece, PieceType, PositionKind
from xiangqi.rules import (
    all_legal_moves,
    evaluate_position,
    is_in_check,
    legal_destinations,
)


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
    UNSUPPORTED = "unsupported"


_PIECE_VALUE = {
    PieceType.GENERAL: 1000,
    PieceType.ROOK: 9,
    PieceType.CANNON: 5,
    PieceType.HORSE: 4,
    PieceType.ELEPHANT: 2,
    PieceType.ADVISOR: 2,
    PieceType.PAWN: 1,
}


def _capture_is_rooted(board: Board, attacker: Coord, target: Coord, side: Color) -> bool:
    """Whether a legal recapture protects target (pinned pseudo-roots do not)."""
    after_capture = board.move_unchecked(attacker, target)
    return any(
        move.end == target
        for move in all_legal_moves(after_capture, side.opponent)
    )


@dataclass(frozen=True, slots=True)
class TacticalAttack:
    attacker: Coord
    attacker_piece: Piece
    target: Coord
    target_piece: Piece
    attacker_value: int
    target_value: int
    rooted: bool


def _legal_capture_pairs(board: Board, side: Color) -> set[tuple[Coord, Coord]]:
    """Return legal attacker/target pairs, excluding empty destinations."""
    pairs: set[tuple[Coord, Coord]] = set()
    for attacker, piece in board.pieces.items():
        if piece.color is not side:
            continue
        for target in legal_destinations(board, attacker, side):
            victim = board.at(target)
            if victim is not None and victim.color is side.opponent:
                pairs.add((attacker, target))
    return pairs


def _new_attacks(
    board_before: Board, board_after: Board, side: Color, move: Move
) -> tuple[TacticalAttack, ...]:
    before_attacks = _legal_capture_pairs(board_before, side)
    attacks: list[TacticalAttack] = []
    for attacker, target in sorted(
        _legal_capture_pairs(board_after, side) - before_attacks,
        key=lambda pair: (
            pair[0].rank,
            pair[0].file,
            pair[1].rank,
            pair[1].file,
        ),
    ):
        attacker_piece = board_after.at(attacker)
        victim = board_after.at(target)
        assert attacker_piece is not None and victim is not None
        attacks.append(
            TacticalAttack(
                attacker=attacker,
                attacker_piece=attacker_piece,
                target=target,
                target_piece=victim,
                attacker_value=_PIECE_VALUE[attacker_piece.kind],
                target_value=_PIECE_VALUE[victim.kind],
                rooted=_capture_is_rooted(board_after, attacker, target, side),
            )
        )
    return tuple(attacks)


def _has_forced_mate_next_move(board: Board, attacker: Color) -> bool:
    """Two-ply kill: every legal defence still permits an immediate mate."""
    defender = attacker.opponent
    replies = all_legal_moves(board, defender)
    if not replies:
        return False
    for reply in replies:
        replied = board.move_unchecked(reply.start, reply.end)
        mate_available = any(
            evaluate_position(
                replied.move_unchecked(candidate.start, candidate.end), defender
            ).kind
            is PositionKind.CHECKMATE
            for candidate in all_legal_moves(replied, attacker)
        )
        if not mate_available:
            return False
    return True


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
    attacks: tuple[TacticalAttack, ...] = ()

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
        if move.end not in legal_destinations(board_before, move.start, side):
            raise ValueError("非法着法：该棋子不能走到目标位置")
        expected = board_before.move_unchecked(move.start, move.end)
        if expected.pieces != board_after.pieces:
            raise ValueError("走后局面不是所给着法的结果")

        attacks = _new_attacks(board_before, board_after, side, move)
        if is_in_check(board_after, side.opponent):
            nature = MoveNature.CHECK
        elif _has_forced_mate_next_move(board_after, side):
            nature = MoveNature.KILL
        elif any(
            attack.attacker_piece.kind
            not in (PieceType.GENERAL, PieceType.PAWN)
            and (not attack.rooted or attack.target_value > attack.attacker_value)
            for attack in attacks
        ):
            nature = MoveNature.CHASE
        else:
            nature = MoveNature.IDLE
        return cls(
            board_before=board_before,
            board_after=board_after,
            side=side,
            move=move,
            before_key=board_before.position_key(side),
            after_key=board_after.position_key(side.opponent),
            nature=nature,
            attacks=attacks,
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


def _validate_history(history: Sequence[PositionFrame]) -> None:
    """Reject forged classifications and discontinuous transition histories."""
    previous: PositionFrame | None = None
    for frame in history:
        rebuilt = PositionFrame.from_transition(
            frame.board_before,
            frame.side,
            frame.move,
            frame.board_after,
        )
        if (
            frame.before_key != rebuilt.before_key
            or frame.after_key != rebuilt.after_key
            or frame.nature is not rebuilt.nature
            or frame.attacks != rebuilt.attacks
        ):
            raise ValueError("棋史中的着法性质或局面摘要与实际转换不一致")
        if previous is not None:
            if previous.board_after.pieces != frame.board_before.pieces:
                raise ValueError("棋史中的相邻局面不连续")
            if frame.side is not previous.side.opponent:
                raise ValueError("棋史中的行动方没有交替")
        previous = frame


def _natures_by_side(cycle: _Cycle) -> dict[Color, tuple[MoveNature, ...]]:
    return {
        color: tuple(frame.nature for frame in cycle.frames if frame.side is color)
        for color in Color
    }


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
    repetition_reason: str,
    prohibited,
) -> Adjudication:
    natures = tuple(frame.nature for frame in cycle.frames)
    by_side = _natures_by_side(cycle)
    offenders = [color for color, kinds in by_side.items() if prohibited(kinds)]
    if len(offenders) == 1:
        responsible = offenders[0]
        responsible_natures = tuple(
            frame.nature for frame in cycle.frames if frame.side is responsible
        )
        labels = {
            MoveNature.CHECK: "长将",
            MoveNature.CHASE: "长捉",
            MoveNature.KILL: "长杀",
        }
        behavior = "、".join(dict.fromkeys(labels[kind] for kind in responsible_natures))
        return Adjudication(
            kind=AdjudicationKind.MUST_CHANGE,
            ruleset=ruleset,
            cycle_start=cycle.start,
            responsible=responsible,
            move_natures=natures,
            responsible_natures=responsible_natures,
            rule_reference=reference,
            reason=f"单方循环构成{behavior}，须由责任方变着",
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


def _unsupported(
    cycle: _Cycle, ruleset: Ruleset, reference: str, reason: str
) -> Adjudication:
    return Adjudication(
        kind=AdjudicationKind.UNSUPPORTED,
        ruleset=ruleset,
        cycle_start=cycle.start,
        responsible=None,
        move_natures=tuple(frame.nature for frame in cycle.frames),
        responsible_natures=(),
        rule_reference=reference,
        reason=reason,
    )


def _evaluate_supported_cycle(
    cycle: _Cycle,
    ruleset: Ruleset,
    reference: str,
    *,
    allow_kill: bool,
) -> Adjudication:
    """Apply only responsibility rows represented by verified classifiers."""
    by_side = _natures_by_side(cycle)
    if all(frame.nature is MoveNature.IDLE for frame in cycle.frames):
        return _result_for_cycle(
            cycle,
            ruleset,
            reference,
            "三次相同局面且双方均为闲着，判和",
            lambda _natures: False,
        )

    supported = {MoveNature.CHECK, MoveNature.CHASE}
    if allow_kill:
        supported.add(MoveNature.KILL)
    offenders = [
        color
        for color, natures in by_side.items()
        if natures and len(set(natures)) == 1 and natures[0] in supported
    ]
    quiet = [
        color
        for color, natures in by_side.items()
        if natures and all(nature is MoveNature.IDLE for nature in natures)
    ]
    if len(offenders) == 1 and len(quiet) == 1:
        prohibited = by_side[offenders[0]][0]
        return _result_for_cycle(
            cycle,
            ruleset,
            reference,
            "",
            lambda natures: bool(natures)
            and all(nature is prohibited for nature in natures),
        )
    return _unsupported(
        cycle,
        ruleset,
        reference,
        "该循环包含混合着法或双方责任棋例，当前受控子集不作自动裁决",
    )


class Chinese2020Adjudicator:
    ruleset = Ruleset.CHINESE_2020
    _reference = "中国棋规2020 第24至26条（棋例术语、循环与禁止着法）"

    def evaluate(self, history: Sequence[PositionFrame]) -> Adjudication:
        _validate_history(history)
        cycle = _find_threefold_cycle(history)
        if cycle is None:
            return _no_decision(self.ruleset, self._reference)
        return self._evaluate_cycle(cycle)

    def _evaluate_cycle(self, cycle: _Cycle) -> Adjudication:
        return _evaluate_supported_cycle(
            cycle,
            self.ruleset,
            self._reference,
            allow_kill=True,
        )


class Asian2003Adjudicator:
    ruleset = Ruleset.ASIAN_2003
    _reference = "亚洲棋规2003 循环局面及禁止着法条款"

    def evaluate(self, history: Sequence[PositionFrame]) -> Adjudication:
        _validate_history(history)
        cycle = _find_threefold_cycle(history)
        if cycle is None:
            return _no_decision(self.ruleset, self._reference)
        return self._evaluate_cycle(cycle)

    def _evaluate_cycle(self, cycle: _Cycle) -> Adjudication:
        return _evaluate_supported_cycle(
            cycle,
            self.ruleset,
            self._reference,
            allow_kill=False,
        )
