"""Repetition and prohibited-move adjudication.

This module deliberately keeps position-cycle detection separate from each
ruleset's responsibility table.  The first increment covers checks and quiet
moves; chase, mate-threat, exchange and sacrifice classification are added by
extending :class:`MoveNature`, without changing stored position frames.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


def _capture_is_rooted(
    board: Board, attacker: Coord, target: Coord, side: Color
) -> bool:
    """Whether a legal recapture protects target (pinned pseudo-roots do not)."""
    after_capture = board.move_unchecked(attacker, target)
    return any(
        move.end == target for move in all_legal_moves(after_capture, side.opponent)
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
            attack.attacker_piece.kind not in (PieceType.GENERAL, PieceType.PAWN)
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

    def evaluate_natures(
        self, by_side: Mapping[Color, Sequence[MoveNature]]
    ) -> Adjudication: ...

    def loss_for_ignored_must_change(self, decision: Adjudication) -> Adjudication: ...


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


def _labels(natures: Sequence[MoveNature]) -> str:
    labels = {
        MoveNature.CHECK: "将",
        MoveNature.CHASE: "捉",
        MoveNature.KILL: "杀",
    }
    unique = tuple(
        dict.fromkeys(labels[nature] for nature in natures if nature in labels)
    )
    if len(unique) == 1:
        return f"长{unique[0]}"
    return "".join(f"一{label}" for label in unique).removeprefix("一")


def _make_result(
    *,
    by_side: Mapping[Color, Sequence[MoveNature]],
    cycle_start: int | None,
    ruleset: Ruleset,
    reference: str,
    kind: AdjudicationKind,
    responsible: Color | None,
    reason: str,
    ordered_natures: Sequence[MoveNature] | None = None,
) -> Adjudication:
    flattened = (
        tuple(ordered_natures)
        if ordered_natures is not None
        else tuple(
            nature
            for index in range(max(map(len, by_side.values()), default=0))
            for color in Color
            for nature in by_side.get(color, ())[index : index + 1]
        )
    )
    return Adjudication(
        kind=kind,
        ruleset=ruleset,
        cycle_start=cycle_start,
        responsible=responsible,
        move_natures=flattened,
        responsible_natures=(
            tuple(by_side.get(responsible, ())) if responsible is not None else ()
        ),
        rule_reference=reference,
        reason=reason,
    )


def _chinese_responsibility(natures: Sequence[MoveNature]) -> int:
    """2020 rules: every all-aggressive check/kill/chase cycle is prohibited."""
    if not natures or any(
        nature not in {MoveNature.CHECK, MoveNature.KILL, MoveNature.CHASE}
        for nature in natures
    ):
        return 0
    return 2 if MoveNature.CHECK in natures else 1


def _asian_responsibility(natures: Sequence[MoveNature]) -> int:
    """2003 rules: long check/chase are prohibited; mixed attacks and kills draw."""
    if natures and all(nature is MoveNature.CHECK for nature in natures):
        return 2
    if natures and all(nature is MoveNature.CHASE for nature in natures):
        return 1
    return 0


def _evaluate_responsibility(
    by_side: Mapping[Color, Sequence[MoveNature]],
    ruleset: Ruleset,
    reference: str,
    cycle_start: int | None,
    ordered_natures: Sequence[MoveNature] | None = None,
) -> Adjudication:
    """Apply the formal responsibility table after tactical classification."""
    normalized = {color: tuple(by_side.get(color, ())) for color in Color}
    if not all(normalized.values()):
        raise ValueError("双方均须提供循环中的着法性质")
    classifier = (
        _chinese_responsibility
        if ruleset is Ruleset.CHINESE_2020
        else _asian_responsibility
    )
    scores = {color: classifier(natures) for color, natures in normalized.items()}
    offenders = [color for color, score in scores.items() if score]
    if not offenders:
        return _make_result(
            by_side=normalized,
            cycle_start=cycle_start,
            ruleset=ruleset,
            reference=reference,
            kind=AdjudicationKind.DRAW,
            responsible=None,
            reason="双方均为允许着法（含闲着或规则允许的混合攻击），判和",
            ordered_natures=ordered_natures,
        )
    if len(offenders) == 2 and scores[offenders[0]] == scores[offenders[1]]:
        return _make_result(
            by_side=normalized,
            cycle_start=cycle_start,
            ruleset=ruleset,
            reference=reference,
            kind=AdjudicationKind.DRAW,
            responsible=None,
            reason="双方同时走出同等责任的禁止着法，双方不变判和",
            ordered_natures=ordered_natures,
        )
    responsible = max(offenders, key=scores.__getitem__)
    behavior = _labels(normalized[responsible])
    return _make_result(
        by_side=normalized,
        cycle_start=cycle_start,
        ruleset=ruleset,
        reference=reference,
        kind=AdjudicationKind.MUST_CHANGE,
        responsible=responsible,
        reason=f"责任方循环构成{behavior}类禁止着法，须由责任方变着",
        ordered_natures=ordered_natures,
    )


class _AdjudicatorBase:
    ruleset: Ruleset
    _reference: str

    def evaluate_natures(
        self, by_side: Mapping[Color, Sequence[MoveNature]]
    ) -> Adjudication:
        """Adjudicate an already verified cycle classification.

        This public seam lets importers and tournament integrations apply the
        responsibility table without manufacturing board histories.
        """
        return _evaluate_responsibility(
            by_side, self.ruleset, self._reference, cycle_start=None
        )

    def loss_for_ignored_must_change(self, decision: Adjudication) -> Adjudication:
        """Escalate a referee/controller MUST_CHANGE notice after noncompliance."""
        if decision.ruleset is not self.ruleset:
            raise ValueError("裁决规则模式不一致")
        if (
            decision.kind is not AdjudicationKind.MUST_CHANGE
            or decision.responsible is None
        ):
            raise ValueError("只有未执行的变着裁决才能升级为判负")
        return Adjudication(
            kind=AdjudicationKind.LOSS,
            ruleset=decision.ruleset,
            cycle_start=decision.cycle_start,
            responsible=decision.responsible,
            move_natures=decision.move_natures,
            responsible_natures=decision.responsible_natures,
            rule_reference=decision.rule_reference,
            reason=f"{decision.reason}；经要求后仍未变着，责任方判负",
        )


class Chinese2020Adjudicator(_AdjudicatorBase):
    ruleset = Ruleset.CHINESE_2020
    _reference = "中国棋规2020 第24至26条（棋例术语、循环与禁止着法）"

    def evaluate(self, history: Sequence[PositionFrame]) -> Adjudication:
        _validate_history(history)
        cycle = _find_threefold_cycle(history)
        if cycle is None:
            return _no_decision(self.ruleset, self._reference)
        return self._evaluate_cycle(cycle)

    def _evaluate_cycle(self, cycle: _Cycle) -> Adjudication:
        return _evaluate_responsibility(
            _natures_by_side(cycle),
            self.ruleset,
            self._reference,
            cycle.start,
            tuple(frame.nature for frame in cycle.frames),
        )


class Asian2003Adjudicator(_AdjudicatorBase):
    ruleset = Ruleset.ASIAN_2003
    _reference = "亚洲棋规2003 循环局面及禁止着法条款"

    def evaluate(self, history: Sequence[PositionFrame]) -> Adjudication:
        _validate_history(history)
        cycle = _find_threefold_cycle(history)
        if cycle is None:
            return _no_decision(self.ruleset, self._reference)
        return self._evaluate_cycle(cycle)

    def _evaluate_cycle(self, cycle: _Cycle) -> Adjudication:
        return _evaluate_responsibility(
            _natures_by_side(cycle),
            self.ruleset,
            self._reference,
            cycle.start,
            tuple(frame.nature for frame in cycle.frames),
        )
