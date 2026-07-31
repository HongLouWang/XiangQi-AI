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


@dataclass(frozen=True, slots=True)
class ClassificationEvidence:
    """Auditable board coordinates supporting a move-nature classification."""

    nature: MoveNature
    actors: tuple[Coord, ...] = ()
    targets: tuple[Coord, ...] = ()
    rationale: str = ""


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


def _exchange_targets(
    board_after: Board, side: Color, move: Move, attacks: Sequence[TacticalAttack]
) -> tuple[Coord, ...]:
    """Same-kind, mutually capturable contacts are invitations to exchange."""
    opponent_captures = _legal_capture_pairs(board_after, side.opponent)
    targets: list[Coord] = []
    for attack in attacks:
        if (
            attack.attacker != move.end
            or attack.attacker_piece.kind is not attack.target_piece.kind
            or (attack.target, move.end) not in opponent_captures
        ):
            continue
        accepted = board_after.move_unchecked(attack.target, move.end)
        if any(
            candidate.end == move.end
            for candidate in all_legal_moves(accepted, side)
        ):
            targets.append(attack.target)
    return tuple(targets)


def _blocked_targets(
    board_before: Board, board_after: Board, side: Color, move: Move
) -> tuple[Coord, ...]:
    """Targets whose opponent capture was stopped by the moved interposer."""
    before = _legal_capture_pairs(board_before, side.opponent)
    after = _legal_capture_pairs(board_after, side.opponent)
    return tuple(
        sorted(
            {
                target
                for attacker, target in before - after
                if target != move.start
                and board_after.at(attacker) is not None
                and board_after.at(target) is not None
            },
            key=lambda coord: (coord.rank, coord.file),
        )
    )


def _sacrifice_attackers(
    board_before: Board, board_after: Board, side: Color, move: Move
) -> tuple[Coord, ...]:
    """Opponent pieces which may legally accept the newly offered moved piece."""
    before = _legal_capture_pairs(board_before, side.opponent)
    return tuple(
        sorted(
            (
                attacker
                for attacker, target in _legal_capture_pairs(
                    board_after, side.opponent
                )
                if target == move.end
                and (attacker, move.start) not in before
            ),
            key=lambda coord: (coord.rank, coord.file),
        )
    )


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
    evidence: ClassificationEvidence = ClassificationEvidence(MoveNature.IDLE)

    @classmethod
    def from_transition(
        cls,
        board_before: Board,
        side: Color,
        move: Move,
        board_after: Board,
        *,
        analyze_kill: bool = True,
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
        exchange_targets = _exchange_targets(board_after, side, move, attacks)
        blocked_targets = _blocked_targets(board_before, board_after, side, move)
        sacrifice_attackers = _sacrifice_attackers(
            board_before, board_after, side, move
        )
        followed_targets = tuple(
            attack.target for attack in attacks if attack.rooted
        )
        if is_in_check(board_after, side.opponent):
            nature = MoveNature.CHECK
            evidence = ClassificationEvidence(
                nature, actors=(move.end,), rationale="走后直接攻击对方将帅"
            )
        elif analyze_kill and _has_forced_mate_next_move(board_after, side):
            nature = MoveNature.KILL
            evidence = ClassificationEvidence(
                nature, actors=(move.end,), rationale="对方所有应着后均存在一步将死"
            )
        elif exchange_targets:
            nature = MoveNature.EXCHANGE
            evidence = ClassificationEvidence(
                nature,
                actors=(move.end,),
                targets=exchange_targets,
                rationale="同兵种互相可吃，形成邀兑",
            )
        elif any(_qualifies_as_chase(attack) for attack in attacks):
            nature = MoveNature.CHASE
            evidence = ClassificationEvidence(
                nature,
                actors=tuple(dict.fromkeys(a.attacker for a in attacks)),
                targets=tuple(
                    attack.target
                    for attack in attacks
                    if _qualifies_as_chase(attack)
                ),
                rationale="产生可得子的新增合法攻击",
            )
        elif blocked_targets:
            nature = MoveNature.BLOCK
            evidence = ClassificationEvidence(
                nature,
                actors=(move.end,),
                targets=blocked_targets,
                rationale="走子阻断对方原有合法吃子线路",
            )
        elif followed_targets:
            nature = MoveNature.FOLLOW
            evidence = ClassificationEvidence(
                nature,
                actors=tuple(dict.fromkeys(a.attacker for a in attacks)),
                targets=followed_targets,
                rationale="新增盯牵有根子但不构成得子",
            )
        elif sacrifice_attackers:
            nature = MoveNature.SACRIFICE
            evidence = ClassificationEvidence(
                nature,
                actors=sacrifice_attackers,
                targets=(move.end,),
                rationale="所走棋子进入对方合法吃子范围",
            )
        else:
            nature = MoveNature.IDLE
            evidence = ClassificationEvidence(nature, rationale="未形成将杀捉兑献拦跟")
        return cls(
            board_before=board_before,
            board_after=board_after,
            side=side,
            move=move,
            before_key=board_before.position_key(side),
            after_key=board_after.position_key(side.opponent),
            nature=nature,
            attacks=attacks,
            evidence=evidence,
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
            analyze_kill=frame.nature is MoveNature.KILL,
        )
        if (
            frame.before_key != rebuilt.before_key
            or frame.after_key != rebuilt.after_key
            or frame.nature is not rebuilt.nature
            or frame.attacks != rebuilt.attacks
            or frame.evidence != rebuilt.evidence
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


def _effective_cycle_natures(cycle: _Cycle) -> tuple[MoveNature, ...]:
    """Complete expensive kill classification only for an actual cycle."""
    return tuple(
        (
            MoveNature.KILL
            if frame.nature is MoveNature.IDLE
            and _has_forced_mate_next_move(frame.board_after, frame.side)
            else frame.nature
        )
        for frame in cycle.frames
    )


def _effective_natures_by_side(
    cycle: _Cycle, effective: Sequence[MoveNature]
) -> dict[Color, tuple[MoveNature, ...]]:
    return {
        color: tuple(
            nature
            for frame, nature in zip(cycle.frames, effective, strict=True)
            if frame.side is color
        )
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


class _CyclePattern(StrEnum):
    ALLOWED = "allowed"
    CHECK = "check"
    KILL = "kill"
    CHASE = "chase"
    CHECK_KILL = "check_kill"
    CHECK_CHASE = "check_chase"
    KILL_CHASE = "kill_chase"
    CHECK_KILL_CHASE = "check_kill_chase"
    JOINT_CHASE = "joint_chase"


_PATTERN_BY_ATTACKS = {
    frozenset({MoveNature.CHECK}): _CyclePattern.CHECK,
    frozenset({MoveNature.KILL}): _CyclePattern.KILL,
    frozenset({MoveNature.CHASE}): _CyclePattern.CHASE,
    frozenset({MoveNature.CHECK, MoveNature.KILL}): _CyclePattern.CHECK_KILL,
    frozenset({MoveNature.CHECK, MoveNature.CHASE}): _CyclePattern.CHECK_CHASE,
    frozenset({MoveNature.KILL, MoveNature.CHASE}): _CyclePattern.KILL_CHASE,
    frozenset(
        {MoveNature.CHECK, MoveNature.KILL, MoveNature.CHASE}
    ): _CyclePattern.CHECK_KILL_CHASE,
}
_CHECK_PATTERNS = frozenset(
    {
        _CyclePattern.CHECK,
        _CyclePattern.CHECK_KILL,
        _CyclePattern.CHECK_CHASE,
        _CyclePattern.CHECK_KILL_CHASE,
    }
)
_KILL_WITHOUT_CHECK_PATTERNS = frozenset(
    {_CyclePattern.KILL, _CyclePattern.KILL_CHASE}
)
_CHASE_WITHOUT_CHECK_PATTERNS = frozenset(
    {
        _CyclePattern.CHASE,
        _CyclePattern.JOINT_CHASE,
        _CyclePattern.KILL_CHASE,
    }
)


@dataclass(frozen=True, slots=True)
class _CycleProfile:
    pattern: _CyclePattern
    target_kinds: frozenset[PieceType] = frozenset()
    every_chase_frame_targets_rook: bool = False
    every_chase_frame_targets_unrooted: bool = False


def _qualifies_as_chase(attack: TacticalAttack) -> bool:
    return (
        attack.attacker_piece.kind
        not in (PieceType.GENERAL, PieceType.PAWN)
        and (not attack.rooted or attack.target_value > attack.attacker_value)
    )


def _has_indispensable_support(
    frame: PositionFrame, attack: TacticalAttack
) -> bool:
    """Counterfactually prove that another friendly piece enables this chase."""
    for supporter, piece in frame.board_after.pieces.items():
        if (
            piece.color is not frame.side
            or piece.kind is PieceType.GENERAL
            or supporter == attack.attacker
        ):
            continue
        without_support = frame.board_after.remove(supporter)
        if is_in_check(without_support, frame.side):
            # Removing a shield of one's own general is not evidence that the
            # piece participates in capturing the target.
            continue
        if (attack.attacker, attack.target) not in _legal_capture_pairs(
            without_support, frame.side
        ):
            return True
        revised = TacticalAttack(
            attacker=attack.attacker,
            attacker_piece=attack.attacker_piece,
            target=attack.target,
            target_piece=attack.target_piece,
            attacker_value=attack.attacker_value,
            target_value=attack.target_value,
            rooted=_capture_is_rooted(
                without_support,
                attack.attacker,
                attack.target,
                frame.side,
            ),
        )
        if not _qualifies_as_chase(revised):
            return True
    return False


def _cycle_pattern(natures: Sequence[MoveNature]) -> _CyclePattern:
    """Map every nature combination to one named, auditable table row."""
    attacks = frozenset(natures)
    if not natures or not attacks.issubset(
        {MoveNature.CHECK, MoveNature.KILL, MoveNature.CHASE}
    ):
        return _CyclePattern.ALLOWED
    return _PATTERN_BY_ATTACKS[attacks]


def _cycle_profile_with_evidence(
    frames: Sequence[PositionFrame], natures: Sequence[MoveNature]
) -> _CycleProfile:
    """Refine long chase into the 26.9 joint-chase table row from attack evidence."""
    pattern = _cycle_pattern(natures)
    if pattern is not _CyclePattern.CHASE:
        return _CycleProfile(pattern)
    chase_frames = tuple(
        frame for frame, nature in zip(frames, natures, strict=True)
        if nature is MoveNature.CHASE
    )
    qualifying_by_frame = tuple(
        tuple(attack for attack in frame.attacks if _qualifies_as_chase(attack))
        for frame in chase_frames
    )
    relevant = tuple(
        attack for attacks in qualifying_by_frame for attack in attacks
    )
    target_kinds = frozenset(attack.target_piece.kind for attack in relevant)
    every_rook = bool(qualifying_by_frame) and all(
        attacks
        and all(
            attack.target_piece.kind is PieceType.ROOK for attack in attacks
        )
        for attacks in qualifying_by_frame
    )
    every_unrooted = bool(qualifying_by_frame) and all(
        attacks and all(not attack.rooted for attack in attacks)
        for attacks in qualifying_by_frame
    )
    joint = bool(chase_frames) and all(
        any(_has_indispensable_support(frame, attack) for attack in attacks)
        or any(
            len(target_attacks) >= 2
            and all(attack.rooted for attack in target_attacks)
            for target in {attack.target for attack in attacks}
            if (
                target_attacks := tuple(
                    attack
                    for attack in attacks
                    if attack.target == target
                )
            )
        )
        for frame, attacks in zip(
            chase_frames, qualifying_by_frame, strict=True
        )
    )
    return _CycleProfile(
        _CyclePattern.JOINT_CHASE if joint else pattern,
        target_kinds,
        every_rook,
        every_unrooted,
    )


@dataclass(frozen=True, slots=True)
class _TableDecision:
    kind: AdjudicationKind
    responsible: Color | None
    reference: str
    reason: str


def _single_or_allowed_decision(
    profiles: Mapping[Color, _CycleProfile],
    *,
    ruleset: Ruleset,
) -> _TableDecision | None:
    prohibited = tuple(
        color
        for color, profile in profiles.items()
        if profile.pattern is not _CyclePattern.ALLOWED
    )
    if not prohibited:
        reference = (
            "中国棋规2020 表项25.2、24.8（兑献拦跟均属闲）"
            if ruleset is Ruleset.CHINESE_2020
            else "AXF 2003 Chapter 4 Table 4-B (both sides permitted/idle)"
        )
        return _TableDecision(
            AdjudicationKind.DRAW,
            None,
            reference,
            "双方循环均为允许着法或闲着，双方不变判和",
        )
    if len(prohibited) == 1:
        responsible = prohibited[0]
        pattern = profiles[responsible].pattern
        pattern_label = {
            _CyclePattern.CHECK: "长将",
            _CyclePattern.KILL: "长杀",
            _CyclePattern.CHASE: "长捉",
            _CyclePattern.CHECK_KILL: "一将一杀",
            _CyclePattern.CHECK_CHASE: "一将一捉",
            _CyclePattern.KILL_CHASE: "一杀一捉",
            _CyclePattern.CHECK_KILL_CHASE: "将杀捉组合",
            _CyclePattern.JOINT_CHASE: "联合长捉",
        }[pattern]
        reference = (
            (
                "中国棋规2020 表项25.1、24.13（单方长将或将类组合）"
                if pattern in _CHECK_PATTERNS
                else "中国棋规2020 表项25.3、24.13（单方禁止着法）"
            )
            if ruleset is Ruleset.CHINESE_2020
            else (
                "AXF 2003 Chapter 4 Table 4-A "
                f"(single-side {pattern.value}: check/chase/kill-TTC)"
            )
        )
        return _TableDecision(
            AdjudicationKind.MUST_CHANGE,
            responsible,
            reference,
            f"{responsible.value}方循环构成{pattern_label}禁止着法，须变着",
        )
    return None


def _chinese_bilateral_decision(
    profiles: Mapping[Color, _CycleProfile],
) -> _TableDecision:
    red_profile = profiles[Color.RED]
    black_profile = profiles[Color.BLACK]
    red = red_profile.pattern
    black = black_profile.pattern
    if (red in _CHECK_PATTERNS) != (black in _CHECK_PATTERNS):
        responsible = Color.RED if red in _CHECK_PATTERNS else Color.BLACK
        return _TableDecision(
            AdjudicationKind.MUST_CHANGE,
            responsible,
            "中国棋规2020 表项25.1、26.9.1（长将优先变着）",
            "双方均有禁止着法，长将或将类组合方须变着",
        )
    if (
        red in _KILL_WITHOUT_CHECK_PATTERNS
    ) != (black in _KILL_WITHOUT_CHECK_PATTERNS):
        responsible = (
            Color.RED if red in _KILL_WITHOUT_CHECK_PATTERNS else Color.BLACK
        )
        return _TableDecision(
            AdjudicationKind.MUST_CHANGE,
            responsible,
            "中国棋规2020 表项26.9.1（单方长杀相对长捉）",
            "双方均有禁止着法，长杀或杀捉组合方须变着",
        )
    if {red, black} == {_CyclePattern.CHASE, _CyclePattern.JOINT_CHASE}:
        responsible = Color.RED if red is _CyclePattern.CHASE else Color.BLACK
        ordinary = profiles[responsible]
        joint_profile = profiles[responsible.opponent]
        if (
            ordinary.every_chase_frame_targets_rook
            and joint_profile.every_chase_frame_targets_rook
        ):
            return _TableDecision(
                AdjudicationKind.MUST_CHANGE,
                responsible,
                "中国棋规2020 表项26.9.2（长捉车相对联合捉车）",
                "单子长捉车的一方相对联合捉车方须变着",
            )
        if (
            ordinary.every_chase_frame_targets_unrooted
            and joint_profile.every_chase_frame_targets_unrooted
        ):
            return _TableDecision(
                AdjudicationKind.MUST_CHANGE,
                responsible,
                "中国棋规2020 表项26.9.3（长捉无根子相对联合捉）",
                "单子长捉无根子的一方相对联合捉方须变着",
            )
    return _TableDecision(
        AdjudicationKind.DRAW,
        None,
        "中国棋规2020 表项26.9.4（其余双方禁止着法）",
        "双方禁止着法同责或不属于26.9.1至26.9.3差等情形，判和",
    )


def _asian_bilateral_decision(
    profiles: Mapping[Color, _CycleProfile],
) -> _TableDecision:
    red = profiles[Color.RED].pattern
    black = profiles[Color.BLACK].pattern
    if (red in _CHECK_PATTERNS) != (black in _CHECK_PATTERNS):
        responsible = Color.RED if red in _CHECK_PATTERNS else Color.BLACK
    elif (
        red in _CHASE_WITHOUT_CHECK_PATTERNS
    ) != (black in _CHASE_WITHOUT_CHECK_PATTERNS):
        responsible = (
            Color.RED if red in _CHASE_WITHOUT_CHECK_PATTERNS else Color.BLACK
        )
    else:
        return _TableDecision(
            AdjudicationKind.DRAW,
            None,
            "AXF 2003 Chapter 4 Table 4-D (same responsibility class)",
            "双方循环属于相同责任级别，双方不变判和",
        )
    return _TableDecision(
        AdjudicationKind.MUST_CHANGE,
        responsible,
        "AXF 2003 Chapter 4 Table 4-C "
        "(check over chase; chase over kill/TTC)",
        "双方循环责任类别不同，较重的将、捉或杀/TTC方须变着",
    )


def _evaluate_responsibility(
    by_side: Mapping[Color, Sequence[MoveNature]],
    ruleset: Ruleset,
    reference: str,
    cycle_start: int | None,
    ordered_natures: Sequence[MoveNature] | None = None,
    profiles_override: Mapping[Color, _CycleProfile] | None = None,
) -> Adjudication:
    """Apply the formal responsibility table after tactical classification."""
    normalized = {color: tuple(by_side.get(color, ())) for color in Color}
    if not all(normalized.values()):
        raise ValueError("双方均须提供循环中的着法性质")
    profiles = (
        dict(profiles_override)
        if profiles_override is not None
        else {
            color: _CycleProfile(_cycle_pattern(natures))
            for color, natures in normalized.items()
        }
    )
    decision = _single_or_allowed_decision(profiles, ruleset=ruleset)
    if decision is None:
        decision = (
            _chinese_bilateral_decision(profiles)
            if ruleset is Ruleset.CHINESE_2020
            else _asian_bilateral_decision(profiles)
        )
    return _make_result(
        by_side=normalized,
        cycle_start=cycle_start,
        ruleset=ruleset,
        reference=decision.reference,
        kind=decision.kind,
        responsible=decision.responsible,
        reason=decision.reason,
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
        effective = _effective_cycle_natures(cycle)
        profiles = {
            color: _cycle_profile_with_evidence(
                tuple(frame for frame in cycle.frames if frame.side is color),
                tuple(
                    nature
                    for frame, nature in zip(
                        cycle.frames, effective, strict=True
                    )
                    if frame.side is color
                ),
            )
            for color in Color
        }
        return _evaluate_responsibility(
            _effective_natures_by_side(cycle, effective),
            self.ruleset,
            self._reference,
            cycle.start,
            effective,
            profiles,
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
        effective = _effective_cycle_natures(cycle)
        profiles = {
            color: _cycle_profile_with_evidence(
                tuple(frame for frame in cycle.frames if frame.side is color),
                tuple(
                    nature
                    for frame, nature in zip(
                        cycle.frames, effective, strict=True
                    )
                    if frame.side is color
                ),
            )
            for color in Color
        }
        return _evaluate_responsibility(
            _effective_natures_by_side(cycle, effective),
            self.ruleset,
            self._reference,
            cycle.start,
            effective,
            profiles,
        )
