from __future__ import annotations

import pytest

import xiangqi.adjudication as adjudication_module
from xiangqi.adjudication import (
    AdjudicationKind,
    Asian2003Adjudicator,
    Chinese2020Adjudicator,
    MoveNature,
    PositionFrame,
    Ruleset,
)
from xiangqi.board import Board
from xiangqi.domain import Color, Coord, Move, Piece, PieceType


def _piece(color: Color, kind: PieceType) -> Piece:
    return Piece(color, kind)


def _play(
    board: Board,
    side: Color,
    start: tuple[int, int],
    end: tuple[int, int],
) -> tuple[Board, PositionFrame]:
    start_coord = Coord(*start)
    end_coord = Coord(*end)
    piece = board.at(start_coord)
    assert piece is not None
    move = Move(start_coord, end_coord, piece, board.at(end_coord))
    after = board.move_unchecked(start_coord, end_coord)
    return after, PositionFrame.from_transition(board, side, move, after)


def _perpetual_check_history() -> list[PositionFrame]:
    board = (
        Board.empty()
        .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
        .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
        .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
        .place(Coord(3, 1), _piece(Color.RED, PieceType.ROOK))
    )
    history: list[PositionFrame] = []
    side = Color.RED
    cycle = (
        ((3, 1), (4, 1)),
        ((4, 0), (3, 0)),
        ((4, 1), (3, 1)),
        ((3, 0), (4, 0)),
    )
    for _ in range(2):
        for start, end in cycle:
            board, frame = _play(board, side, start, end)
            history.append(frame)
            side = side.opponent
    return history


def _ordinary_repetition_history() -> list[PositionFrame]:
    board = (
        Board.empty()
        .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
        .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
        .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
        .place(Coord(0, 8), _piece(Color.RED, PieceType.ROOK))
        .place(Coord(8, 1), _piece(Color.BLACK, PieceType.ROOK))
    )
    history: list[PositionFrame] = []
    side = Color.RED
    cycle = (
        ((0, 8), (1, 8)),
        ((8, 1), (7, 1)),
        ((1, 8), (0, 8)),
        ((7, 1), (8, 1)),
    )
    for _ in range(2):
        for start, end in cycle:
            board, frame = _play(board, side, start, end)
            history.append(frame)
            side = side.opponent
    return history


def _rook_chase_history(
    *, rooted: bool = False, target_kind: PieceType = PieceType.ROOK
) -> list[PositionFrame]:
    """A real four-ply position loop: the red rook follows the black rook."""
    board = (
        Board.empty()
        .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
        .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
        .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
        .place(Coord(0, 5), _piece(Color.RED, PieceType.ROOK))
        .place(Coord(1, 3), _piece(Color.BLACK, target_kind))
    )
    if rooted:
        board = board.place(Coord(1, 0), _piece(Color.BLACK, PieceType.ROOK))
    history: list[PositionFrame] = []
    side = Color.RED
    cycle = (
        ((0, 5), (1, 5)),
        ((1, 3), (0, 3)),
        ((1, 5), (0, 5)),
        ((0, 3), (1, 3)),
    )
    for _ in range(2):
        for start, end in cycle:
            board, frame = _play(board, side, start, end)
            history.append(frame)
            side = side.opponent
    return history


def _pawn_follow_history() -> list[PositionFrame]:
    """The same geometric follow by a pawn is an explicit long-chase exception."""
    board = (
        Board.empty()
        .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
        .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
        .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
        .place(Coord(0, 4), _piece(Color.RED, PieceType.PAWN))
        .place(Coord(1, 4), _piece(Color.BLACK, PieceType.ROOK))
    )
    # A single transition is enough to prove classification of the exception.
    # Rebuild it without capture: the pawn newly attacks a rook on its flank.
    board = board.remove(Coord(1, 4)).place(
        Coord(2, 4), _piece(Color.BLACK, PieceType.ROOK)
    )
    board, frame = _play(board, Color.RED, (0, 4), (1, 4))
    return [frame]


def _one_check_one_chase_history() -> list[PositionFrame]:
    """A real cycle where Red alternates one check and one chase."""
    board = (
        Board.empty()
        .place(Coord(3, 0), _piece(Color.BLACK, PieceType.GENERAL))
        .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
        .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
        .place(Coord(2, 5), _piece(Color.RED, PieceType.ROOK))
        .place(Coord(2, 3), _piece(Color.BLACK, PieceType.ROOK))
    )
    history: list[PositionFrame] = []
    side = Color.RED
    cycle = (
        ((2, 5), (3, 5)),
        ((3, 0), (4, 0)),
        ((3, 5), (2, 5)),
        ((4, 0), (3, 0)),
    )
    for _ in range(2):
        for start, end in cycle:
            board, frame = _play(board, side, start, end)
            history.append(frame)
            side = side.opponent
    return history


def test_position_frame_classifies_real_check_and_idle_moves() -> None:
    check_history = _perpetual_check_history()
    idle_history = _ordinary_repetition_history()

    assert [frame.nature for frame in check_history[:2]] == [
        MoveNature.CHECK,
        MoveNature.IDLE,
    ]
    assert all(frame.nature is MoveNature.IDLE for frame in idle_history)


@pytest.mark.parametrize(
    "adjudicator",
    [Chinese2020Adjudicator(), Asian2003Adjudicator()],
)
def test_single_side_perpetual_check_must_change(adjudicator) -> None:
    result = adjudicator.evaluate(_perpetual_check_history())

    assert result.kind is AdjudicationKind.MUST_CHANGE
    assert result.cycle_start == 0
    assert result.responsible is Color.RED
    assert result.responsible_natures == (MoveNature.CHECK,) * 4
    assert result.move_natures == (MoveNature.CHECK, MoveNature.IDLE) * 4
    assert result.ruleset is adjudicator.ruleset
    assert result.rule_reference


@pytest.mark.parametrize(
    ("adjudicator", "ruleset"),
    [
        (Chinese2020Adjudicator(), Ruleset.CHINESE_2020),
        (Asian2003Adjudicator(), Ruleset.ASIAN_2003),
    ],
)
def test_threefold_ordinary_repetition_is_draw(adjudicator, ruleset) -> None:
    result = adjudicator.evaluate(_ordinary_repetition_history())

    assert result.kind is AdjudicationKind.DRAW
    assert result.cycle_start == 0
    assert result.responsible is None
    assert result.ruleset is ruleset
    assert set(result.move_natures) == {MoveNature.IDLE}


@pytest.mark.parametrize(
    "adjudicator",
    [Chinese2020Adjudicator(), Asian2003Adjudicator()],
)
def test_two_occurrences_are_not_enough_for_a_decision(adjudicator) -> None:
    history = _ordinary_repetition_history()[:4]

    result = adjudicator.evaluate(history)

    assert result.kind is AdjudicationKind.NO_DECISION
    assert result.cycle_start is None


def test_position_frame_classifies_new_profitable_attack_as_chase() -> None:
    history = _rook_chase_history()

    assert [frame.nature for frame in history[:4]] == [
        MoveNature.CHASE,
        MoveNature.IDLE,
        MoveNature.CHASE,
        MoveNature.IDLE,
    ]


def test_position_frame_classifies_unrooted_lesser_piece_as_chase() -> None:
    history = _rook_chase_history(target_kind=PieceType.CANNON)

    assert history[0].nature is MoveNature.CHASE
    assert history[0].attacks[0].target_piece.kind is PieceType.CANNON


def test_a_genuinely_rooted_target_is_not_classified_as_chase() -> None:
    history = _rook_chase_history(rooted=True)

    assert history[0].nature is MoveNature.FOLLOW


def test_a_pinned_defender_is_only_a_fake_root() -> None:
    board = (
        Board.empty()
        .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
        .place(Coord(3, 9), _piece(Color.RED, PieceType.GENERAL))
        .place(Coord(0, 5), _piece(Color.RED, PieceType.ROOK))
        .place(Coord(4, 5), _piece(Color.RED, PieceType.ROOK))
        .place(Coord(1, 3), _piece(Color.BLACK, PieceType.ROOK))
        .place(Coord(4, 3), _piece(Color.BLACK, PieceType.ROOK))
    )

    _, frame = _play(board, Color.RED, (0, 5), (1, 5))

    assert frame.nature is MoveNature.EXCHANGE
    assert frame.attacks[0].rooted is False


def test_a_pinned_attacker_does_not_create_a_fake_chase() -> None:
    board = (
        Board.empty()
        .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
        .place(Coord(1, 9), _piece(Color.RED, PieceType.GENERAL))
        .place(Coord(1, 0), _piece(Color.BLACK, PieceType.ROOK))
        .place(Coord(0, 7), _piece(Color.RED, PieceType.HORSE))
        .place(Coord(3, 4), _piece(Color.BLACK, PieceType.ROOK))
    )

    _, frame = _play(board, Color.RED, (0, 7), (1, 5))

    assert frame.nature is MoveNature.SACRIFICE
    assert frame.attacks == ()


def test_discovered_legal_attack_is_recorded_as_a_chase() -> None:
    board = (
        Board.empty()
        .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
        .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
        .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
        .place(Coord(0, 8), _piece(Color.RED, PieceType.ROOK))
        .place(Coord(0, 5), _piece(Color.RED, PieceType.HORSE))
        .place(Coord(0, 3), _piece(Color.BLACK, PieceType.ROOK))
    )

    _, frame = _play(board, Color.RED, (0, 5), (2, 6))

    assert frame.nature is MoveNature.CHASE
    assert tuple(attack.target for attack in frame.attacks) == (Coord(0, 3),)


def test_general_and_pawn_attacks_are_not_long_chase_moves() -> None:
    assert _pawn_follow_history()[0].nature is MoveNature.IDLE
    board = (
        Board.empty()
        .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
        .place(Coord(3, 9), _piece(Color.RED, PieceType.GENERAL))
        .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
        .place(Coord(4, 8), _piece(Color.BLACK, PieceType.ADVISOR))
    )
    _, general_frame = _play(board, Color.RED, (3, 9), (4, 9))

    assert general_frame.nature is MoveNature.IDLE


def test_position_frame_classifies_a_forced_mate_next_move_as_kill() -> None:
    board = (
        Board.empty()
        .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
        .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
        .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
        .place(Coord(0, 1), _piece(Color.RED, PieceType.ROOK))
        .place(Coord(0, 2), _piece(Color.RED, PieceType.ROOK))
    )

    _, frame = _play(board, Color.RED, (0, 1), (1, 1))

    assert frame.nature is MoveNature.KILL


@pytest.mark.parametrize(
    "adjudicator",
    [Chinese2020Adjudicator(), Asian2003Adjudicator()],
)
def test_single_side_perpetual_chase_of_unrooted_rook_must_change(adjudicator) -> None:
    result = adjudicator.evaluate(_rook_chase_history())

    assert result.kind is AdjudicationKind.MUST_CHANGE
    assert result.responsible is Color.RED
    assert result.responsible_natures == (MoveNature.CHASE,) * 4
    assert "长捉" in result.reason
    assert result.rule_reference


def test_rulesets_keep_independent_one_check_one_chase_policy() -> None:
    history = _one_check_one_chase_history()
    assert tuple(frame.nature for frame in history if frame.side is Color.RED) == (
        MoveNature.CHECK,
        MoveNature.CHASE,
        MoveNature.CHECK,
        MoveNature.CHASE,
    )

    chinese = Chinese2020Adjudicator().evaluate(history)
    asian = Asian2003Adjudicator().evaluate(history)

    assert chinese.kind is AdjudicationKind.MUST_CHANGE
    assert chinese.responsible is Color.RED
    assert asian.kind is AdjudicationKind.MUST_CHANGE
    assert asian.responsible is Color.RED
    assert chinese.rule_reference != asian.rule_reference


@pytest.mark.parametrize(
    ("natures", "chinese_kind", "asian_kind"),
    [
        (
            (MoveNature.CHECK, MoveNature.KILL),
            AdjudicationKind.MUST_CHANGE,
            AdjudicationKind.MUST_CHANGE,
        ),
        (
            (MoveNature.CHECK, MoveNature.CHASE),
            AdjudicationKind.MUST_CHANGE,
            AdjudicationKind.MUST_CHANGE,
        ),
        (
            (MoveNature.KILL, MoveNature.CHASE),
            AdjudicationKind.MUST_CHANGE,
            AdjudicationKind.MUST_CHANGE,
        ),
        (
            (MoveNature.CHECK, MoveNature.KILL, MoveNature.CHASE),
            AdjudicationKind.MUST_CHANGE,
            AdjudicationKind.MUST_CHANGE,
        ),
        (
            (MoveNature.KILL,),
            AdjudicationKind.MUST_CHANGE,
            AdjudicationKind.MUST_CHANGE,
        ),
        (
            (MoveNature.CHASE,),
            AdjudicationKind.MUST_CHANGE,
            AdjudicationKind.MUST_CHANGE,
        ),
        ((MoveNature.IDLE,), AdjudicationKind.DRAW, AdjudicationKind.DRAW),
    ],
)
def test_formal_single_side_responsibility_table(
    natures, chinese_kind, asian_kind
) -> None:
    chinese = Chinese2020Adjudicator().evaluate_natures(
        {Color.RED: natures, Color.BLACK: (MoveNature.IDLE,) * len(natures)}
    )
    asian = Asian2003Adjudicator().evaluate_natures(
        {Color.RED: natures, Color.BLACK: (MoveNature.IDLE,) * len(natures)}
    )

    assert chinese.kind is chinese_kind
    assert asian.kind is asian_kind
    assert chinese.responsible is (
        Color.RED if chinese_kind is AdjudicationKind.MUST_CHANGE else None
    )
    assert asian.responsible is (
        Color.RED if asian_kind is AdjudicationKind.MUST_CHANGE else None
    )
    assert "表项" in chinese.rule_reference
    assert "Table" in asian.rule_reference


@pytest.mark.parametrize(
    "nature",
    [
        MoveNature.EXCHANGE,
        MoveNature.SACRIFICE,
        MoveNature.BLOCK,
        MoveNature.FOLLOW,
        MoveNature.IDLE,
    ],
)
@pytest.mark.parametrize(
    "adjudicator",
    [Chinese2020Adjudicator(), Asian2003Adjudicator()],
)
def test_each_permitted_nature_is_explicitly_mapped_to_draw(
    adjudicator, nature: MoveNature
) -> None:
    result = adjudicator.evaluate_natures(
        {Color.RED: (nature,) * 2, Color.BLACK: (MoveNature.IDLE,) * 2}
    )

    assert result.kind is AdjudicationKind.DRAW
    assert result.responsible is None
    assert "允许" in result.reason or "闲" in result.reason
    assert "表项" in result.rule_reference or "Table" in result.rule_reference


@pytest.mark.parametrize(
    ("red", "black", "responsible", "reference"),
    [
        (MoveNature.CHECK, MoveNature.CHASE, Color.RED, "25.1"),
        (MoveNature.KILL, MoveNature.CHASE, Color.RED, "26.9.1"),
        (MoveNature.CHASE, MoveNature.CHASE, None, "26.9.4"),
        (MoveNature.KILL, MoveNature.KILL, None, "26.9.4"),
    ],
)
def test_chinese_bilateral_prohibited_move_table_is_explicit(
    red: MoveNature,
    black: MoveNature,
    responsible: Color | None,
    reference: str,
) -> None:
    result = Chinese2020Adjudicator().evaluate_natures(
        {Color.RED: (red,) * 2, Color.BLACK: (black,) * 2}
    )

    assert result.responsible is responsible
    assert result.kind is (
        AdjudicationKind.DRAW
        if responsible is None
        else AdjudicationKind.MUST_CHANGE
    )
    assert reference in result.rule_reference


@pytest.mark.parametrize(
    ("red", "black", "responsible", "table_item"),
    [
        (MoveNature.CHECK, MoveNature.CHASE, Color.RED, "Table 4-C"),
        (MoveNature.CHECK, MoveNature.KILL, Color.RED, "Table 4-C"),
        (MoveNature.CHASE, MoveNature.KILL, Color.RED, "Table 4-C"),
        (MoveNature.KILL, MoveNature.KILL, None, "Table 4-D"),
        (MoveNature.CHASE, MoveNature.CHASE, None, "Table 4-D"),
    ],
)
def test_asian_bilateral_responsibility_table_is_explicit(
    red: MoveNature,
    black: MoveNature,
    responsible: Color | None,
    table_item: str,
) -> None:
    result = Asian2003Adjudicator().evaluate_natures(
        {Color.RED: (red,) * 2, Color.BLACK: (black,) * 2}
    )

    assert result.responsible is responsible
    assert result.kind is (
        AdjudicationKind.DRAW
        if responsible is None
        else AdjudicationKind.MUST_CHANGE
    )
    assert table_item in result.rule_reference


def test_rulesets_explicitly_reverse_kill_versus_chase_priority() -> None:
    by_side = {
        Color.RED: (MoveNature.KILL,) * 2,
        Color.BLACK: (MoveNature.CHASE,) * 2,
    }

    chinese = Chinese2020Adjudicator().evaluate_natures(by_side)
    asian = Asian2003Adjudicator().evaluate_natures(by_side)

    assert chinese.responsible is Color.RED
    assert asian.responsible is Color.BLACK
    assert "26.9.1" in chinese.rule_reference
    assert "Table 4-C" in asian.rule_reference


def _two_attackers_frame(
    *, target_kind: PieceType, rooted: bool
) -> PositionFrame:
    board = (
        Board.empty()
        .place(Coord(5, 0), _piece(Color.BLACK, PieceType.GENERAL))
        .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
        .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
        .place(Coord(1, 2), _piece(Color.RED, PieceType.HORSE))
        .place(Coord(2, 1), _piece(Color.RED, PieceType.HORSE))
        .place(Coord(2, 2), _piece(Color.RED, PieceType.ROOK))
        .place(Coord(3, 3), _piece(Color.BLACK, target_kind))
    )
    if rooted:
        board = board.place(
            Coord(5, 4), _piece(Color.BLACK, PieceType.HORSE)
        )
    _, frame = _play(board, Color.RED, (2, 2), (3, 2))
    return frame


def _single_attacker_frame(
    *, target_kind: PieceType, rooted: bool
) -> PositionFrame:
    board = (
        Board.empty()
        .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
        .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
        .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
        .place(Coord(0, 7), _piece(Color.RED, PieceType.HORSE))
        .place(Coord(3, 4), _piece(Color.BLACK, target_kind))
    )
    if rooted:
        board = board.place(
            Coord(3, 0), _piece(Color.BLACK, PieceType.ROOK)
        )
    _, frame = _play(board, Color.RED, (0, 7), (1, 5))
    return frame


def test_two_independent_attacks_on_unrooted_target_are_not_joint_chase() -> None:
    frame = _two_attackers_frame(target_kind=PieceType.ROOK, rooted=False)
    assert frame.nature is MoveNature.CHASE
    assert [attack.target for attack in frame.attacks].count(Coord(3, 3)) >= 2
    profile = adjudication_module._cycle_profile_with_evidence(
        (frame,), (MoveNature.CHASE,)
    )
    assert profile.pattern.value == "chase"
    assert profile.every_chase_frame_targets_unrooted is True


def test_rooted_target_requiring_both_new_attackers_is_joint_chase() -> None:
    frame = _two_attackers_frame(target_kind=PieceType.ROOK, rooted=True)

    profile = adjudication_module._cycle_profile_with_evidence(
        (frame,), (MoveNature.CHASE,)
    )

    assert profile.pattern.value == "joint_chase"
    assert profile.target_kinds == frozenset({PieceType.ROOK})


@pytest.mark.parametrize(
    ("ordinary_kind", "ordinary_rooted", "responsible", "reference"),
    [
        (PieceType.ROOK, True, Color.RED, "26.9.2"),
        (PieceType.CANNON, False, None, "26.9.4"),
        (PieceType.CANNON, True, None, "26.9.4"),
    ],
)
def test_chinese_ordinary_chase_versus_joint_chase_uses_target_evidence(
    ordinary_kind: PieceType,
    ordinary_rooted: bool,
    responsible: Color | None,
    reference: str,
) -> None:
    ordinary_frame = _single_attacker_frame(
        target_kind=ordinary_kind, rooted=ordinary_rooted
    )
    joint_frame = _two_attackers_frame(
        target_kind=ordinary_kind, rooted=True
    )
    red_profile = adjudication_module._cycle_profile_with_evidence(
        (ordinary_frame,), (MoveNature.CHASE,)
    )
    black_profile = adjudication_module._cycle_profile_with_evidence(
        (joint_frame,), (MoveNature.CHASE,)
    )
    decision = adjudication_module._chinese_bilateral_decision(
        {Color.RED: red_profile, Color.BLACK: black_profile}
    )

    assert decision.responsible is responsible
    assert decision.kind is (
        AdjudicationKind.DRAW
        if responsible is None
        else AdjudicationKind.MUST_CHANGE
    )
    assert reference in decision.reference


def test_nonqualifying_incidental_rook_attack_does_not_upgrade_profile() -> None:
    board = (
        Board.empty()
        .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
        .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
        .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
        .place(Coord(0, 5), _piece(Color.RED, PieceType.ROOK))
        .place(Coord(1, 3), _piece(Color.BLACK, PieceType.CANNON))
        .place(Coord(3, 5), _piece(Color.BLACK, PieceType.ROOK))
        .place(Coord(3, 0), _piece(Color.BLACK, PieceType.ROOK))
    )
    _, frame = _play(board, Color.RED, (0, 5), (1, 5))

    profile = adjudication_module._cycle_profile_with_evidence(
        (frame,), (MoveNature.CHASE,)
    )

    assert frame.nature is MoveNature.CHASE
    assert {attack.target_piece.kind for attack in frame.attacks} == {
        PieceType.CANNON,
        PieceType.ROOK,
    }
    assert profile.target_kinds == frozenset({PieceType.CANNON})
    assert profile.every_chase_frame_targets_rook is False


def test_only_some_cycle_frames_chasing_rook_does_not_trigger_2692() -> None:
    rook_frame = _single_attacker_frame(target_kind=PieceType.ROOK, rooted=True)
    cannon_frame = _single_attacker_frame(
        target_kind=PieceType.CANNON, rooted=True
    )
    ordinary = adjudication_module._cycle_profile_with_evidence(
        (rook_frame, cannon_frame), (MoveNature.CHASE,) * 2
    )
    joint = adjudication_module._cycle_profile_with_evidence(
        (
            _two_attackers_frame(target_kind=PieceType.ROOK, rooted=True),
        )
        * 2,
        (MoveNature.CHASE,) * 2,
    )

    decision = adjudication_module._chinese_bilateral_decision(
        {Color.RED: ordinary, Color.BLACK: joint}
    )

    assert decision.kind is AdjudicationKind.DRAW
    assert "26.9.4" in decision.reference


def test_ordinary_rook_vs_joint_rooted_cannon_does_not_trigger_2692() -> None:
    ordinary = adjudication_module._cycle_profile_with_evidence(
        (_single_attacker_frame(target_kind=PieceType.ROOK, rooted=True),),
        (MoveNature.CHASE,),
    )
    joint = adjudication_module._cycle_profile_with_evidence(
        (_two_attackers_frame(target_kind=PieceType.CANNON, rooted=True),),
        (MoveNature.CHASE,),
    )

    decision = adjudication_module._chinese_bilateral_decision(
        {Color.RED: ordinary, Color.BLACK: joint}
    )

    assert decision.kind is AdjudicationKind.DRAW
    assert "26.9.4" in decision.reference


def test_only_some_cycle_frames_chasing_unrooted_piece_does_not_trigger_2693() -> None:
    ordinary = adjudication_module._cycle_profile_with_evidence(
        (
            _single_attacker_frame(target_kind=PieceType.CANNON, rooted=False),
            _single_attacker_frame(target_kind=PieceType.CANNON, rooted=True),
        ),
        (MoveNature.CHASE,) * 2,
    )
    joint = adjudication_module._cycle_profile_with_evidence(
        (
            _two_attackers_frame(target_kind=PieceType.CANNON, rooted=True),
        )
        * 2,
        (MoveNature.CHASE,) * 2,
    )

    decision = adjudication_module._chinese_bilateral_decision(
        {Color.RED: ordinary, Color.BLACK: joint}
    )

    assert decision.kind is AdjudicationKind.DRAW
    assert "26.9.4" in decision.reference


def test_both_sides_every_cycle_frame_chasing_rook_triggers_2692() -> None:
    ordinary_frames = tuple(
        _single_attacker_frame(target_kind=PieceType.ROOK, rooted=True)
        for _ in range(2)
    )
    joint_frames = tuple(
        _two_attackers_frame(target_kind=PieceType.ROOK, rooted=True)
        for _ in range(2)
    )
    ordinary = adjudication_module._cycle_profile_with_evidence(
        ordinary_frames, (MoveNature.CHASE,) * 2
    )
    joint = adjudication_module._cycle_profile_with_evidence(
        joint_frames, (MoveNature.CHASE,) * 2
    )

    decision = adjudication_module._chinese_bilateral_decision(
        {Color.RED: ordinary, Color.BLACK: joint}
    )

    assert decision.responsible is Color.RED
    assert "26.9.2" in decision.reference


def test_2693_requires_both_profiles_to_mark_every_frame_unrooted() -> None:
    profile = adjudication_module._CycleProfile
    pattern = adjudication_module._CyclePattern
    ordinary = profile(
        pattern.CHASE,
        frozenset({PieceType.CANNON}),
        every_chase_frame_targets_unrooted=True,
    )
    joint = profile(
        pattern.JOINT_CHASE,
        frozenset({PieceType.CANNON}),
        every_chase_frame_targets_unrooted=True,
    )

    decision = adjudication_module._chinese_bilateral_decision(
        {Color.RED: ordinary, Color.BLACK: joint}
    )

    assert decision.responsible is Color.RED
    assert "26.9.3" in decision.reference


def test_fourth_unchanged_cycle_keeps_explicit_loss_evidence() -> None:
    adjudicator = Chinese2020Adjudicator()
    must_change = adjudicator.evaluate(_perpetual_check_history())

    loss = adjudicator.loss_for_ignored_must_change(must_change)

    assert loss.kind is AdjudicationKind.LOSS
    assert loss.responsible is Color.RED
    assert loss.rule_reference == must_change.rule_reference
    assert loss.move_natures == must_change.move_natures


@pytest.mark.parametrize(
    "adjudicator",
    [Chinese2020Adjudicator(), Asian2003Adjudicator()],
)
def test_long_check_has_priority_over_opponents_long_chase(adjudicator) -> None:
    result = adjudicator.evaluate_natures(
        {
            Color.RED: (MoveNature.CHECK,) * 2,
            Color.BLACK: (MoveNature.CHASE,) * 2,
        }
    )

    assert result.kind is AdjudicationKind.MUST_CHANGE
    assert result.responsible is Color.RED


@pytest.mark.parametrize(
    ("adjudicator", "red", "black"),
    [
        (
            Chinese2020Adjudicator(),
            (MoveNature.KILL, MoveNature.CHASE),
            (MoveNature.CHASE, MoveNature.KILL),
        ),
        (
            Asian2003Adjudicator(),
            (MoveNature.CHASE,) * 2,
            (MoveNature.CHASE,) * 2,
        ),
    ],
)
def test_both_sides_with_equal_responsibility_draw(adjudicator, red, black) -> None:
    result = adjudicator.evaluate_natures({Color.RED: red, Color.BLACK: black})

    assert result.kind is AdjudicationKind.DRAW
    assert result.responsible is None


@pytest.mark.parametrize(
    "adjudicator",
    [Chinese2020Adjudicator(), Asian2003Adjudicator()],
)
def test_ignored_must_change_has_explicit_loss_path(adjudicator) -> None:
    first = adjudicator.evaluate(_perpetual_check_history())

    result = adjudicator.loss_for_ignored_must_change(first)

    assert result.kind is AdjudicationKind.LOSS
    assert result.responsible is Color.RED
    assert "未变着" in result.reason


@pytest.mark.parametrize(
    "adjudicator",
    [Chinese2020Adjudicator(), Asian2003Adjudicator()],
)
def test_general_and_pawn_chase_exception_is_allowed(adjudicator) -> None:
    result = adjudicator.evaluate_natures(
        {
            Color.RED: (MoveNature.IDLE,) * 2,
            Color.BLACK: (MoveNature.IDLE,) * 2,
        }
    )

    assert result.kind is AdjudicationKind.DRAW
    assert "允许" in result.reason or "闲着" in result.reason


def test_tampered_move_nature_is_rejected() -> None:
    history = _ordinary_repetition_history()
    frame = history[0]
    object.__setattr__(frame, "nature", MoveNature.CHECK)

    with pytest.raises(ValueError, match="着法性质"):
        Chinese2020Adjudicator().evaluate(history)


def test_illegal_transition_is_rejected() -> None:
    board = (
        Board.empty()
        .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
        .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
        .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
        .place(Coord(0, 9), _piece(Color.RED, PieceType.ROOK))
    )
    move = Move(
        Coord(0, 9),
        Coord(1, 8),
        _piece(Color.RED, PieceType.ROOK),
        None,
    )

    with pytest.raises(ValueError, match="非法"):
        PositionFrame.from_transition(
            board,
            Color.RED,
            move,
            board.move_unchecked(move.start, move.end),
        )


def test_frame_records_root_and_exchange_evidence() -> None:
    unrooted = _rook_chase_history()[0]
    rooted = _rook_chase_history(rooted=True)[0]

    assert len(unrooted.attacks) == 1
    assert unrooted.attacks[0].target_piece.kind is PieceType.ROOK
    assert unrooted.attacks[0].attacker_value == 9
    assert unrooted.attacks[0].target_value == 9
    assert unrooted.attacks[0].rooted is False
    assert rooted.attacks[0].rooted is True


@pytest.mark.parametrize(
    ("board", "start", "end", "expected_target"),
    [
        (
            Board.empty()
            .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
            .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
            .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
            .place(Coord(0, 7), _piece(Color.RED, PieceType.ROOK))
            .place(Coord(3, 5), _piece(Color.RED, PieceType.ROOK))
            .place(Coord(0, 3), _piece(Color.BLACK, PieceType.ROOK)),
            (0, 7),
            (0, 5),
            Coord(0, 3),
        ),
        (
            Board.empty()
            .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
            .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
            .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
            .place(Coord(0, 7), _piece(Color.RED, PieceType.HORSE))
            .place(Coord(1, 9), _piece(Color.RED, PieceType.ROOK))
            .place(Coord(3, 4), _piece(Color.BLACK, PieceType.HORSE))
            .place(Coord(3, 0), _piece(Color.BLACK, PieceType.ROOK)),
            (0, 7),
            (1, 5),
            Coord(3, 4),
        ),
    ],
)
def test_real_transition_classifies_same_kind_invitation_as_exchange(
    board: Board,
    start: tuple[int, int],
    end: tuple[int, int],
    expected_target: Coord,
) -> None:
    _, frame = _play(board, Color.RED, start, end)

    assert frame.nature is MoveNature.EXCHANGE
    assert frame.evidence.nature is MoveNature.EXCHANGE
    assert expected_target in frame.evidence.targets


@pytest.mark.parametrize(
    ("board", "start", "end", "expected_attacker"),
    [
        (
            Board.empty()
            .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
            .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
            .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
            .place(Coord(0, 7), _piece(Color.RED, PieceType.HORSE))
            .place(Coord(1, 0), _piece(Color.BLACK, PieceType.ROOK)),
            (0, 7),
            (1, 5),
            Coord(1, 0),
        ),
        (
            Board.empty()
            .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
            .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
            .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
            .place(Coord(2, 9), _piece(Color.RED, PieceType.ELEPHANT))
            .place(Coord(0, 0), _piece(Color.BLACK, PieceType.ROOK)),
            (2, 9),
            (0, 7),
            Coord(0, 0),
        ),
    ],
)
def test_real_transition_classifies_unanswered_offer_as_sacrifice(
    board: Board,
    start: tuple[int, int],
    end: tuple[int, int],
    expected_attacker: Coord,
) -> None:
    _, frame = _play(board, Color.RED, start, end)

    assert frame.nature is MoveNature.SACRIFICE
    assert frame.evidence.nature is MoveNature.SACRIFICE
    assert expected_attacker in frame.evidence.actors


@pytest.mark.parametrize(
    ("board", "start", "end", "blocked_target"),
    [
        (
            Board.empty()
            .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
            .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
            .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
            .place(Coord(1, 6), _piece(Color.RED, PieceType.HORSE))
            .place(Coord(0, 5), _piece(Color.RED, PieceType.ROOK))
            .place(Coord(0, 0), _piece(Color.BLACK, PieceType.ROOK)),
            (1, 6),
            (0, 4),
            Coord(0, 5),
        ),
        (
            Board.empty()
            .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
            .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
            .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
            .place(Coord(7, 6), _piece(Color.RED, PieceType.HORSE))
            .place(Coord(8, 5), _piece(Color.RED, PieceType.ROOK))
            .place(Coord(8, 0), _piece(Color.BLACK, PieceType.ROOK)),
            (7, 6),
            (8, 4),
            Coord(8, 5),
        ),
    ],
)
def test_real_transition_classifies_non_attacking_interposition_as_block(
    board: Board,
    start: tuple[int, int],
    end: tuple[int, int],
    blocked_target: Coord,
) -> None:
    _, frame = _play(board, Color.RED, start, end)

    assert frame.nature is MoveNature.BLOCK
    assert frame.evidence.nature is MoveNature.BLOCK
    assert blocked_target in frame.evidence.targets


@pytest.mark.parametrize(
    "target_kind",
    [PieceType.ROOK, PieceType.CANNON],
)
def test_real_transition_classifies_pressure_on_rooted_piece_as_follow(
    target_kind: PieceType,
) -> None:
    board = (
        Board.empty()
        .place(Coord(4, 0), _piece(Color.BLACK, PieceType.GENERAL))
        .place(Coord(4, 9), _piece(Color.RED, PieceType.GENERAL))
        .place(Coord(4, 5), _piece(Color.RED, PieceType.PAWN))
        .place(Coord(0, 5), _piece(Color.RED, PieceType.ROOK))
        .place(Coord(1, 3), _piece(Color.BLACK, target_kind))
        .place(Coord(1, 0), _piece(Color.BLACK, PieceType.ROOK))
    )

    _, frame = _play(board, Color.RED, (0, 5), (1, 5))

    assert frame.nature is MoveNature.FOLLOW
    assert frame.evidence.nature is MoveNature.FOLLOW
    assert Coord(1, 3) in frame.evidence.targets


def test_aggressive_natures_keep_priority_over_permitted_move_natures() -> None:
    assert _perpetual_check_history()[0].nature is MoveNature.CHECK
    assert (
        _rook_chase_history(target_kind=PieceType.CANNON)[0].nature
        is MoveNature.CHASE
    )
