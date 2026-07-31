from __future__ import annotations

import pytest

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
    assert asian.kind is AdjudicationKind.DRAW
    assert asian.responsible is None
    assert chinese.rule_reference != asian.rule_reference


@pytest.mark.parametrize(
    ("natures", "chinese_kind", "asian_kind"),
    [
        (
            (MoveNature.CHECK, MoveNature.KILL),
            AdjudicationKind.MUST_CHANGE,
            AdjudicationKind.DRAW,
        ),
        (
            (MoveNature.CHECK, MoveNature.CHASE),
            AdjudicationKind.MUST_CHANGE,
            AdjudicationKind.DRAW,
        ),
        (
            (MoveNature.KILL, MoveNature.CHASE),
            AdjudicationKind.MUST_CHANGE,
            AdjudicationKind.DRAW,
        ),
        ((MoveNature.KILL,), AdjudicationKind.MUST_CHANGE, AdjudicationKind.DRAW),
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
