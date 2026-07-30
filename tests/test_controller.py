from __future__ import annotations

from threading import Event, Thread

import pytest

from xiangqi.adjudication import AdjudicationKind, Ruleset
from xiangqi.board import Board
from xiangqi.controller import (
    ControlError,
    ControllerKind,
    GameController,
    GameEventKind,
    StaleVersionError,
)
from xiangqi.domain import Color, Coord, Move, PositionKind
from xiangqi.notation import format_move
from xiangqi.record import (
    AdjudicationRecord,
    GameRecord,
    MoveRecord,
    ResultRecord,
)
from xiangqi.rules import is_in_check


def _record(
    *,
    fen: str = Board.standard().to_fen(),
    side: Color = Color.RED,
    ruleset: Ruleset = Ruleset.CHINESE_2020,
    moves: tuple[tuple[Coord, Coord], ...] = (),
) -> GameRecord:
    board = Board.from_fen(fen)
    records: list[MoveRecord] = []
    current = side
    for start, end in moves:
        piece = board.at(start)
        assert piece is not None
        move = Move(start, end, piece, board.at(end))
        notation = format_move(board, move)
        board = board.move_unchecked(start, end)
        current = current.opponent
        records.append(
            MoveRecord.from_move(
                move,
                notation=notation,
                position_after=board.position_key(current),
                in_check=is_in_check(board, current),
            )
        )
    return GameRecord(
        ruleset=ruleset,
        initial_fen=fen,
        initial_side=side,
        moves=tuple(records),
        result=ResultRecord(status="ongoing"),
    )


def test_new_controller_has_human_sides_and_legal_moves() -> None:
    controller = GameController.new(ruleset=Ruleset.ASIAN_2003)

    state = controller.get_state()

    assert state.ruleset is Ruleset.ASIAN_2003
    assert state.side_to_move is Color.RED
    assert state.ply == 0
    assert state.version == 0
    assert state.result is None
    assert state.controllers[Color.RED].kind is ControllerKind.HUMAN
    assert state.controllers[Color.BLACK].kind is ControllerKind.HUMAN
    assert Coord(0, 5) in controller.get_legal_moves()[Coord(0, 6)]


def test_move_increments_version_and_rejects_stale_or_illegal_commands() -> None:
    controller = GameController.new()
    before = controller.get_state()

    event = controller.make_move(
        Coord(0, 6), Coord(0, 5), expected_version=before.version
    )

    assert event.kind is GameEventKind.MOVE
    assert controller.get_state().version == 1
    with pytest.raises(StaleVersionError):
        controller.make_move(
            Coord(0, 3), Coord(0, 4), expected_version=before.version
        )
    unchanged = controller.get_state()
    with pytest.raises(ControlError, match="非法"):
        controller.make_move(Coord(0, 3), Coord(1, 3))
    assert controller.get_state() == unchanged


def test_each_move_event_contains_transition_and_terminal_status() -> None:
    controller = GameController.from_record(
        _record(fen="4k4/4RR3/3R5/9/9/9/9/9/9/4K4")
    )

    event = controller.make_move(Coord(3, 2), Coord(3, 1))

    assert event.before_board.to_fen() == "4k4/4RR3/3R5/9/9/9/9/9/9/4K4"
    assert event.after_board == controller.get_state().board
    assert event.in_check
    assert event.checkmate
    assert not event.stalemate
    assert event.position.kind is PositionKind.CHECKMATE
    assert event.result is not None
    assert event.result.status == "red_win"
    assert event.adjudication is not None


def test_undo_is_unlimited_and_terminal_game_can_resume() -> None:
    controller = GameController.from_record(
        _record(fen="4k4/4RR3/3R5/9/9/9/9/9/9/4K4")
    )
    controller.make_move(Coord(3, 2), Coord(3, 1))
    assert controller.get_state().result is not None

    controller.undo()
    assert controller.get_state().result is None
    controller.make_move(Coord(3, 2), Coord(3, 1))
    controller.undo(999)

    assert controller.get_state().ply == 0
    assert controller.get_state().board.to_fen() == controller.record.initial_fen


def test_draw_requires_opponent_and_undo_clears_pending_offer() -> None:
    controller = GameController.new()
    controller.offer_draw(Color.RED)

    assert controller.get_state().pending_draw is Color.RED
    with pytest.raises(ControlError, match="对方"):
        controller.respond_draw(Color.RED, True)
    controller.respond_draw(Color.BLACK, False)
    assert controller.get_state().pending_draw is None

    controller.make_move(Coord(0, 6), Coord(0, 5))
    controller.offer_draw(Color.BLACK)
    controller.undo()
    assert controller.get_state().pending_draw is None


def test_undo_at_initial_position_clears_pending_draw() -> None:
    controller = GameController.new()
    controller.offer_draw(Color.RED)

    controller.undo()

    assert controller.get_state().ply == 0
    assert controller.get_state().pending_draw is None


def test_draw_acceptance_finishes_game_and_terminal_undo_resumes() -> None:
    controller = GameController.new()
    controller.offer_draw(Color.RED)

    controller.respond_draw(Color.BLACK, True)

    assert controller.get_state().result == ResultRecord(
        status="draw", reason="双方同意和棋"
    )
    with pytest.raises(ControlError, match="已经结束"):
        controller.make_move(Coord(0, 6), Coord(0, 5))
    controller.undo()
    assert controller.get_state().result is None


def test_pending_draw_pauses_play_until_opponent_responds() -> None:
    controller = GameController.new()
    controller.offer_draw(Color.RED)

    with pytest.raises(ControlError, match="和棋"):
        controller.make_move(Coord(0, 6), Coord(0, 5))

    controller.respond_draw(Color.BLACK, False)
    controller.make_move(Coord(0, 6), Coord(0, 5))
    assert controller.get_state().ply == 1


def test_callback_errors_are_isolated_and_do_not_rollback() -> None:
    controller = GameController.new()
    received = []
    controller.register_callback(
        lambda _event: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    controller.register_callback(received.append)

    event = controller.make_move(Coord(0, 6), Coord(0, 5))

    assert controller.get_state().ply == 1
    assert received == [event]
    assert len(controller.callback_errors) == 1


def test_external_control_claims_any_or_both_sides_and_conflicts() -> None:
    controller = GameController.new()
    red = controller.claim_side(Color.RED, "bot", ControllerKind.PYTHON)
    black = controller.claim_side(Color.BLACK, "bot", ControllerKind.NETWORK)

    with pytest.raises(ControlError, match="已被"):
        controller.claim_side(Color.RED, "other", ControllerKind.NETWORK)
    with pytest.raises(ControlError, match="控制权"):
        controller.make_move(Coord(0, 6), Coord(0, 5))
    controller.make_move(Coord(0, 6), Coord(0, 5), actor=red.token)
    controller.make_move(Coord(0, 3), Coord(0, 4), actor=black.token)

    controller.release_side(Color.RED, "bot", red.token)
    assert controller.get_state().controllers[Color.RED].kind is ControllerKind.HUMAN


def test_replay_cursor_is_read_only_then_branch_truncates_future() -> None:
    controller = GameController.from_record(
        _record(
            moves=(
                (Coord(0, 6), Coord(0, 5)),
                (Coord(0, 3), Coord(0, 4)),
                (Coord(2, 6), Coord(2, 5)),
            )
        )
    )
    original = controller.record

    replay = controller.set_replay_cursor(1)

    assert replay.replay_cursor == 1
    assert replay.board.at(Coord(0, 5)) is not None
    assert controller.record == original
    with pytest.raises(ControlError, match="回放"):
        controller.make_move(Coord(0, 3), Coord(0, 4))

    controller.branch_from_replay()
    assert controller.get_state().replay_cursor is None
    assert len(controller.record.moves) == 1
    controller.make_move(Coord(2, 3), Coord(2, 4))
    assert len(controller.record.moves) == 2


def test_from_record_preserves_ruleset_and_failed_load_is_transactional(
    tmp_path,
) -> None:
    controller = GameController.from_record(
        _record(
            ruleset=Ruleset.ASIAN_2003,
            moves=((Coord(0, 6), Coord(0, 5)),),
        )
    )
    before = controller.get_state()
    broken = tmp_path / "broken.json"
    broken.write_text('{"format_version": 1}', encoding="utf-8")

    with pytest.raises(ValueError):
        controller.load_record(broken)

    assert controller.get_state() == before
    assert controller.record.ruleset is Ruleset.ASIAN_2003


def test_record_loading_holds_the_same_write_lock_as_moves(monkeypatch) -> None:
    controller = GameController.new()
    validation_started = Event()
    allow_validation = Event()
    move_finished = Event()

    def slow_load(_path):
        validation_started.set()
        assert allow_validation.wait(2)
        return _record()

    monkeypatch.setattr("xiangqi.controller.load_and_validate", slow_load)
    loader = Thread(target=controller.load_record, args=("ignored.json",))
    mover = Thread(
        target=lambda: (
            controller.make_move(Coord(0, 6), Coord(0, 5)),
            move_finished.set(),
        )
    )

    loader.start()
    assert validation_started.wait(2)
    mover.start()
    assert not move_finished.wait(0.1)
    allow_validation.set()
    loader.join(2)
    mover.join(2)

    assert move_finished.is_set()
    assert controller.get_state().ply == 1


def test_import_rejects_incorrect_per_move_check_metadata() -> None:
    record = _record(moves=((Coord(0, 6), Coord(0, 5)),))
    tampered_move = record.moves[0].model_copy(update={"in_check": True})
    tampered = record.model_copy(update={"moves": (tampered_move,)})

    with pytest.raises(ControlError, match="将军"):
        GameController.from_record(tampered)


def test_import_rejects_incorrect_per_move_adjudication_metadata() -> None:
    record = _record(moves=((Coord(0, 6), Coord(0, 5)),))
    tampered_move = record.moves[0].model_copy(
        update={
            "adjudication": AdjudicationRecord(
                kind=AdjudicationKind.DRAW,
                reason="伪造裁决",
            )
        }
    )
    tampered = record.model_copy(update={"moves": (tampered_move,)})

    with pytest.raises(ControlError, match="裁决"):
        GameController.from_record(tampered)


@pytest.mark.parametrize("ruleset", list(Ruleset))
def test_responsible_side_cannot_repeat_a_prohibited_long_check(
    ruleset: Ruleset,
) -> None:
    fen = "4k4/3R5/9/9/9/4P4/9/9/9/4K4"
    cycle = (
        (Coord(3, 1), Coord(4, 1)),
        (Coord(4, 0), Coord(3, 0)),
        (Coord(4, 1), Coord(3, 1)),
        (Coord(3, 0), Coord(4, 0)),
    )
    controller = GameController.from_record(
        _record(fen=fen, ruleset=ruleset, moves=cycle * 2)
    )
    assert controller.get_state().adjudication is not None
    assert (
        controller.get_state().adjudication.kind
        is AdjudicationKind.MUST_CHANGE
    )

    before = controller.get_state()
    with pytest.raises(ControlError, match="变着"):
        controller.make_move(Coord(3, 1), Coord(4, 1))

    assert controller.get_state() == before


def test_imported_terminal_metadata_must_match_replayed_position() -> None:
    record = _record(moves=((Coord(0, 6), Coord(0, 5)),)).model_copy(
        update={
            "result": ResultRecord(
                status="red_win", reason="伪造", winner=Color.RED
            )
        }
    )

    with pytest.raises(ControlError, match="结果"):
        GameController.from_record(record)
