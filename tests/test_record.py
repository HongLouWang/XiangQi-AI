import json
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from xiangqi.adjudication import AdjudicationKind, MoveNature, Ruleset
from xiangqi.board import Board
from xiangqi.domain import Color, Coord, Move
from xiangqi.notation import format_move
from xiangqi.record import (
    AdjudicationRecord,
    DrawEventRecord,
    GameRecord,
    MoveRecord,
    PlayerRecord,
    RecordError,
    ResultRecord,
    export_json,
    load_and_validate,
    load_json,
)


def _sample_record(ruleset: Ruleset = Ruleset.CHINESE_2020) -> GameRecord:
    board = Board.standard()
    side = Color.RED
    move_specs = (
        (Coord(1, 7), Coord(1, 0)),
        (Coord(7, 0), Coord(6, 2)),
        (Coord(1, 0), Coord(1, 1)),
    )
    moves: list[MoveRecord] = []
    for start, end in move_specs:
        piece = board.at(start)
        assert piece is not None
        move = Move(start, end, piece, board.at(end))
        notation = format_move(board, move)
        next_board = board.move_unchecked(start, end)
        moves.append(
            MoveRecord.from_move(
                move,
                notation=notation,
                position_after=next_board.position_key(side.opponent),
                adjudication={
                    "kind": AdjudicationKind.NO_DECISION,
                    "ruleset": ruleset,
                    "cycle_start": None,
                    "move_natures": (),
                    "responsible_natures": (),
                    "related_moves": (),
                    "reason": "尚无裁决",
                },
            )
        )
        board = next_board
        side = side.opponent
    return GameRecord(
        created_at=datetime(2026, 7, 31, tzinfo=UTC),
        ruleset=ruleset,
        players=(
            PlayerRecord(name="玩家1", color=Color.RED),
            PlayerRecord(name="玩家2", color=Color.BLACK),
        ),
        initial_fen=Board.standard().to_fen(),
        moves=tuple(moves),
        result=ResultRecord(status="ongoing"),
    )


def test_json_record_round_trip_preserves_rules_moves_and_result(tmp_path) -> None:
    record = _sample_record(ruleset=Ruleset.ASIAN_2003)
    path = tmp_path / "game.xqjson"

    export_json(record, path)

    assert load_json(path) == record
    assert load_and_validate(path) == record


def test_load_validation_replays_and_rejects_tampered_position(tmp_path) -> None:
    record = _sample_record()
    payload = record.model_dump(mode="json")
    payload["moves"][1]["position_after"] = "0" * 64
    path = tmp_path / "tampered.xqjson"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RecordError, match="第 2 步"):
        load_and_validate(path)

    assert record == _sample_record()


def test_load_rejects_illegal_move_without_replacing_existing_record(tmp_path) -> None:
    current = _sample_record()
    payload = current.model_dump(mode="json")
    payload["moves"][1]["end"] = [8, 8]
    path = tmp_path / "illegal.xqjson"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RecordError, match="第 2 步"):
        load_and_validate(path)

    assert current == _sample_record()


@pytest.mark.parametrize(
    "contents",
    [
        "{not json",
        '{"format_version": 999}',
        '{"format_version": 1, "ruleset": "unknown"}',
    ],
)
def test_damaged_or_unsupported_json_is_wrapped_as_record_error(
    tmp_path, contents: str
) -> None:
    path = tmp_path / "broken.xqjson"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(RecordError):
        load_json(path)


def test_models_reject_duplicate_player_colors() -> None:
    with pytest.raises(ValidationError):
        GameRecord(
            created_at=datetime(2026, 7, 31, tzinfo=UTC),
            players=(
                PlayerRecord(name="甲", color=Color.RED),
                PlayerRecord(name="乙", color=Color.RED),
            ),
            initial_fen=Board.standard().to_fen(),
        )


def test_export_does_not_leave_named_temporary_file(tmp_path) -> None:
    path = tmp_path / "game.xqjson"
    export_json(_sample_record(), path)
    assert [item for item in tmp_path.iterdir() if item != path] == []


def test_created_at_is_required_and_timezone_aware() -> None:
    payload = _sample_record().model_dump()
    payload.pop("created_at")
    with pytest.raises(ValidationError):
        GameRecord.model_validate(payload)

    payload["created_at"] = datetime(2026, 7, 31)  # noqa: DTZ001 - invalid fixture
    with pytest.raises(ValidationError, match="时区"):
        GameRecord.model_validate(payload)


def test_created_at_with_nonzero_offset_is_normalized_to_utc() -> None:
    payload = _sample_record().model_dump()
    payload["created_at"] = datetime(
        2026, 7, 31, 12, tzinfo=timezone(timedelta(hours=8))
    )

    record = GameRecord.model_validate(payload)

    assert record.created_at == datetime(2026, 7, 31, 4, tzinfo=UTC)
    assert record.created_at.tzinfo is UTC


def test_adjudication_record_round_trip_preserves_replay_evidence(tmp_path) -> None:
    evidence = AdjudicationRecord(
        kind=AdjudicationKind.MUST_CHANGE,
        ruleset=Ruleset.ASIAN_2003,
        cycle_start=4,
        move_natures=(MoveNature.CHECK, MoveNature.IDLE) * 2,
        responsible_natures=(MoveNature.CHECK,) * 2,
        related_moves=(5, 6, 7, 8),
        reason="红方长将",
        responsible=Color.RED,
        rule_reference="亚洲棋规",
    )
    record = _sample_record(Ruleset.ASIAN_2003)
    move = record.moves[-1].model_copy(update={"adjudication": evidence})
    record = record.model_copy(update={"moves": (*record.moves[:-1], move)})
    path = tmp_path / "evidence.xqjson"

    export_json(record, path)
    loaded = load_json(path)

    assert loaded.moves[-1].adjudication == evidence


def test_negotiated_draw_requires_ordered_offer_and_opponent_acceptance() -> None:
    base = _sample_record().model_dump()
    base["result"] = {
        "status": "draw",
        "reason": "双方同意和棋",
        "winner": None,
    }

    for events in (
        [],
        [{"action": "accept", "actor": "black", "ply": 3}],
        [
            {"action": "offer", "actor": "red", "ply": 3},
            {"action": "accept", "actor": "red", "ply": 3},
        ],
    ):
        base["draw_events"] = events
        with pytest.raises(ValidationError, match="和棋"):
            GameRecord.model_validate(base)

    valid = GameRecord.model_validate(
        {
            **base,
            "draw_events": [
                DrawEventRecord(action="offer", actor=Color.RED, ply=3),
                DrawEventRecord(action="accept", actor=Color.BLACK, ply=3),
            ],
        }
    )
    assert valid.result.status == "draw"


def test_draw_rejection_must_follow_an_opponent_offer() -> None:
    record = _sample_record()
    with pytest.raises(ValidationError, match="和棋"):
        GameRecord.model_validate(
            {
                **record.model_dump(),
                "draw_events": [
                    {"action": "offer", "actor": "black", "ply": 1},
                    {"action": "reject", "actor": "black", "ply": 1},
                ],
            }
        )


def test_draw_event_ply_cannot_move_backwards() -> None:
    record = _sample_record()
    payload = record.model_dump()
    payload["draw_events"] = [
        {"action": "offer", "actor": "red", "ply": 2},
        {"action": "reject", "actor": "black", "ply": 2},
        {"action": "offer", "actor": "black", "ply": 1},
        {"action": "reject", "actor": "red", "ply": 1},
    ]

    with pytest.raises(ValidationError, match="顺序"):
        GameRecord.model_validate(payload)
