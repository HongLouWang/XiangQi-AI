import json

import pytest
from pydantic import ValidationError

from xiangqi.adjudication import AdjudicationKind, Ruleset
from xiangqi.board import Board
from xiangqi.domain import Color, Coord, Move
from xiangqi.notation import format_move
from xiangqi.record import (
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
                    "reason": "尚无裁决",
                },
            )
        )
        board = next_board
        side = side.opponent
    return GameRecord(
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
