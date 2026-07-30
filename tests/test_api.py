from __future__ import annotations

from fastapi.testclient import TestClient

from xiangqi.api import create_api
from xiangqi.controller import GameController


def _command(**values):
    return {
        "request_id": "req-1",
        "controller_id": "remote-1",
        "expected_version": 0,
        **values,
    }


def test_http_state_legal_moves_and_openapi_are_serializable() -> None:
    client = TestClient(create_api(GameController.new()))

    state = client.get("/state")
    legal = client.get("/legal-moves")

    assert state.status_code == 200
    assert state.json()["side_to_move"] == "red"
    assert state.json()["board"]["0,6"] == {"color": "red", "kind": "pawn"}
    assert [0, 5] in legal.json()["moves"]["0,6"]
    assert client.get("/docs").status_code == 200


def test_claim_move_release_uses_secret_token_and_versions() -> None:
    client = TestClient(create_api(GameController.new()))
    claim = client.post("/control/red/claim", json=_command())

    assert claim.status_code == 200
    token = claim.json()["token"]
    assert token
    assert claim.json()["version"] == 1

    stale = client.post(
        "/move",
        json=_command(
            token=token,
            start=[0, 6],
            end=[0, 5],
        ),
    )
    assert stale.status_code == 409
    assert stale.json()["code"] == "stale_version"

    moved = client.post(
        "/move",
        json=_command(
            token=token,
            expected_version=1,
            start=[0, 6],
            end=[0, 5],
        ),
    )
    assert moved.status_code == 200
    assert moved.json()["event"]["kind"] == "move"
    assert moved.json()["version"] == 2

    denied = client.post(
        "/control/red/release",
        json=_command(token="wrong", expected_version=2),
    )
    assert denied.status_code == 409
    assert denied.json()["code"] == "control_error"

    released = client.post(
        "/control/red/release",
        json=_command(token=token, expected_version=2),
    )
    assert released.status_code == 200


def test_illegal_move_and_invalid_payload_have_uniform_errors() -> None:
    client = TestClient(create_api(GameController.new()))

    illegal = client.post(
        "/move",
        json=_command(start=[0, 6], end=[1, 6]),
    )
    invalid = client.post("/move", json={"request_id": "bad"})

    assert illegal.status_code == 422
    assert illegal.json()["code"] == "invalid_command"
    assert set(illegal.json()) == {"code", "message", "details"}
    assert invalid.status_code == 422
    assert invalid.json()["code"] == "validation_error"


def test_draw_undo_and_record_round_trip(tmp_path) -> None:
    controller = GameController.new()
    client = TestClient(create_api(controller))

    offered = client.post(
        "/draw/offer",
        json=_command(side="red"),
    )
    assert offered.json()["event"]["kind"] == "draw_offered"
    refused = client.post(
        "/draw/respond",
        json=_command(
            side="black", accept=False, expected_version=1
        ),
    )
    assert refused.json()["event"]["kind"] == "draw_responded"

    move = client.post(
        "/move",
        json=_command(
            start=[0, 6], end=[0, 5], expected_version=2
        ),
    )
    assert move.status_code == 200
    undo = client.post(
        "/undo",
        json=_command(steps=99, expected_version=3),
    )
    assert undo.json()["state"]["ply"] == 0

    destination = tmp_path / "game.json"
    exported = client.post(
        "/record/export",
        json=_command(
            path=str(destination), format="json", expected_version=4
        ),
    )
    assert exported.status_code == 200
    loaded = client.post(
        "/record/import",
        json=_command(path=str(destination), expected_version=4),
    )
    assert loaded.status_code == 200
    assert loaded.json()["event"]["kind"] == "record_loaded"


def test_websocket_observers_receive_events_and_can_submit_commands() -> None:
    app = create_api(GameController.new(), event_queue_size=2)
    client = TestClient(app)

    with client.websocket_connect("/ws") as observer:
        assert observer.receive_json()["type"] == "ready"
        with client.websocket_connect("/ws") as commander:
            assert commander.receive_json()["type"] == "ready"
            commander.send_json(
                {
                    "command": "move",
                    **_command(start=[0, 6], end=[0, 5]),
                }
            )
            reply = commander.receive_json()
            observed = observer.receive_json()

    assert reply["type"] == "response"
    assert reply["request_id"] == "req-1"
    assert reply["ok"] is True
    assert observed["type"] == "event"
    assert observed["event"]["kind"] == "move"


def test_websocket_can_claim_and_release_either_side() -> None:
    client = TestClient(create_api(GameController.new()))

    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        socket.send_json(
            {
                "command": "claim",
                **_command(side="black"),
            }
        )
        claimed = socket.receive_json()
        token = claimed["token"]
        # Consume the control_changed event before submitting the next command.
        assert socket.receive_json()["event"]["kind"] == "control_changed"
        socket.send_json(
            {
                "command": "release",
                **_command(
                    side="black", token=token, expected_version=1
                ),
            }
        )
        released = socket.receive_json()

    assert claimed["ok"] is True
    assert token
    assert released["ok"] is True
    assert released["event"]["kind"] == "control_changed"


def test_websocket_terminal_event_survives_full_bounded_queue() -> None:
    from xiangqi.api import EventBroker

    broker = EventBroker(queue_size=1)
    queue = broker.subscribe()
    stale_event = object()
    broker.publish(stale_event)

    # The broker accepts arbitrary objects internally and must replace a stale
    # queued item when a terminal event is published.
    class Terminal:
        result = object()

    terminal = Terminal()
    broker.publish(terminal)
    broker.publish(object())
    assert queue.get_nowait() is terminal


def test_leased_identity_is_enforced_for_move_and_global_mutations(
    tmp_path,
) -> None:
    controller = GameController.new()
    client = TestClient(create_api(controller))
    token = client.post(
        "/control/red/claim", json=_command()
    ).json()["token"]

    wrong_identity = client.post(
        "/move",
        json=_command(
            controller_id="imposter",
            token=token,
            start=[0, 6],
            end=[0, 5],
            expected_version=1,
        ),
    )
    unauthenticated_undo = client.post(
        "/undo",
        json=_command(expected_version=1),
    )
    unauthenticated_import = client.post(
        "/record/import",
        json=_command(path=str(tmp_path / "missing.json"), expected_version=1),
    )

    assert wrong_identity.status_code == 409
    assert unauthenticated_undo.status_code == 409
    assert unauthenticated_import.status_code == 409


def test_websocket_non_object_payload_returns_structured_error() -> None:
    client = TestClient(create_api(GameController.new()))

    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        socket.send_json([])
        response = socket.receive_json()

    assert response["type"] == "response"
    assert response["ok"] is False
    assert response["code"] == "invalid_command"
