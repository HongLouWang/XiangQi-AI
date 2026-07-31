from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

from fastapi.testclient import TestClient

from xiangqi.adjudication import Ruleset
from xiangqi.api import ControllerHub, EventBroker, create_api
from xiangqi.controller import GameController
from xiangqi.domain import Coord


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
            request_id="stale-move",
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
            request_id="valid-move",
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
        json=_command(
            request_id="wrong-release",
            token="wrong",
            expected_version=2,
        ),
    )
    assert denied.status_code == 409
    assert denied.json()["code"] == "control_error"

    released = client.post(
        "/control/red/release",
        json=_command(
            request_id="valid-release",
            token=token,
            expected_version=2,
        ),
    )
    assert released.status_code == 200


def test_illegal_move_and_invalid_payload_have_uniform_errors() -> None:
    client = TestClient(create_api(GameController.new()))
    claim = client.post("/control/red/claim", json=_command())
    token = claim.json()["token"]

    illegal = client.post(
        "/move",
        json=_command(
            request_id="illegal-move",
            token=token,
            expected_version=1,
            start=[0, 6],
            end=[1, 6],
        ),
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

    red_token = client.post("/control/red/claim", json=_command()).json()["token"]
    offered = client.post(
        "/draw/offer",
        json=_command(
            request_id="offer",
            side="red",
            token=red_token,
            expected_version=1,
        ),
    )
    assert offered.json()["event"]["kind"] == "draw_offered"
    refused = client.post(
        "/draw/respond",
        json=_command(
            request_id="unauthorized-response",
            side="black",
            accept=False,
            expected_version=2,
        ),
    )
    assert refused.status_code == 409

    black_claim = client.post(
        "/control/black/claim",
        json=_command(
            request_id="claim-black",
            controller_id="remote-2",
            expected_version=2,
        ),
    )
    black_token = black_claim.json()["token"]
    refused = client.post(
        "/draw/respond",
        json=_command(
            request_id="respond",
            controller_id="remote-2",
            token=black_token,
            side="black",
            accept=False,
            expected_version=3,
        ),
    )
    assert refused.json()["event"]["kind"] == "draw_responded"

    move = client.post(
        "/move",
        json=_command(
            request_id="move",
            token=red_token,
            start=[0, 6],
            end=[0, 5],
            expected_version=4,
        ),
    )
    assert move.status_code == 200
    undo = client.post(
        "/undo",
        json=_command(
            request_id="undo",
            token=red_token,
            steps=99,
            expected_version=5,
        ),
    )
    assert undo.json()["state"]["ply"] == 0

    destination = tmp_path / "game.json"
    exported = client.post(
        "/record/export",
        json=_command(
            request_id="export",
            token=red_token,
            path=str(destination),
            format="json",
            expected_version=6,
        ),
    )
    assert exported.status_code == 200
    loaded = client.post(
        "/record/import",
        json=_command(
            request_id="import",
            token=red_token,
            path=str(destination),
            format="json",
            expected_version=6,
        ),
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
                    "command": "claim",
                    **_command(side="red"),
                }
            )
            claimed = commander.receive_json()
            token = claimed["token"]
            commander.receive_json()
            assert observer.receive_json()["event"]["kind"] == "control_changed"
            commander.send_json(
                {
                    "command": "move",
                    **_command(
                        request_id="move",
                        token=token,
                        expected_version=1,
                        start=[0, 6],
                        end=[0, 5],
                    ),
                }
            )
            reply = commander.receive_json()
            observed = observer.receive_json()

    assert reply["type"] == "response"
    assert reply["request_id"] == "move"
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
                    request_id="release-black",
                    side="black",
                    token=token,
                    expected_version=1,
                ),
            }
        )
        released = socket.receive_json()

    assert claimed["ok"] is True
    assert token
    assert released["ok"] is True
    assert released["event"]["kind"] == "control_changed"


def test_websocket_terminal_event_survives_full_bounded_queue() -> None:
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


def test_event_broker_subscribe_publish_and_unsubscribe_are_thread_safe() -> None:
    from concurrent.futures import ThreadPoolExecutor

    broker = EventBroker(queue_size=4)

    def churn(index: int) -> None:
        subscriber = broker.subscribe()
        broker.publish(index)
        broker.unsubscribe(subscriber)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(churn, range(500)))

    assert broker.subscriber_count == 0


def test_unclaimed_http_and_websocket_moves_are_rejected() -> None:
    controller = GameController.new()
    client = TestClient(create_api(controller))

    denied = client.post("/move", json=_command(start=[0, 6], end=[0, 5]))
    assert denied.status_code == 409
    assert controller.get_state().ply == 0

    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        socket.send_json(
            {
                "command": "move",
                **_command(start=[0, 6], end=[0, 5]),
            }
        )
        reply = socket.receive_json()
    assert reply["ok"] is False
    assert controller.get_state().ply == 0


def test_duplicate_request_id_is_rejected_without_executing_twice() -> None:
    controller = GameController.new()
    client = TestClient(create_api(controller))
    claim = client.post("/control/red/claim", json=_command())
    token = claim.json()["token"]
    payload = _command(
        request_id="move-once",
        token=token,
        expected_version=1,
        start=[0, 6],
        end=[0, 5],
    )

    first = client.post("/move", json=payload)
    duplicate = client.post("/move", json=payload)

    assert first.status_code == 200
    assert duplicate.status_code == 409
    assert "request_id" in duplicate.json()["message"]
    assert controller.get_state().ply == 1


def test_event_contains_before_after_boards_and_complete_adjudication() -> None:
    controller = GameController.new()
    client = TestClient(create_api(controller))
    token = client.post("/control/red/claim", json=_command()).json()["token"]

    response = client.post(
        "/move",
        json=_command(
            request_id="move-with-state",
            token=token,
            expected_version=1,
            start=[0, 6],
            end=[0, 5],
        ),
    )
    event = response.json()["event"]

    assert event["before_board"]["0,6"]["kind"] == "pawn"
    assert "0,6" not in event["after_board"]
    assert event["after_board"]["0,5"]["kind"] == "pawn"
    assert set(event["adjudication"]) >= {
        "kind",
        "ruleset",
        "cycle_start",
        "responsible",
        "move_natures",
        "responsible_natures",
        "related_moves",
        "rule_reference",
        "reason",
    }


def test_controller_hub_switches_api_to_replacement_controller() -> None:
    original = GameController.new()
    hub = ControllerHub(original)
    client = TestClient(create_api(hub))
    replacement = GameController.new(ruleset=Ruleset.ASIAN_2003)

    hub.replace(replacement)

    assert client.get("/state").json()["ruleset"] == "asian_2003"
    assert client.get("/state").json()["version"] == 0


def test_network_import_accepts_explicit_json_and_text_formats(
    tmp_path,
) -> None:
    source = GameController.new()
    source.make_move(Coord(0, 6), Coord(0, 5))
    json_path = tmp_path / "record.data"
    text_path = tmp_path / "record.moves"
    source.export_record(json_path, "json")
    source.export_record(text_path, "text")

    hub = ControllerHub(GameController.new())
    client = TestClient(create_api(hub))
    token = client.post("/control/red/claim", json=_command()).json()["token"]
    for index, (path, format_name) in enumerate(
        ((json_path, "json"), (text_path, "text"))
    ):
        response = client.post(
            "/record/import",
            json=_command(
                request_id=f"import-{index}",
                token=token,
                expected_version=hub.current.get_state().version,
                path=str(path),
                format=format_name,
            ),
        )
        assert response.status_code == 200, response.json()
        assert hub.current.get_state().ply == 1


def test_text_import_preserves_lease_increments_version_and_allows_undo(
    tmp_path,
) -> None:
    source = GameController.new()
    source.make_move(Coord(0, 6), Coord(0, 5))
    text_path = tmp_path / "record.moves"
    source.export_record(text_path, "text")

    controller = GameController.new()
    client = TestClient(create_api(controller))
    claim = client.post("/control/red/claim", json=_command(request_id="claim-red"))
    token = claim.json()["token"]
    imported = client.post(
        "/record/import",
        json=_command(
            request_id="import-text",
            token=token,
            expected_version=1,
            path=str(text_path),
            format="text",
        ),
    )

    assert imported.status_code == 200, imported.json()
    assert imported.json()["event"]["kind"] == "record_loaded"
    assert imported.json()["version"] == 2
    assert imported.json()["state"]["controllers"]["red"] == {
        "kind": "network",
        "controller_id": "remote-1",
    }

    undone = client.post(
        "/undo",
        json=_command(
            request_id="undo-imported",
            token=token,
            expected_version=2,
        ),
    )
    assert undone.status_code == 200, undone.json()
    assert undone.json()["state"]["ply"] == 0


def test_text_import_broadcasts_record_loaded_over_websocket(tmp_path) -> None:
    source = GameController.new()
    source.make_move(Coord(0, 6), Coord(0, 5))
    text_path = tmp_path / "record.moves"
    source.export_record(text_path, "text")

    client = TestClient(create_api(GameController.new()))
    claim = client.post(
        "/control/red/claim", json=_command(request_id="claim-before-ws")
    )
    token = claim.json()["token"]
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        response = client.post(
            "/record/import",
            json=_command(
                request_id="import-for-ws",
                token=token,
                expected_version=1,
                path=str(text_path),
                format="text",
            ),
        )
        message = socket.receive_json()

    assert response.status_code == 200, response.json()
    assert message["type"] == "event"
    assert message["event"]["kind"] == "record_loaded"
    assert message["event"]["version"] == 2


def test_request_id_is_global_across_controller_identities_and_claims() -> None:
    client = TestClient(create_api(GameController.new()))
    first = client.post(
        "/control/red/claim",
        json=_command(request_id="global-id", controller_id="remote-a"),
    )
    duplicate = client.post(
        "/control/black/claim",
        json=_command(
            request_id="global-id",
            controller_id="remote-b",
            expected_version=1,
        ),
    )

    assert first.status_code == 200
    assert duplicate.status_code == 409
    assert "request_id" in duplicate.json()["message"]


def test_invalid_text_import_is_transactional_and_preserves_network_lease(
    tmp_path,
) -> None:
    invalid_path = tmp_path / "invalid.moves"
    invalid_path.write_text("这不是合法着法\n", encoding="utf-8")
    client = TestClient(create_api(GameController.new()))
    claim = client.post("/control/red/claim", json=_command(request_id="claim"))
    token = claim.json()["token"]

    failed = client.post(
        "/record/import",
        json=_command(
            request_id="bad-import",
            token=token,
            expected_version=1,
            path=str(invalid_path),
            format="text",
        ),
    )

    assert failed.status_code == 422
    state = client.get("/state").json()
    assert state["version"] == 1
    assert state["ply"] == 0
    assert state["controllers"]["red"] == {
        "kind": "network",
        "controller_id": "remote-1",
    }
    released = client.post(
        "/control/red/release",
        json=_command(
            request_id="release-after-failure",
            token=token,
            expected_version=1,
        ),
    )
    assert released.status_code == 200


def test_websocket_rejects_request_id_already_used_by_http() -> None:
    client = TestClient(create_api(GameController.new()))
    claimed = client.post(
        "/control/red/claim",
        json=_command(request_id="cross-protocol-id", controller_id="http-client"),
    )
    assert claimed.status_code == 200

    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        socket.send_json(
            {
                "command": "claim",
                **_command(
                    request_id="cross-protocol-id",
                    controller_id="ws-client",
                    expected_version=1,
                    side="black",
                ),
            }
        )
        response = socket.receive_json()

    assert response["type"] == "response"
    assert response["ok"] is False
    assert response["code"] == "invalid_command"
    assert "request_id" in response["message"]


def test_request_id_history_is_bounded_lru() -> None:
    app = create_api(GameController.new(), request_history_size=2)
    client = TestClient(app)
    claim = client.post(
        "/control/red/claim",
        json=_command(request_id="oldest", controller_id="remote-a"),
    )
    token = claim.json()["token"]
    for request_id in ("middle", "newest"):
        failed = client.post(
            "/undo",
            json=_command(
                request_id=request_id,
                controller_id="remote-a",
                token=token,
                expected_version=1,
            ),
        )
        assert failed.status_code == 409

    assert list(app.state.processed_requests) == ["middle", "newest"]
    reused_after_eviction = client.post(
        "/control/black/claim",
        json=_command(
            request_id="oldest",
            controller_id="remote-b",
            expected_version=1,
        ),
    )
    assert reused_after_eviction.status_code == 200


def test_duplicate_request_refreshes_lru_recency() -> None:
    app = create_api(GameController.new(), request_history_size=2)
    client = TestClient(app)
    oldest = client.post(
        "/control/red/claim",
        json=_command(request_id="oldest", controller_id="remote-a"),
    )
    red_token = oldest.json()["token"]
    client.post(
        "/control/black/claim",
        json=_command(
            request_id="middle",
            controller_id="remote-b",
            expected_version=1,
        ),
    )

    touched = client.post(
        "/undo",
        json=_command(
            request_id="oldest",
            controller_id="remote-a",
            token=red_token,
            expected_version=2,
        ),
    )
    client.post(
        "/undo",
        json=_command(
            request_id="newest",
            controller_id="remote-a",
            token=red_token,
            expected_version=2,
        ),
    )

    assert touched.status_code == 409
    assert list(app.state.processed_requests) == ["oldest", "newest"]


def test_in_flight_request_id_cannot_be_evicted_by_concurrent_http_claim(
    monkeypatch,
) -> None:
    controller = GameController.new()
    original_claim = controller.claim_side
    first_started = Event()
    allow_first_to_finish = Event()
    calls_lock = Lock()
    slow_calls = 0

    def delayed_claim(side, controller_id, *args, **kwargs):
        nonlocal slow_calls
        if controller_id == "slow-client":
            with calls_lock:
                slow_calls += 1
                is_first = slow_calls == 1
            if is_first:
                first_started.set()
                assert allow_first_to_finish.wait(timeout=5)
        return original_claim(side, controller_id, *args, **kwargs)

    monkeypatch.setattr(controller, "claim_side", delayed_claim)
    client = TestClient(create_api(controller, request_history_size=1))
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            client.post,
            "/control/red/claim",
            json=_command(request_id="in-flight", controller_id="slow-client"),
        )
        assert first_started.wait(timeout=5)
        pressure = client.post(
            "/control/black/claim",
            json=_command(request_id="pressure", controller_id="other-client"),
        )
        duplicate = client.post(
            "/control/red/claim",
            json=_command(
                request_id="in-flight",
                controller_id="slow-client",
                expected_version=1,
            ),
        )
        allow_first_to_finish.set()
        first.result(timeout=5)

    assert pressure.status_code == 200
    assert duplicate.status_code == 409
    assert "重复 request_id" in duplicate.json()["message"]
    assert slow_calls == 1


def test_leased_identity_is_enforced_for_move_and_global_mutations(
    tmp_path,
) -> None:
    controller = GameController.new()
    client = TestClient(create_api(controller))
    token = client.post("/control/red/claim", json=_command()).json()["token"]

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
