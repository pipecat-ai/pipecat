#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json

import pytest

from pipecat.services.soniox.stream_reconnect import (
    SonioxReconnectRefusedError,
    SonioxReconnectTracker,
)

RECONNECT_OFFER = {"type": "reconnect", "reconnect_url": "wss://soniox/r", "reconnect_token": "t1"}


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _tracker_with_offer(clock=None) -> SonioxReconnectTracker:
    tracker = SonioxReconnectTracker(clock=clock or _Clock())
    assert tracker.handle_server_message(RECONNECT_OFFER)
    return tracker


def _pong(ping: str) -> dict:
    return {"type": "pong", "id": json.loads(ping)["id"]}


def test_can_reconnect_once_soniox_offers_a_reconnect():
    tracker = SonioxReconnectTracker()
    assert not tracker.can_reconnect

    tracker.handle_server_message(RECONNECT_OFFER)

    assert tracker.can_reconnect
    assert tracker.reconnect_url == "wss://soniox/r"


def test_a_response_before_any_offer_disables_reconnect():
    tracker = SonioxReconnectTracker()
    tracker.track_sent(b"audio")

    assert not tracker.handle_server_message({"tokens": []})
    tracker.track_sent(b"audio")

    assert not tracker.can_reconnect
    assert tracker.unacknowledged_after(0) == []


def test_sent_messages_are_numbered_from_one_in_send_order():
    tracker = _tracker_with_offer()

    for message in (b"a", '{"type": "finalize"}', b"b"):
        tracker.track_sent(message)

    assert tracker.unacknowledged_after(1) == [(2, '{"type": "finalize"}'), (3, b"b")]


def test_a_pong_acknowledges_the_messages_sent_before_its_ping():
    tracker = _tracker_with_offer()
    tracker.track_sent(b"a")
    tracker.track_sent(b"b")
    ping = tracker.create_ping()
    tracker.track_sent(b"c")

    assert tracker.handle_server_message(_pong(ping))

    assert tracker.unacknowledged_after(0) == [(3, b"c")]


def test_a_pong_also_answers_earlier_pings():
    clock = _Clock()
    tracker = _tracker_with_offer(clock)
    tracker.create_ping()
    clock.now = 1
    later = tracker.create_ping()

    tracker.handle_server_message(_pong(later))
    clock.now = 10

    assert not tracker.is_pong_overdue(5)


def test_pong_is_overdue_after_the_timeout():
    clock = _Clock()
    tracker = _tracker_with_offer(clock)
    tracker.create_ping()

    clock.now = 4.9
    assert not tracker.is_pong_overdue(5)
    clock.now = 5
    assert tracker.is_pong_overdue(5)


def test_reconnect_request_carries_the_newest_token_and_last_seq():
    tracker = _tracker_with_offer()
    tracker.handle_server_message({"tokens": [], "seq": 7})

    assert json.loads(tracker.create_reconnect_request()) == {
        "type": "reconnect",
        "reconnect_token": "t1",
        "last_seq": 7,
    }


def test_reconnect_reply_acknowledges_received_messages_and_updates_the_token():
    tracker = _tracker_with_offer()
    for message in (b"a", b"b", b"c"):
        tracker.track_sent(message)

    received_count = tracker.handle_reconnect_reply(
        {"type": "reconnected", "reconnect_token": "t2", "last_request": 2}
    )

    assert received_count == 2
    assert tracker.unacknowledged_after(0) == [(3, b"c")]
    assert json.loads(tracker.create_reconnect_request())["reconnect_token"] == "t2"


def test_reconnect_reply_clears_pending_pings():
    clock = _Clock()
    tracker = _tracker_with_offer(clock)
    tracker.create_ping()

    tracker.handle_reconnect_reply(
        {"type": "reconnected", "reconnect_token": "t2", "last_request": 0}
    )
    clock.now = 10

    assert not tracker.is_pong_overdue(5)


@pytest.mark.parametrize(
    "reply,error",
    [
        ({"error_type": "reconnect_failed", "error_code": 400}, SonioxReconnectRefusedError),
        ({"error_type": "service_unavailable", "error_code": 503}, ConnectionError),
    ],
)
def test_failed_reconnect_reply_raises(reply, error):
    tracker = _tracker_with_offer()

    with pytest.raises(error):
        tracker.handle_reconnect_reply({"tokens": [], **reply})


def test_reconnect_is_disabled_after_the_window_expires():
    clock = _Clock()
    tracker = _tracker_with_offer(clock)
    tracker.track_sent(b"a")
    tracker.mark_disconnected()

    clock.now = 20
    assert tracker.can_reconnect
    clock.now = 20.1
    tracker.track_sent(b"b")

    assert not tracker.can_reconnect
    assert tracker.unacknowledged_after(0) == []


def test_reconnect_window_starts_at_the_first_disconnect():
    clock = _Clock()
    tracker = _tracker_with_offer(clock)
    tracker.mark_disconnected()
    clock.now = 15
    tracker.mark_disconnected()
    clock.now = 21

    assert not tracker.can_reconnect
