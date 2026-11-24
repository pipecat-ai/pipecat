#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json

import pytest

from pipecat.services.soniox.stream_reconnect import (
    SonioxReconnectRefusedError,
    SonioxStreamReconnect,
)

OFFER = {"type": "reconnect", "reconnect_url": "wss://soniox/r", "reconnect_token": "t1"}


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _offered_stream(clock=None) -> SonioxStreamReconnect:
    stream = SonioxStreamReconnect(clock=clock or _Clock())
    assert stream.handle_message(OFFER)
    return stream


def _pong(ping: str) -> dict:
    return {"type": "pong", "id": json.loads(ping)["id"]}


def test_a_stream_is_resumable_once_soniox_offers_a_reconnect():
    stream = SonioxStreamReconnect()
    assert not stream.resumable

    stream.handle_message(OFFER)

    assert stream.resumable
    assert stream.url == "wss://soniox/r"


def test_a_stream_that_responds_without_an_offer_holds_nothing():
    stream = SonioxStreamReconnect()
    stream.hold(b"audio")

    assert not stream.handle_message({"tokens": []})
    stream.hold(b"audio")

    assert not stream.resumable
    assert stream.held_after(0) == []


def test_held_messages_are_numbered_from_one_in_send_order():
    stream = _offered_stream()

    for message in (b"a", '{"type": "finalize"}', b"b"):
        stream.hold(message)

    assert stream.held_after(1) == [(2, '{"type": "finalize"}'), (3, b"b")]


def test_a_pong_confirms_the_messages_sent_before_its_ping():
    stream = _offered_stream()
    stream.hold(b"a")
    stream.hold(b"b")
    ping = stream.ping()
    stream.hold(b"c")

    assert stream.handle_message(_pong(ping))

    assert stream.held_after(0) == [(3, b"c")]


def test_a_pong_answers_the_pings_sent_before_it():
    clock = _Clock()
    stream = _offered_stream(clock)
    stream.ping()
    clock.now = 1
    later = stream.ping()

    stream.handle_message(_pong(later))
    clock.now = 10

    assert not stream.ping_overdue(5)


def test_a_ping_is_overdue_once_its_pong_is_late():
    clock = _Clock()
    stream = _offered_stream(clock)
    stream.ping()

    clock.now = 4.9
    assert not stream.ping_overdue(5)
    clock.now = 5
    assert stream.ping_overdue(5)


def test_the_reconnect_request_carries_the_newest_token_and_last_seq():
    stream = _offered_stream()
    stream.handle_message({"tokens": [], "seq": 7})

    assert json.loads(stream.reconnect_request()) == {
        "type": "reconnect",
        "reconnect_token": "t1",
        "last_seq": 7,
    }


def test_resuming_keeps_only_the_messages_soniox_lacks_and_takes_the_new_token():
    stream = _offered_stream()
    for message in (b"a", b"b", b"c"):
        stream.hold(message)

    sent = stream.resumed({"type": "reconnected", "reconnect_token": "t2", "last_request": 2})

    assert sent == 2
    assert stream.held_after(0) == [(3, b"c")]
    assert json.loads(stream.reconnect_request())["reconnect_token"] == "t2"


def test_resuming_forgets_the_pings_of_the_old_connection():
    clock = _Clock()
    stream = _offered_stream(clock)
    stream.ping()

    stream.resumed({"type": "reconnected", "reconnect_token": "t2", "last_request": 0})
    clock.now = 10

    assert not stream.ping_overdue(5)


@pytest.mark.parametrize(
    "reply,error",
    [
        ({"error_type": "reconnect_failed", "error_code": 400}, SonioxReconnectRefusedError),
        ({"error_type": "service_unavailable", "error_code": 503}, ConnectionError),
    ],
)
def test_a_reconnect_soniox_does_not_accept_raises(reply, error):
    stream = _offered_stream()

    with pytest.raises(error):
        stream.resumed({"tokens": [], **reply})


def test_a_stream_past_its_reconnect_window_is_no_longer_resumable():
    clock = _Clock()
    stream = _offered_stream(clock)
    stream.hold(b"a")
    stream.connection_lost()

    clock.now = 20
    assert stream.resumable
    clock.now = 20.1
    stream.hold(b"b")

    assert not stream.resumable
    assert stream.held_after(0) == []


def test_the_reconnect_window_runs_from_the_first_drop_until_resumed():
    clock = _Clock()
    stream = _offered_stream(clock)
    stream.connection_lost()
    clock.now = 15
    stream.connection_lost()
    clock.now = 21

    assert not stream.resumable
