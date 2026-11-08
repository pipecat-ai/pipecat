#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

import pytest

from pipecat.utils.base_object import BaseObject


class _Events(BaseObject):
    def __init__(self):
        super().__init__()
        self._register_event_handler("on_sync_event", sync=True)
        self._register_event_handler("on_background_event")


class TestBaseObjectEventHandlers(unittest.IsolatedAsyncioTestCase):
    async def test_handler_removing_itself_does_not_skip_the_next_handler(self):
        events = _Events()
        calls = []

        async def once(obj):
            calls.append("once")
            obj.remove_event_handler("on_sync_event", once)

        async def every_time(obj):
            calls.append("every_time")

        events.add_event_handler("on_sync_event", once)
        events.add_event_handler("on_sync_event", every_time)

        await events._call_event_handler("on_sync_event")
        await events._call_event_handler("on_sync_event")

        self.assertEqual(calls, ["once", "every_time", "every_time"])

    async def test_handler_added_during_an_event_runs_from_the_next_event(self):
        events = _Events()
        calls = []

        async def later(obj):
            calls.append("later")

        async def adder(obj):
            calls.append("adder")
            obj.add_event_handler("on_sync_event", later)

        events.add_event_handler("on_sync_event", adder)

        await events._call_event_handler("on_sync_event")
        self.assertEqual(calls, ["adder"])


@pytest.mark.asyncio
@pytest.mark.parametrize("event_name", ["on_sync_event", "on_background_event"])
@pytest.mark.parametrize("wrapped", [False, True], ids=["callable-instance", "sync-wrapper"])
async def test_event_handler_awaits_returned_coroutine(event_name, wrapped):
    events = _Events()
    calls = []

    class Handler:
        async def __call__(self, obj, value, *, suffix):
            await asyncio.sleep(0)
            calls.append((obj, value + suffix))

    handler = Handler()

    def wrapper(*args, **kwargs):
        return handler(*args, **kwargs)

    events.add_event_handler(event_name, wrapper if wrapped else handler)

    await events._call_event_handler(event_name, "hello", suffix="!")
    await events.cleanup()

    assert calls == [(events, "hello!")]


@pytest.mark.asyncio
@pytest.mark.parametrize("event_name", ["on_sync_event", "on_background_event"])
async def test_event_handler_runs_synchronous_handlers(event_name):
    events = _Events()
    calls = []

    def handler(obj):
        calls.append(obj)
        return "finished"

    events.add_event_handler(event_name, handler)

    await events._call_event_handler(event_name)
    await events.cleanup()

    assert calls == [events]
