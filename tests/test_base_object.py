#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.utils.base_object import BaseObject


class _Events(BaseObject):
    def __init__(self):
        super().__init__()
        self._register_event_handler("on_sync_event", sync=True)


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
