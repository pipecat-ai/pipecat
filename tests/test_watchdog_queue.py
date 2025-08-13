#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.asyncio.watchdog_priority_queue import WatchdogPriorityQueue
from pipecat.utils.asyncio.watchdog_queue import WatchdogQueue


class TestWatchdogQueue(unittest.IsolatedAsyncioTestCase):
    async def test_simple_item(self):
        queue = WatchdogQueue(TaskManager())
        await queue.put(1)
        await queue.put(2)
        await queue.put(3)
        self.assertEqual(await queue.get(), 1)
        queue.task_done()
        self.assertEqual(await queue.get(), 2)
        queue.task_done()
        self.assertEqual(await queue.get(), 3)
        queue.task_done()

    async def test_watchdog_sentinel(self):
        queue = WatchdogQueue(TaskManager())
        await queue.put(1)
        self.assertEqual(await queue.get(), 1)
        queue.task_done()
        # The get should throw an exception.
        queue.cancel()
        try:
            await queue.get()
            assert False
        except asyncio.CancelledError:
            assert True


class TestWatchdogPriorityQueue(unittest.IsolatedAsyncioTestCase):
    async def test_simple_item(self):
        queue = WatchdogPriorityQueue(TaskManager(), tuple_size=2)
        await queue.put((3, 1))
        await queue.put((2, 1))
        await queue.put((1, 1))
        self.assertEqual(await queue.get(), (1, 1))
        queue.task_done()
        self.assertEqual(await queue.get(), (2, 1))
        queue.task_done()
        self.assertEqual(await queue.get(), (3, 1))
        queue.task_done()

    async def test_watchdog_sentinel(self):
        queue = WatchdogPriorityQueue(TaskManager(), tuple_size=2)
        await queue.put((0, 1))
        # The get should throw an exception because the watchdog sentinel has
        # higher priority.
        queue.cancel()
        try:
            await queue.get()
            assert False
        except asyncio.CancelledError:
            assert True
