#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.utils.sync.event_notifier import EventNotifier


class TestEventNotifier(unittest.IsolatedAsyncioTestCase):
    async def test_two_waiters_one_notify(self):
        """Both waiters must wake from a single notify() (multi-consumer one-shot)."""
        notifier = EventNotifier()
        results = []

        async def waiter(label):
            await notifier.wait()
            results.append(label)

        t1 = asyncio.create_task(waiter("A"))
        t2 = asyncio.create_task(waiter("B"))

        # Let both waiters enter wait()
        await asyncio.sleep(0.01)

        await notifier.notify()

        # Let both waiters complete
        await asyncio.sleep(0.01)

        await t1
        await t2

        self.assertEqual(sorted(results), ["A", "B"])

    async def test_repeated_notify_coalescing(self):
        """Multiple notify() calls before wait() coalesce into one signal."""
        notifier = EventNotifier()

        await notifier.notify()
        await notifier.notify()
        await notifier.notify()

        # Should return immediately since event is already set
        done = False

        async def waiter():
            nonlocal done
            await notifier.wait()
            done = True

        await asyncio.wait_for(waiter(), timeout=1.0)
        self.assertTrue(done)

    async def test_single_consumer_repeated_cycles(self):
        """notify/wait/notify/wait works correctly in a loop."""
        notifier = EventNotifier()
        cycle_count = 0

        async def consumer():
            nonlocal cycle_count
            for _ in range(3):
                await notifier.wait()
                cycle_count += 1

        task = asyncio.create_task(consumer())

        for _ in range(3):
            await asyncio.sleep(0.01)
            await notifier.notify()

        await asyncio.wait_for(task, timeout=2.0)
        self.assertEqual(cycle_count, 3)


if __name__ == "__main__":
    unittest.main()
