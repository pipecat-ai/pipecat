#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression tests for asyncio.Event race conditions (pipecat#3402)."""

import asyncio
import unittest

from pipecat.utils.sync.event_notifier import EventNotifier


class TestEventNotifierRace(unittest.IsolatedAsyncioTestCase):
    async def test_multi_consumer_no_lost_notification(self):
        """Verify that a notification arriving between two waiters' wake-ups
        is not lost by the second waiter's clear()."""
        notifier = EventNotifier()
        results = []

        async def waiter(name: str):
            await notifier.wait()
            results.append(name)

        # Two consumers waiting
        t1 = asyncio.create_task(waiter("A"))
        t2 = asyncio.create_task(waiter("B"))
        await asyncio.sleep(0.01)

        # First notify wakes one consumer
        await notifier.notify()
        await asyncio.sleep(0.01)

        # Second notify should wake the other consumer (not be lost)
        await notifier.notify()
        await asyncio.sleep(0.01)

        # Both should have completed
        assert "A" in results, f"Waiter A was not notified. Results: {results}"
        assert "B" in results, f"Waiter B was not notified. Results: {results}"

        t1.cancel()
        t2.cancel()
        try:
            await t1
        except asyncio.CancelledError:
            pass
        try:
            await t2
        except asyncio.CancelledError:
            pass

    async def test_single_consumer_basic(self):
        """Basic single-consumer notify/wait cycle works."""
        notifier = EventNotifier()
        received = asyncio.Event()

        async def consumer():
            await notifier.wait()
            received.set()

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)

        await notifier.notify()
        await asyncio.sleep(0.01)

        assert received.is_set(), "Single consumer did not receive notification"
        await task


class TestIdleProcessorRace(unittest.IsolatedAsyncioTestCase):
    async def test_activity_during_async_callback_not_lost(self):
        """Regression test for pipecat#3402.

        Scenario: timeout fires → async callback starts → activity set()
        during callback → callback ends → next iteration checks event.

        Buggy: finally:clear() erases the set() → next iteration sees cleared
        event → wait_for times out → false-positive callback.
        Fixed: clear() only in try → set() survives → next iteration's wait()
        returns immediately (no timeout) → signal consumed correctly.

        We instrument the handler to record whether the iteration right after
        the first callback saw the event (success path) or timed out (timeout path).
        """
        timeout = 0.1

        async def run_pattern(use_finally_clear: bool) -> bool:
            """Returns True if the activity signal was seen after callback."""
            idle_event = asyncio.Event()
            cb_started = asyncio.Event()
            signal_seen_after_cb = asyncio.Event()
            timed_out_after_cb = asyncio.Event()

            async def handler():
                first_cb_done = False
                while True:
                    try:
                        await asyncio.wait_for(idle_event.wait(), timeout=timeout)
                        if not use_finally_clear:
                            idle_event.clear()
                        if first_cb_done:
                            signal_seen_after_cb.set()
                            return
                    except asyncio.TimeoutError:
                        if not first_cb_done:
                            cb_started.set()
                            await asyncio.sleep(timeout * 0.5)
                            first_cb_done = True
                        else:
                            timed_out_after_cb.set()
                            return
                    finally:
                        if use_finally_clear:
                            idle_event.clear()

            task = asyncio.create_task(handler())

            await asyncio.wait_for(cb_started.wait(), timeout=2.0)
            idle_event.set()

            done, _ = await asyncio.wait(
                [
                    asyncio.create_task(signal_seen_after_cb.wait()),
                    asyncio.create_task(timed_out_after_cb.wait()),
                ],
                timeout=2.0,
                return_when=asyncio.FIRST_COMPLETED,
            )

            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            return signal_seen_after_cb.is_set()

        buggy_saw_signal = await run_pattern(use_finally_clear=True)
        fixed_saw_signal = await run_pattern(use_finally_clear=False)

        assert not buggy_saw_signal, (
            "Buggy pattern should NOT see the signal (finally:clear erases it)"
        )
        assert fixed_saw_signal, (
            "Fixed pattern should see the signal (set during callback survives)"
        )


if __name__ == "__main__":
    unittest.main()
