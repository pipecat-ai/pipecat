#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for :mod:`pipecat.utils.asyncio.task_manager`."""

import asyncio
import unittest

from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


def _make_task_manager() -> TaskManager:
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    return tm


class TestCancelTask(unittest.IsolatedAsyncioTestCase):
    """Behaviour of :meth:`TaskManager.cancel_task`."""

    async def test_cancels_another_task(self):
        """A task cancelled from outside its own coroutine completes promptly."""
        tm = _make_task_manager()
        started = asyncio.Event()

        async def _runner():
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                raise

        task = tm.create_task(_runner(), name="runner")
        await started.wait()
        await tm.cancel_task(task)
        self.assertTrue(task.done())
        self.assertTrue(task.cancelled())

    async def test_self_cancellation_unwinds_caller(self):
        """A task that cancels itself via ``cancel_task`` must not continue
        running once ``cancel_task`` returns to the awaiter.

        Previously ``cancel_task`` called ``task.cancel()`` and then
        ``await task`` on the current task, which caused the in-flight
        ``CancelledError`` to be raised at the ``await`` and immediately
        swallowed by the ``except asyncio.CancelledError`` arm. Control flowed
        back to the caller with the cancellation suppressed, so the task body
        continued executing past the supposed cancellation point. This test
        asserts that the body unwinds and does not reach lines after the
        ``cancel_task`` call.
        """
        tm = _make_task_manager()
        reached_after_cancel = False
        finally_ran = False

        async def _runner():
            nonlocal reached_after_cancel, finally_ran
            try:
                current = asyncio.current_task()
                assert current is not None
                await tm.cancel_task(current)
                # Must be unreachable: the task cancelled itself, so cancel_task
                # should not return normally here.
                reached_after_cancel = True
            finally:
                finally_ran = True

        task = tm.create_task(_runner(), name="self-cancel")
        # Wait for the task to finish unwinding.
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.CancelledError:
            pass
        self.assertTrue(task.done())
        self.assertTrue(task.cancelled())
        self.assertFalse(
            reached_after_cancel,
            "code after cancel_task ran — self-cancellation was silently suppressed",
        )
        self.assertTrue(finally_ran, "finally block did not run during unwind")


if __name__ == "__main__":
    unittest.main()
