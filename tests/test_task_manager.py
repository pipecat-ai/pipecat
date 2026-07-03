#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the asyncio TaskManager."""

import asyncio
import inspect
import unittest

from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


class TestTaskManagerCreateTask(unittest.IsolatedAsyncioTestCase):
    """Tests for TaskManager.create_task() cancellation handling."""

    def _create_task_manager(self) -> TaskManager:
        task_manager = TaskManager()
        task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        return task_manager

    async def test_cancel_before_run_closes_coroutine(self):
        """A task cancelled before its coroutine starts must not leak it.

        Regression test: ``create_task`` wraps the coroutine in an inner
        ``run_coroutine()`` that only awaits it once that wrapper runs. If the
        task is cancelled before the wrapper reaches ``await coroutine``, the
        inner coroutine used to be dropped un-awaited, emitting
        ``RuntimeWarning: coroutine '...' was never awaited``.

        We assert on the coroutine's state directly rather than capturing the
        warning: CPython emits the never-awaited warning from the GC finalizer
        inside asyncio's managed context, which ``warnings.catch_warnings`` does
        not reliably intercept. ``create_task`` now closes the un-started
        coroutine in its done callback, so a fixed implementation leaves it in
        ``CORO_CLOSED``; an unfixed one leaves it in ``CORO_CREATED``.
        """
        task_manager = self._create_task_manager()

        async def never_runs():
            await asyncio.sleep(0)

        coro = never_runs()
        task = task_manager.create_task(coro, "never_runs")
        # Cancel before the event loop ever steps run_coroutine().
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(
            inspect.getcoroutinestate(coro),
            inspect.CORO_CLOSED,
            "create_task left a coroutine un-awaited (still in CORO_CREATED state)",
        )

    async def test_cancel_after_start_propagates_into_coroutine(self):
        """A started-then-cancelled task must still run the coroutine's cleanup.

        The fix for the pre-start case must not force-close coroutines that have
        already begun running — cancellation has to propagate into them so their
        ``finally``/``except CancelledError`` cleanup executes.
        """
        task_manager = self._create_task_manager()
        cleanup_ran = asyncio.Event()

        async def long_handler():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cleanup_ran.set()
                raise

        task = task_manager.create_task(long_handler(), "long_handler")
        # Let the coroutine start and suspend at the sleep before cancelling.
        # A single event-loop yield is enough: the task is already queued, so it
        # runs through to its first real suspension (asyncio.sleep(10)).
        await asyncio.sleep(0)
        await task_manager.cancel_task(task)

        self.assertTrue(cleanup_ran.is_set())

    async def test_normal_completion_returns_value(self):
        """A coroutine that runs to completion still returns its result."""
        task_manager = self._create_task_manager()

        async def returns_value():
            return 42

        task = task_manager.create_task(returns_value(), "returns_value")
        self.assertEqual(await task, 42)


if __name__ == "__main__":
    unittest.main()
