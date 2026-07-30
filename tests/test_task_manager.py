#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the asyncio TaskManager."""

import asyncio
import inspect
import io
import unittest

from loguru import logger

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

    async def test_cancel_timeout_logs_where_the_task_is_blocked(self):
        """A slow cancel must name the await that is holding it up.

        A task that is slow to cancel blocks whatever cancelled it. For a frame
        processor's process task that is the inline cancel inside
        `_start_interruption`, so barge-in propagation stalls for the whole
        duration — and source reading alone cannot say which library call is at
        fault. The warning therefore has to carry a file:line.

        The stack must be sampled while the task is still pending. Reading it
        after an `asyncio.wait_for` TimeoutError never works: wait_for waits for
        the cancellation to complete before raising, so by then the coroutine
        has no frame left.
        """
        task_manager = self._create_task_manager()
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()

        async def slow_to_cancel():
            try:
                await asyncio.sleep(10)
            finally:
                # Un-cancellable cleanup, e.g. an HTTP stream aclose() over a
                # dead socket. Shielded so the cancellation in flight does not
                # cut it short.
                cleanup_started.set()
                await asyncio.shield(release_cleanup.wait())

        task = task_manager.create_task(slow_to_cancel(), "slow_to_cancel")
        await asyncio.sleep(0)

        async def release_after_the_threshold():
            await cleanup_started.wait()
            await asyncio.sleep(0.2)
            release_cleanup.set()

        releaser = asyncio.create_task(release_after_the_threshold())

        log_output = io.StringIO()
        handler_id = logger.add(log_output, level="WARNING", format="{message}")
        try:
            await task_manager.cancel_task(task, timeout=0.05)
        finally:
            logger.remove(handler_id)
            release_cleanup.set()
            await releaser

        log_text = log_output.getvalue()
        self.assertIn("timed out waiting for task to cancel", log_text)
        self.assertIn("blocked at:", log_text)
        # The frame must name the blocked coroutine, not just this module's
        # task wrapper — `Task.get_stack()` alone only reports the latter.
        self.assertIn("test_task_manager.py:", log_text)
        self.assertIn("in slow_to_cancel", log_text)
        # And the true cost of the cancel — what actually serializes barge-in.
        self.assertIn("cancel completed after", log_text)


if __name__ == "__main__":
    unittest.main()
