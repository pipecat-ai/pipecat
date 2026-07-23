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


class TestTaskManagerCancelTask(unittest.IsolatedAsyncioTestCase):
    """Tests for TaskManager.cancel_task() cancellation handling."""

    def _create_task_manager(self) -> TaskManager:
        return TaskManager(loop=asyncio.get_running_loop())

    async def test_caller_cancellation_propagates(self):
        """``cancel_task()`` must not swallow the caller's own cancellation.

        Regression test: ``cancel_task`` awaits the cancelled task and treats
        any ``CancelledError`` raised at that point as proof that ``task``
        finished cancelling. But if the *caller* of ``cancel_task`` is itself
        cancelled while suspended there, the ``CancelledError`` is the
        caller's own cancellation — and ``except CancelledError: pass``
        swallowed it, so the caller resumed as if it was never cancelled.

        Services awaiting ``cancel_task()`` from a ``finally`` block during
        teardown (e.g. ``DeepgramSTTService._connection_handler`` cancelling
        its keepalive task) would then survive the pipeline's cancellation
        and, being reconnect loops, reconnect after having been cancelled —
        leaving an orphaned task running forever (observed in production as
        multi-day zombie reconnect loops).

        No assertion is made about the child's fate: cancelling a task that
        is suspended at ``await child`` also cancels ``child`` itself (asyncio
        propagates cancellation into the awaited future), so the child dies
        promptly in both the buggy and the fixed implementation. The bug is
        strictly about the caller's fate.
        """
        task_manager = self._create_task_manager()

        child_started = asyncio.Event()
        release_child_cleanup = asyncio.Event()

        async def slow_dying_child():
            child_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                # Cleanup that outlives a single cancel — models a websocket
                # close handshake or a send blocked on a dead socket.
                await release_child_cleanup.wait()

        child = task_manager.create_task(slow_dying_child(), "slow_dying_child")
        await child_started.wait()

        caller_resumed_after_own_cancel = False

        async def caller():
            nonlocal caller_resumed_after_own_cancel
            await task_manager.cancel_task(child)
            caller_resumed_after_own_cancel = True

        caller_task = asyncio.get_running_loop().create_task(caller())
        # Let the caller suspend at cancel_task's `await task`.
        await asyncio.sleep(0.05)
        # Cancel the CALLER, not the child.
        caller_task.cancel()
        await asyncio.sleep(0.05)

        try:
            self.assertTrue(
                caller_task.cancelled(),
                "cancel_task swallowed the caller's own cancellation: the "
                "caller completed normally after being cancelled",
            )
            self.assertFalse(caller_resumed_after_own_cancel)
        finally:
            release_child_cleanup.set()
            await asyncio.gather(child, caller_task, return_exceptions=True)

    async def test_child_cancellation_still_absorbed(self):
        """The normal case must keep working: awaiting the cancelled child's
        ``CancelledError`` is absorbed and ``cancel_task`` returns cleanly."""
        task_manager = self._create_task_manager()

        async def long_handler():
            await asyncio.sleep(10)

        task = task_manager.create_task(long_handler(), "long_handler")
        await asyncio.sleep(0)
        # Must not raise even though awaiting `task` raises CancelledError.
        await task_manager.cancel_task(task)
        self.assertTrue(task.cancelled())


if __name__ == "__main__":
    unittest.main()
