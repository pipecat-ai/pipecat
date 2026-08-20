#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the asyncio TaskManager."""

import asyncio
import inspect
import unittest

from pipecat.utils.asyncio.task_manager import TaskManager


class TestTaskManagerCreateTask(unittest.IsolatedAsyncioTestCase):
    """Tests for TaskManager.create_task() cancellation handling."""

    def _create_task_manager(self) -> TaskManager:
        task_manager = TaskManager()
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


class TestTaskManagerRegistry(unittest.IsolatedAsyncioTestCase):
    """Tests for how TaskManager tracks concurrently-running tasks."""

    async def test_same_name_tasks_tracked_independently(self):
        """Concurrent tasks that share a name are each tracked separately.

        Task names are not unique: :meth:`BaseObject.create_task` derives the
        name from the coroutine's ``co_name`` when none is given, so tasks
        started from the same method on the same object — the parallel
        function-call tasks, for example — all share a single name.
        """
        task_manager = TaskManager()

        both_running = asyncio.Event()
        release = asyncio.Event()
        running = 0

        async def handler():
            nonlocal running
            running += 1
            if running == 2:
                both_running.set()
            await release.wait()

        task1 = task_manager.create_task(handler(), "svc::_run_function_call")
        task2 = task_manager.create_task(handler(), "svc::_run_function_call")

        await both_running.wait()
        current = task_manager.current_tasks()
        self.assertEqual(len(current), 2)
        self.assertIn(task1, current)
        self.assertIn(task2, current)

        release.set()
        await asyncio.gather(task1, task2)
        self.assertEqual(len(task_manager.current_tasks()), 0)


class TestTaskManagerCancelTask(unittest.IsolatedAsyncioTestCase):
    """Tests for TaskManager.cancel_task() cancellation handling."""

    def _create_task_manager(self) -> TaskManager:
        return TaskManager(loop=asyncio.get_running_loop())

    async def test_caller_cancellation_propagates(self):
        """``cancel_task()`` must not swallow the caller's own cancellation.

        A caller cancelled while suspended in ``cancel_task`` has to die.
        Services tear down by awaiting ``cancel_task()`` from a ``finally``
        block (e.g. ``DeepgramSTTService._connection_handler`` cancelling its
        keepalive task); a reconnect loop that outlived its own cancellation
        would reconnect unsupervised and run forever.

        No assertion is made about the child: cancelling a task suspended at
        ``await child`` cancels ``child`` as well, so its state says nothing
        about the caller's.
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
        """The child's own ``CancelledError`` is absorbed, not propagated."""
        task_manager = self._create_task_manager()

        async def long_handler():
            await asyncio.sleep(10)

        task = task_manager.create_task(long_handler(), "long_handler")
        await asyncio.sleep(0)
        # Must not raise even though awaiting `task` raises CancelledError.
        await task_manager.cancel_task(task)
        self.assertTrue(task.cancelled())

    async def test_already_cancelled_caller_finishes_cleanup(self):
        """An already-cancelled caller still completes the rest of its cleanup.

        A task that has been cancelled carries a non-zero ``cancelling()``
        count for the rest of its life, including throughout the ``finally``
        block where it tears down its children. Cancelling a child there is
        not a fresh cancellation of the caller, so ``cancel_task`` must return
        normally and let the remaining cleanup — closing a websocket, say —
        run to completion.
        """
        task_manager = self._create_task_manager()
        steps = []

        async def child():
            await asyncio.Event().wait()

        async def caller():
            task = task_manager.create_task(child(), "child")
            await asyncio.sleep(0)
            try:
                await asyncio.Event().wait()
            finally:
                await task_manager.cancel_task(task)
                steps.append("cancel_task returned")
                await asyncio.sleep(0)
                steps.append("cleanup finished")

        caller_task = asyncio.get_running_loop().create_task(caller())
        await asyncio.sleep(0.05)
        caller_task.cancel()
        await asyncio.gather(caller_task, return_exceptions=True)

        self.assertEqual(steps, ["cancel_task returned", "cleanup finished"])
        self.assertTrue(caller_task.cancelled())

    async def test_cancelling_the_running_task_is_ignored(self):
        """A task that asks to cancel itself carries on instead of dying.

        Awaiting your own task never completes, and the self-cancel raises the
        caller's own ``cancelling()`` count, so propagating it would kill the
        caller at that line and abandon whatever it still had to do.
        """
        task_manager = self._create_task_manager()
        steps = []

        async def handler():
            steps.append("before")
            task = asyncio.current_task()
            assert task is not None
            await task_manager.cancel_task(task)
            steps.append("after")

        task = task_manager.create_task(handler(), "self_canceller")
        await task

        self.assertEqual(steps, ["before", "after"])
        self.assertFalse(task.cancelled())


if __name__ == "__main__":
    unittest.main()
