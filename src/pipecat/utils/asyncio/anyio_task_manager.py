#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AnyIO-based task management for trio and asyncio support.

This module provides :class:`AnyioTaskManager`, a drop-in alternative to
:class:`~pipecat.utils.asyncio.task_manager.TaskManager` that uses
`anyio <https://anyio.readthedocs.io/>`_ task groups under the hood so that
Pipecat pipelines can run on either ``asyncio`` or ``trio``.

Structured concurrency (the model anyio/trio use) requires every task to
live inside a task group's lexical scope. That's fundamentally different
from asyncio's "fire and forget" ``create_task``. To bridge the gap,
``AnyioTaskManager`` is an async context manager that owns a long-lived
task group; ``create_task`` spawns children into it and returns a
:class:`TaskHandle` that supports the subset of the ``asyncio.Task`` API
Pipecat depends on (name, cancel, done, await).

Usage::

    async with AnyioTaskManager() as tm:
        handle = tm.create_task(worker(), name="worker")
        ...
        await tm.cancel_task(handle)

.. note::
   This is experimental. Most services still import ``asyncio`` directly
   and will not run under trio until migrated to
   :mod:`pipecat.utils.asyncio.compat`.
"""

from __future__ import annotations

import traceback
from typing import Any, Coroutine, Dict, Optional, Sequence

import anyio
import anyio.abc
from loguru import logger

from pipecat.utils.asyncio.task_manager import BaseTaskManager, TaskManagerParams


class TaskHandle:
    """A cancellable handle to a task running in an anyio task group.

    Mimics the small slice of the :class:`asyncio.Task` API that Pipecat
    uses (``get_name``, ``cancel``, ``done``, ``result``, ``__await__``) so
    it can stand in for an ``asyncio.Task`` throughout the codebase while
    still working under trio.
    """

    def __init__(self, name: str) -> None:
        """Initialize the handle.

        Args:
            name: Human-readable task name for logging.
        """
        self._name = name
        self._cancel_scope: Optional[anyio.CancelScope] = None
        self._cancel_requested = False
        self._done = anyio.Event()
        self._result: Any = None
        self._exception: Optional[BaseException] = None

    def get_name(self) -> str:
        """Return the task name."""
        return self._name

    def set_name(self, name: str) -> None:
        """Set the task name."""
        self._name = name

    def cancel(self) -> None:
        """Request cancellation of the task.

        If the task hasn't started yet the request is recorded and applied
        as soon as the cancel scope exists.
        """
        self._cancel_requested = True
        if self._cancel_scope is not None:
            self._cancel_scope.cancel()

    def cancelled(self) -> bool:
        """Return ``True`` if cancellation was requested."""
        return self._cancel_requested

    def done(self) -> bool:
        """Return ``True`` if the task has finished (successfully or not)."""
        return self._done.is_set()

    def result(self) -> Any:
        """Return the task's result, raising if it failed or isn't done."""
        if not self.done():
            raise RuntimeError(f"Task {self._name!r} is not done")
        if self._exception is not None:
            raise self._exception
        return self._result

    def exception(self) -> Optional[BaseException]:
        """Return the exception raised by the task, or ``None``."""
        if not self.done():
            raise RuntimeError(f"Task {self._name!r} is not done")
        return self._exception

    async def wait(self) -> Any:
        """Wait for the task to finish and return its result."""
        await self._done.wait()
        return self.result()

    def __await__(self):
        """Allow ``await handle`` like ``await asyncio.Task``."""
        return self.wait().__await__()


class AnyioTaskManager(BaseTaskManager):
    """AnyIO-backed task manager supporting both asyncio and trio.

    Must be used as an async context manager so the internal task group
    has a well-defined lifetime (structured concurrency requires this)::

        async with AnyioTaskManager() as tm:
            task = tm.create_task(coro(), name="my-task")
            ...
        # all child tasks are cancelled/joined on exit
    """

    def __init__(self) -> None:
        """Initialize the task manager with no active task group."""
        self._task_group: Optional[anyio.abc.TaskGroup] = None
        self._tasks: Dict[str, TaskHandle] = {}
        self._params: Optional[TaskManagerParams] = None

    async def __aenter__(self) -> "AnyioTaskManager":
        """Enter the context, starting the internal task group."""
        self._task_group = anyio.create_task_group()
        await self._task_group.__aenter__()
        return self

    async def __aexit__(self, *exc_info) -> Optional[bool]:
        """Exit the context, cancelling and joining all child tasks."""
        if self._task_group is not None:
            tg, self._task_group = self._task_group, None
            tg.cancel_scope.cancel()
            return await tg.__aexit__(*exc_info)
        return None

    def setup(self, params: TaskManagerParams) -> None:
        """Store configuration parameters.

        Args:
            params: Configuration (the ``loop`` field is unused under anyio).
        """
        if not self._params:
            self._params = params

    def get_event_loop(self):
        """Return the asyncio event loop if running on asyncio.

        Raises:
            RuntimeError: If running under trio (no event loop concept).
        """
        import asyncio

        from pipecat.utils.asyncio.compat import current_backend

        if current_backend() != "asyncio":
            raise RuntimeError(
                "get_event_loop() is not available under trio; "
                "use anyio primitives from pipecat.utils.asyncio.compat instead"
            )
        return asyncio.get_running_loop()

    def create_task(self, coroutine: Coroutine, name: str) -> TaskHandle:  # type: ignore[override]
        """Spawn a child task in the managed task group.

        Args:
            coroutine: The coroutine to run.
            name: Name for logging and lookup.

        Returns:
            A :class:`TaskHandle` that can be awaited or cancelled.

        Raises:
            RuntimeError: If the task manager context hasn't been entered.
        """
        if self._task_group is None:
            raise RuntimeError(
                "AnyioTaskManager must be used as an async context manager "
                "before create_task() can be called"
            )

        handle = TaskHandle(name)

        async def run() -> None:
            cancelled_exc = anyio.get_cancelled_exc_class()
            try:
                with anyio.CancelScope() as scope:
                    handle._cancel_scope = scope
                    if handle._cancel_requested:
                        scope.cancel()
                    try:
                        handle._result = await coroutine
                    except cancelled_exc:
                        logger.trace(f"{name}: task cancelled")
                        raise
                    except Exception as e:
                        handle._exception = e
                        tb = traceback.extract_tb(e.__traceback__)
                        last = tb[-1]
                        logger.error(
                            f"{name} unexpected exception ({last.filename}:{last.lineno}): {e}"
                        )
            finally:
                handle._done.set()
                self._tasks.pop(name, None)

        self._tasks[name] = handle
        self._task_group.start_soon(run, name=name)
        logger.trace(f"{name}: task created")
        return handle

    async def cancel_task(  # type: ignore[override]
        self, task: TaskHandle, timeout: Optional[float] = None
    ) -> None:
        """Cancel a task and wait for it to finish.

        Args:
            task: The handle returned by :meth:`create_task`.
            timeout: Optional seconds to wait before giving up.
        """
        name = task.get_name()
        task.cancel()
        try:
            if timeout:
                with anyio.fail_after(timeout):
                    await task._done.wait()
            else:
                await task._done.wait()
        except TimeoutError:
            logger.warning(f"{name}: timed out waiting for task to cancel")
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            last = tb[-1]
            logger.error(
                f"{name} unexpected exception while cancelling task "
                f"({last.filename}:{last.lineno}): {e}"
            )

    def current_tasks(self) -> Sequence[TaskHandle]:  # type: ignore[override]
        """Return all live task handles."""
        return list(self._tasks.values())
