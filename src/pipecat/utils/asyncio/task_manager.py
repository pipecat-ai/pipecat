#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Asyncio task management.

This module provides task management functionality. Includes both abstract base
classes and concrete implementations for managing asyncio tasks with
comprehensive monitoring and cleanup capabilities.
"""

import asyncio
import inspect
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Coroutine, Sequence
from contextvars import Context
from dataclasses import dataclass
from typing import Any

from loguru import logger

from pipecat.utils.deprecation import deprecated

# How many innermost frames to report when a task refuses to cancel. Enough to
# see past the awaiting wrappers to the library call that is actually blocking.
_CANCEL_STACK_FRAME_LIMIT = 8


def _describe_blocked_frames(task: asyncio.Task) -> str:
    """Summarize where a task is suspended, for cancel-timeout diagnostics.

    Walks the coroutine's ``cr_await`` chain rather than using
    ``Task.get_stack()``: a suspended coroutine has no ``f_back``, so
    ``get_stack()`` returns only the outermost frame — which for a managed task
    is always this module's own wrapper and says nothing about the blockage.
    The await chain reaches the library call that actually swallowed the
    cancellation.

    Args:
        task: The task that did not finish cancelling in time.

    Returns:
        A compact ``file:line in function`` chain, outermost first, or a short
        placeholder when no stack is available (e.g. the task did finish after
        all, so the timeout was a race).
    """
    try:
        awaitable: Any = task.get_coro()
        chain: list[str] = []
        while awaitable is not None and len(chain) < _CANCEL_STACK_FRAME_LIMIT:
            frame = getattr(awaitable, "cr_frame", None) or getattr(awaitable, "gi_frame", None)
            if frame is None:
                break
            chain.append(f"{frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}")
            awaitable = (
                getattr(awaitable, "cr_await", None)
                or getattr(awaitable, "ag_await", None)
                or getattr(awaitable, "gi_yieldfrom", None)
            )
        if not chain:
            return "<no frame>" if task.done() else "<frame unavailable>"
        return " -> ".join(chain)
    except Exception as e:  # pragma: no cover - diagnostics must never raise
        return f"<stack unavailable: {e}>"


@deprecated(
    "`TaskManagerParams` is deprecated since 1.5.0 and will be removed in 2.0.0. "
    "Use `TaskManager` instead."
)
@dataclass
class TaskManagerParams:
    """Configuration parameters for task manager initialization.

    .. deprecated:: 1.5.0
        Use :class:`TaskManager` (pass ``loop`` to its constructor) instead.
        Will be removed in 2.0.0.

    Parameters:
        loop: The asyncio event loop to use for task management.
    """

    loop: asyncio.AbstractEventLoop


class BaseTaskManager(ABC):
    """Abstract base class for asyncio task management.

    Provides the interface for creating, monitoring, and managing asyncio tasks.
    """

    @abstractmethod
    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop used by this task manager.

        Returns:
            The asyncio event loop instance.
        """
        pass

    @abstractmethod
    def create_task(
        self,
        coroutine: Coroutine,
        name: str,
        context: Context | None = None,
    ) -> asyncio.Task:
        """Creates and schedules a new asyncio Task that runs the given coroutine.

        The task is added to a global set of created tasks.

        Args:
            coroutine: The coroutine to be executed within the task.
            name: The name to assign to the task for identification.
            context: Optional context manager to use when creating the task.

        Returns:
            The created task object.
        """
        pass

    @abstractmethod
    async def cancel_task(self, task: asyncio.Task, timeout: float | None = None):
        """Cancels the given asyncio Task and awaits its completion with an optional timeout.

        This function removes the task from the set of registered tasks upon
        completion or failure.

        Args:
            task: The task to be cancelled.
            timeout: The optional timeout in seconds to wait for the task to cancel.
        """
        pass

    @abstractmethod
    def current_tasks(self) -> Sequence[asyncio.Task]:
        """Returns the list of currently created/registered tasks.

        Returns:
            Sequence of currently managed asyncio tasks.
        """
        pass


@dataclass
class TaskData:
    """Internal data structure for tracking task metadata.

    Parameters:
        task: The asyncio Task being managed.
    """

    task: asyncio.Task


class TaskManager(BaseTaskManager):
    """Concrete implementation of BaseTaskManager.

    Manages asyncio tasks. Provides comprehensive task lifecycle management
    including creation, monitoring, cancellation, and cleanup.

    """

    def __init__(
        self,
        *,
        context: Context | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """Initialize the task manager with empty task registry.

        Args:
            context: Optional context manager to use when creating tasks.
            loop: Event loop to use. If None, uses the current running loop.
        """
        self._context = context
        self._loop = loop or asyncio.get_running_loop()
        self._tasks: dict[str, TaskData] = {}

    @deprecated(
        "`TaskManager.setup` is deprecated since 1.5.0 and will be removed in 2.0.0. "
        "Use `TaskManager` instead."
    )
    def setup(self, params: TaskManagerParams):
        """Initialize the task manager with configuration parameters.

        .. deprecated:: 1.5.0
            Use the :class:`TaskManager` constructor (``loop`` / ``context``)
            instead. Will be removed in 2.0.0.

        Args:
            params: Configuration parameters for task management.
        """
        pass

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop used by this task manager.

        Returns:
            The asyncio event loop instance.
        """
        return self._loop

    def create_task(
        self,
        coroutine: Coroutine,
        name: str,
        context: Context | None = None,
    ) -> asyncio.Task:
        """Creates and schedules a new asyncio Task that runs the given coroutine.

        The task is added to a global set of created tasks.

        Args:
            coroutine: The coroutine to be executed within the task.
            name: The name to assign to the task for identification.
            context: Optional context manager to use when creating the task.

        Returns:
            The created task object.

        Raises:
            Exception: If the task manager is not properly set up.
        """

        async def run_coroutine():
            try:
                return await coroutine
            except asyncio.CancelledError:
                logger.trace(f"{name}: task cancelled")
                # Re-raise the exception to ensure the task is cancelled.
                raise
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                last = tb[-1]
                logger.error(f"{name} unexpected exception ({last.filename}:{last.lineno}): {e}")

        task = self._loop.create_task(run_coroutine(), context=context or self._context)
        task.set_name(name)

        def close_unawaited_coroutine(_: asyncio.Task):
            # If the task is cancelled before run_coroutine() ever runs, the
            # wrapper never reaches `await coroutine`, leaving the inner
            # coroutine un-awaited and emitting a spurious "coroutine was never
            # awaited" RuntimeWarning. Close it explicitly in that case. The
            # iscoroutine() guard keeps getcoroutinestate() from raising on
            # non-native awaitables that the type contract technically permits.
            if inspect.iscoroutine(coroutine) and (
                inspect.getcoroutinestate(coroutine) == inspect.CORO_CREATED
            ):
                coroutine.close()

        task.add_done_callback(close_unawaited_coroutine)
        task.add_done_callback(self._task_done_handler)
        self._add_task(TaskData(task=task))
        logger.trace(f"{name}: task created")
        return task

    async def cancel_task(self, task: asyncio.Task, timeout: float | None = None):
        """Cancels the given asyncio Task and awaits its completion.

        This function removes the task from the set of registered tasks upon
        completion or failure.

        Note:
            ``timeout`` is a REPORTING threshold, not a bound. Cancelling an
            asyncio task and then awaiting it always waits for the cancellation
            to actually complete — ``asyncio.wait_for`` is no exception, it
            cancels the inner task and then waits for it before raising
            ``TimeoutError``. So a task that is slow to cancel still blocks its
            canceller for as long as it takes; passing a timeout only means a
            warning is emitted, carrying the stack the task was blocked in at the
            moment the threshold was crossed. Turning this into a real bound
            means letting a still-running task be orphaned, which needs a
            generation guard on the frames it may still push — deliberately not
            done here.

        Args:
            task: The task to be cancelled.
            timeout: Optional threshold in seconds. Exceeding it logs a warning
                naming where the task is blocked; it does not abandon the wait.
        """
        name = task.get_name()
        task.cancel()
        started = time.monotonic()
        slow_to_cancel = False
        try:
            if timeout:
                done, _ = await asyncio.wait({task}, timeout=timeout)
                if not done:
                    slow_to_cancel = True
                    # Snapshot while the task is genuinely still pending: once it
                    # finishes there is no stack left to report, which is why
                    # reading the stack after a wait_for() TimeoutError always
                    # came back empty.
                    logger.warning(
                        f"{name}: timed out waiting for task to cancel after "
                        f"{timeout}s (still waiting); blocked at: "
                        f"{_describe_blocked_frames(task)}"
                    )
            await task
        except asyncio.CancelledError:
            # Here are sure the task is cancelled properly.
            pass
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            last = tb[-1]
            logger.error(
                f"{name} unexpected exception while cancelling task ({last.filename}:{last.lineno}): {e}"
            )
        except BaseException as e:
            tb = traceback.extract_tb(e.__traceback__)
            last = tb[-1]
            logger.critical(
                f"{name} fatal base exception while cancelling task ({last.filename}:{last.lineno}): {e}"
            )
            raise
        finally:
            if slow_to_cancel:
                # The real cost: this is how long the canceller was blocked. For
                # an interruption that is how long barge-in took to propagate.
                logger.warning(f"{name}: cancel completed after {time.monotonic() - started:.1f}s")

    def current_tasks(self) -> Sequence[asyncio.Task]:
        """Returns the list of currently created/registered tasks.

        Returns:
            Sequence of currently managed asyncio tasks.
        """
        return [data.task for data in self._tasks.values()]

    def _add_task(self, task_data: TaskData):
        """Add a task to the internal registry.

        Args:
            task_data: The task metadata.
        """
        name = task_data.task.get_name()
        self._tasks[name] = task_data

    def _task_done_handler(self, task: asyncio.Task):
        """Handle task completion by removing the task from the registry.

        Args:
            task: The completed asyncio task.
        """
        name = task.get_name()
        try:
            del self._tasks[name]
        except KeyError as e:
            logger.trace(f"{name}: unable to remove task data (already removed?): {e}")
