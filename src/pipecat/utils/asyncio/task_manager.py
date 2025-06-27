#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Coroutine, Dict, Optional, Sequence

from loguru import logger

WATCHDOG_TIMEOUT = 5.0


@dataclass
class TaskManagerParams:
    loop: asyncio.AbstractEventLoop
    enable_watchdog_timers: bool = False
    enable_watchdog_logging: bool = False
    watchdog_timeout: float = WATCHDOG_TIMEOUT


class BaseTaskManager(ABC):
    @abstractmethod
    def setup(self, params: TaskManagerParams):
        pass

    @abstractmethod
    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        pass

    @abstractmethod
    def create_task(
        self,
        coroutine: Coroutine,
        name: str,
        *,
        enable_watchdog_logging: Optional[bool] = None,
        enable_watchdog_timers: Optional[bool] = None,
        watchdog_timeout: Optional[float] = None,
    ) -> asyncio.Task:
        """
        Creates and schedules a new asyncio Task that runs the given coroutine.

        The task is added to a global set of created tasks.

        Args:
            loop (asyncio.AbstractEventLoop): The event loop to use for creating the task.
            coroutine (Coroutine): The coroutine to be executed within the task.
            name (str): The name to assign to the task for identification.
            enable_watchdog_logging(bool): whether this task should log watchdog processing times.
            enable_watchdog_timers(bool): whether this task should have a watchdog timer.
            watchdog_timeout(float): watchdog timer timeout for this task.

        Returns:
            asyncio.Task: The created task object.
        """
        pass

    @abstractmethod
    async def wait_for_task(self, task: asyncio.Task, timeout: Optional[float] = None):
        """Wait for an asyncio.Task to complete with optional timeout handling.

        This function awaits the specified asyncio.Task and handles scenarios for
        timeouts, cancellations, and other exceptions. It also ensures that the task
        is removed from the set of registered tasks upon completion or failure.

        Args:
            task (asyncio.Task): The asyncio Task to wait for.
            timeout (Optional[float], optional): The maximum number of seconds
                to wait for the task to complete. If None, waits indefinitely.
                Defaults to None.
        """
        pass

    @abstractmethod
    async def cancel_task(self, task: asyncio.Task, timeout: Optional[float] = None):
        """Cancels the given asyncio Task and awaits its completion with an
        optional timeout.

        This function removes the task from the set of registered tasks upon
        completion or failure.

        Args:
            task (asyncio.Task): The task to be cancelled.
            timeout (Optional[float]): The optional timeout in seconds to wait for the task to cancel.

        """
        pass

    @abstractmethod
    def current_tasks(self) -> Sequence[asyncio.Task]:
        """Returns the list of currently created/registered tasks."""
        pass

    @abstractmethod
    def task_reset_watchdog(self):
        """Resets the running task watchdog timer. If not reset, a warning will
        be logged indicating the task is stalling.

        """
        pass

    @property
    @abstractmethod
    def task_watchdog_enabled(self) -> bool:
        """Whether the current running task has a watchdog timer enabled."""
        pass


@dataclass
class TaskData:
    task: asyncio.Task
    watchdog_timer: asyncio.Event
    enable_watchdog_logging: bool
    enable_watchdog_timers: bool
    watchdog_timeout: float
    watchdog_task: Optional[asyncio.Task]


class TaskManager(BaseTaskManager):
    def __init__(self) -> None:
        self._tasks: Dict[str, TaskData] = {}
        self._params: Optional[TaskManagerParams] = None

    def setup(self, params: TaskManagerParams):
        if not self._params:
            self._params = params

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        if not self._params:
            raise Exception("TaskManager is not setup: unable to get event loop")
        return self._params.loop

    def create_task(
        self,
        coroutine: Coroutine,
        name: str,
        *,
        enable_watchdog_logging: Optional[bool] = None,
        enable_watchdog_timers: Optional[bool] = None,
        watchdog_timeout: Optional[float] = None,
    ) -> asyncio.Task:
        """
        Creates and schedules a new asyncio Task that runs the given coroutine.

        The task is added to a global set of created tasks.

        Args:
            loop (asyncio.AbstractEventLoop): The event loop to use for creating the task.
            coroutine (Coroutine): The coroutine to be executed within the task.
            name (str): The name to assign to the task for identification.
            enable_watchdog_logging(bool): whether this task should log watchdog processing time.
            enable_watchdog_timers(bool): whether this task should have a watchdog timer.
            watchdog_timeout(float): watchdog timer timeout for this task.

        Returns:
            asyncio.Task: The created task object.
        """

        async def run_coroutine():
            try:
                await coroutine
            except asyncio.CancelledError:
                logger.trace(f"{name}: task cancelled")
                # Re-raise the exception to ensure the task is cancelled.
                raise
            except Exception as e:
                logger.exception(f"{name}: unexpected exception: {e}")

        if not self._params:
            raise Exception("TaskManager is not setup: unable to get event loop")

        task = self._params.loop.create_task(run_coroutine())
        task.set_name(name)
        task.add_done_callback(self._task_done_handler)
        self._add_task(
            TaskData(
                task=task,
                watchdog_timer=asyncio.Event(),
                enable_watchdog_logging=(
                    enable_watchdog_logging
                    if enable_watchdog_logging
                    else self._params.enable_watchdog_logging
                ),
                enable_watchdog_timers=(
                    enable_watchdog_timers
                    if enable_watchdog_timers
                    else self._params.enable_watchdog_timers
                ),
                watchdog_timeout=(
                    watchdog_timeout if watchdog_timeout else self._params.watchdog_timeout
                ),
                watchdog_task=None,
            ),
        )
        logger.trace(f"{name}: task created")
        return task

    async def wait_for_task(self, task: asyncio.Task, timeout: Optional[float] = None):
        """Wait for an asyncio.Task to complete with optional timeout handling.

        This function awaits the specified asyncio.Task and handles scenarios for
        timeouts, cancellations, and other exceptions. It also ensures that the task
        is removed from the set of registered tasks upon completion or failure.

        Args:
            task (asyncio.Task): The asyncio Task to wait for.
            timeout (Optional[float], optional): The maximum number of seconds
                to wait for the task to complete. If None, waits indefinitely.
                Defaults to None.
        """
        name = task.get_name()
        try:
            if timeout:
                await asyncio.wait_for(task, timeout=timeout)
            else:
                await task
        except asyncio.TimeoutError:
            logger.warning(f"{name}: timed out waiting for task to finish")
        except asyncio.CancelledError:
            logger.trace(f"{name}: unexpected task cancellation (maybe Ctrl-C?)")
            raise
        except Exception as e:
            logger.exception(f"{name}: unexpected exception while stopping task: {e}")

    async def cancel_task(self, task: asyncio.Task, timeout: Optional[float] = None):
        """Cancels the given asyncio Task and awaits its completion with an
        optional timeout.

        This function removes the task from the set of registered tasks upon
        completion or failure.

        Args:
            task (asyncio.Task): The task to be cancelled.
            timeout (Optional[float]): The optional timeout in seconds to wait for the task to cancel.

        """
        name = task.get_name()
        task.cancel()
        try:
            # Make sure to reset watchdog if a task is cancelled.
            self.reset_watchdog(task)
            if timeout:
                await asyncio.wait_for(task, timeout=timeout)
            else:
                await task
        except asyncio.TimeoutError:
            logger.warning(f"{name}: timed out waiting for task to cancel")
        except asyncio.CancelledError:
            # Here are sure the task is cancelled properly.
            pass
        except Exception as e:
            logger.exception(f"{name}: unexpected exception while cancelling task: {e}")
        except BaseException as e:
            logger.critical(f"{name}: fatal base exception while cancelling task: {e}")
            raise

    def reset_watchdog(self, task: asyncio.Task):
        name = task.get_name()
        if name in self._tasks and self._tasks[name].enable_watchdog_timers:
            self._tasks[name].watchdog_timer.set()

    def current_tasks(self) -> Sequence[asyncio.Task]:
        """Returns the list of currently created/registered tasks."""
        return [data.task for data in self._tasks.values()]

    def task_reset_watchdog(self):
        """Resets the running task watchdog timer. If not reset on time, a warning
        will be logged indicating the task is stalling.

        """
        task = asyncio.current_task()
        if task:
            self.reset_watchdog(task)

    @property
    def task_watchdog_enabled(self) -> bool:
        task = asyncio.current_task()
        if not task:
            return False
        name = task.get_name()
        return name in self._tasks and self._tasks[name].enable_watchdog_timers

    def _add_task(self, task_data: TaskData):
        name = task_data.task.get_name()
        self._tasks[name] = task_data
        if self._params and task_data.enable_watchdog_timers:
            watchdog_task = self.get_event_loop().create_task(
                self._watchdog_task_handler(task_data)
            )
            task_data.watchdog_task = watchdog_task

    async def _watchdog_task_handler(self, task_data: TaskData):
        name = task_data.task.get_name()
        timer = task_data.watchdog_timer
        enable_watchdog_logging = task_data.enable_watchdog_logging
        watchdog_timeout = task_data.watchdog_timeout

        while True:
            try:
                start_time = time.time()
                await asyncio.wait_for(timer.wait(), timeout=watchdog_timeout)
                total_time = time.time() - start_time
                if enable_watchdog_logging:
                    logger.debug(f"{name} time between watchdog timer resets: {total_time:.20f}")
            except asyncio.TimeoutError:
                logger.warning(
                    f"{name}: task is taking too long {WATCHDOG_TIMEOUT} second(s) (forgot to reset watchdog?)"
                )
            finally:
                timer.clear()

    def _task_done_handler(self, task: asyncio.Task):
        name = task.get_name()
        try:
            task_data = self._tasks[name]
            if task_data.watchdog_task:
                task_data.watchdog_task.cancel()
                task_data.watchdog_task = None
            del self._tasks[name]
        except KeyError as e:
            logger.trace(f"{name}: unable to remove task data (already removed?): {e}")
