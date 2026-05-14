#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared registry for tracking known tasks across runners."""

from collections import defaultdict
from collections.abc import Callable, Coroutine

from loguru import logger

from pipecat.registry.types import TaskReadyData

WatchHandler = Callable[[TaskReadyData], Coroutine]


class TaskRegistry:
    """Tracks all known tasks across local and remote runners.

    Owned by a runner and shared with its tasks. Organizes tasks into
    local (this runner) and remote (other runners) so they are easy to
    distinguish. Deduplication is built in: each task name is
    registered at most once.

    Notifications use a targeted watch mechanism: call
    ``watch(task_name, handler)`` to be notified when a specific task
    registers.
    """

    def __init__(self, runner_name: str):
        """Initialize the TaskRegistry.

        Args:
            runner_name: Name of the runner that owns this registry.
        """
        self._runner_name = runner_name
        self._local_tasks: dict[str, TaskReadyData] = {}
        self._remote_tasks: dict[str, dict[str, TaskReadyData]] = defaultdict(dict)
        self._watches: dict[str, list[WatchHandler]] = defaultdict(list)

    @property
    def runner_name(self) -> str:
        """The name of the runner that owns this registry."""
        return self._runner_name

    @property
    def local_tasks(self) -> list[str]:
        """Names of tasks registered under this runner."""
        return list(self._local_tasks.keys())

    @property
    def remote_tasks(self) -> list[str]:
        """Names of tasks registered under remote runners."""
        result: list[str] = []
        for tasks in self._remote_tasks.values():
            result.extend(tasks.keys())
        return result

    def get(self, task_name: str) -> TaskReadyData | None:
        """Look up a registered task by name.

        Args:
            task_name: The task name to look up.

        Returns:
            The task's ``TaskReadyData``, or None if not found.
        """
        if task_name in self._local_tasks:
            return self._local_tasks[task_name]
        for tasks in self._remote_tasks.values():
            if task_name in tasks:
                return tasks[task_name]
        return None

    def __contains__(self, task_name: str) -> bool:
        return self.get(task_name) is not None

    async def watch(self, task_name: str, handler: WatchHandler) -> None:
        """Watch for a specific task's registration.

        If the task is already registered, the handler fires immediately.

        Args:
            task_name: The task name to watch for.
            handler: Async callable invoked with the task's data.
        """
        self._watches[task_name].append(handler)
        existing = self.get(task_name)
        if existing:
            await handler(existing)

    async def register(self, task_data: TaskReadyData) -> bool:
        """Register a task. Returns True if the task was new.

        If the task is already registered, this is a no-op and returns
        False. Otherwise the task is added and watchers are notified.

        Args:
            task_data: Information about the task to register.

        Returns:
            True if the task was newly registered, False if already known.
        """
        is_local = task_data.runner == self._runner_name
        target = self._local_tasks if is_local else self._remote_tasks[task_data.runner]

        if task_data.task_name in target:
            return False

        # Warn if the same name exists on a different runner
        existing = self.get(task_data.task_name)
        if existing and existing.runner != task_data.runner:
            logger.warning(
                f"Task '{task_data.task_name}' registered on both "
                f"'{existing.runner}' and '{task_data.runner}'"
            )

        target[task_data.task_name] = task_data
        locality = "local" if is_local else task_data.runner
        logger.debug(f"Task '{task_data.task_name}' ready ({locality})")
        await self._notify(task_data)
        return True

    async def _notify(self, task_data: TaskReadyData) -> None:
        """Notify watchers of a new registration."""
        for handler in self._watches.get(task_data.task_name, []):
            await handler(task_data)
