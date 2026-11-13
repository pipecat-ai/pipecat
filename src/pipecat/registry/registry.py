#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared registry for tracking known workers across runners."""

from collections import defaultdict
from collections.abc import Callable, Coroutine

from loguru import logger

from pipecat.registry.types import WorkerReadyData

WatchHandler = Callable[[WorkerReadyData], Coroutine]


class WorkerRegistry:
    """Tracks all known workers across local and remote runners.

    Owned by a runner and shared with its workers. Organizes workers into
    local (this runner) and remote (other runners) so they are easy to
    distinguish. Deduplication is built in: each worker name is
    registered at most once.

    Notifications use a targeted watch mechanism: call
    ``watch(worker_name, handler)`` to be notified when a specific worker
    registers.
    """

    def __init__(self, runner_name: str):
        """Initialize the WorkerRegistry.

        Args:
            runner_name: Name of the runner that owns this registry.
        """
        self._runner_name = runner_name
        self._local_workers: dict[str, WorkerReadyData] = {}
        self._remote_workers: dict[str, dict[str, WorkerReadyData]] = defaultdict(dict)
        self._watches: dict[str, list[WatchHandler]] = defaultdict(list)

    @property
    def runner_name(self) -> str:
        """The name of the runner that owns this registry."""
        return self._runner_name

    @property
    def local_workers(self) -> list[str]:
        """Names of workers registered under this runner."""
        return list(self._local_workers.keys())

    @property
    def remote_workers(self) -> list[str]:
        """Names of workers registered under remote runners."""
        result: list[str] = []
        for workers in self._remote_workers.values():
            result.extend(workers.keys())
        return result

    def get(self, worker_name: str) -> WorkerReadyData | None:
        """Look up a registered worker by name.

        Args:
            worker_name: The worker name to look up.

        Returns:
            The worker's ``WorkerReadyData``, or None if not found.
        """
        if worker_name in self._local_workers:
            return self._local_workers[worker_name]
        for workers in self._remote_workers.values():
            if worker_name in workers:
                return workers[worker_name]
        return None

    def __contains__(self, worker_name: str) -> bool:
        return self.get(worker_name) is not None

    async def watch(self, worker_name: str, handler: WatchHandler) -> None:
        """Watch for a specific worker's registration.

        Idempotent: registering the same ``(worker_name, handler)`` pair
        more than once is a no-op (otherwise the handler would fire
        multiple times when the worker registers — e.g. when a parent
        both calls ``add_workers(child)`` (which auto-watches) and
        declares a ``@worker_ready(name=child.name)`` handler that the
        framework also installs).

        If the worker is already registered, the handler fires immediately.

        Args:
            worker_name: The worker name to watch for.
            handler: Async callable invoked with the worker's data.
        """
        handlers = self._watches[worker_name]
        if handler in handlers:
            return
        handlers.append(handler)
        existing = self.get(worker_name)
        if existing:
            await handler(existing)

    async def register(self, worker_data: WorkerReadyData) -> bool:
        """Register a worker. Returns True if the worker was new.

        If the worker is already registered, this is a no-op and returns
        False. Otherwise the worker is added and watchers are notified.

        Args:
            worker_data: Information about the worker to register.

        Returns:
            True if the worker was newly registered, False if already known.
        """
        is_local = worker_data.runner == self._runner_name
        target = self._local_workers if is_local else self._remote_workers[worker_data.runner]

        if worker_data.worker_name in target:
            return False

        # Warn if the same name exists on a different runner
        existing = self.get(worker_data.worker_name)
        if existing and existing.runner != worker_data.runner:
            logger.warning(
                f"Worker '{worker_data.worker_name}' registered on both "
                f"'{existing.runner}' and '{worker_data.runner}'"
            )

        target[worker_data.worker_name] = worker_data
        locality = "local" if is_local else worker_data.runner
        logger.debug(f"Worker '{worker_data.worker_name}' ready ({locality})")
        await self._notify(worker_data)
        return True

    async def _notify(self, worker_data: WorkerReadyData) -> None:
        """Notify watchers of a new registration."""
        for handler in self._watches.get(worker_data.worker_name, []):
            await handler(worker_data)
