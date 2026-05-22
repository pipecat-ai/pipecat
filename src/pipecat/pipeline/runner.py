#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipeline runner for managing pipeline worker execution and orchestration.

This module provides the :class:`PipelineRunner` class. It runs
:class:`~pipecat.pipeline.worker.PipelineWorker` instances to completion and
also acts as the host for spawned :class:`~pipecat.pipeline.base_worker.BaseWorker`
instances — owning the shared :class:`~pipecat.bus.WorkerBus`,
the worker registry, and the worker manager that backs the entire session.

For a typical single-pipeline bot, use :meth:`PipelineRunner.run` with the
worker:

.. code-block:: python

    runner = PipelineRunner()
    await runner.run(worker)

``run()`` returns when ``worker`` finishes.

For multi-worker setups, add the additional workers alongside the main one:

.. code-block:: python

    runner = PipelineRunner()
    await runner.add_workers(CodeWorker("code_worker", ...))
    await runner.run(worker)

Optionally, ``add_workers`` every worker (including the main pipeline) and call
``run()`` with no argument. In that form ``run()`` blocks until
:meth:`PipelineRunner.end` / :meth:`PipelineRunner.cancel` is called (or an
incoming ``BusEndMessage`` / ``BusCancelMessage`` triggers the same path) —
added workers finishing on their own does **not** unblock it.
"""

import asyncio
import gc
import signal
import uuid
from dataclasses import dataclass, field

from loguru import logger

from pipecat.bus import (
    AsyncQueueBus,
    BusAddWorkerMessage,
    BusCancelMessage,
    BusCancelWorkerMessage,
    BusEndMessage,
    BusEndWorkerMessage,
    BusMessage,
    BusWorkerRegistryMessage,
    WorkerBus,
)
from pipecat.bus.subscriber import BusSubscriber
from pipecat.pipeline.base_worker import BaseWorker
from pipecat.pipeline.utils import run_setup_hook
from pipecat.pipeline.worker import PipelineWorker, PipelineWorkerParams
from pipecat.registry import WorkerRegistry
from pipecat.registry.types import WorkerReadyData, WorkerRegistryEntry
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams
from pipecat.utils.base_object import BaseObject


@dataclass
class _WorkerEntry:
    """A worker registered on the runner and its background asyncio worker."""

    worker: BaseWorker
    runner_task: asyncio.Task | None = field(default=None, repr=False)


class PipelineRunner(BaseObject, BusSubscriber):
    """Manages pipeline worker execution.

    Provides a high-level interface for running pipeline workers with
    automatic signal handling (SIGINT/SIGTERM), optional garbage
    collection, proper cleanup of resources, and a worker bus + registry
    for multi-worker orchestration.

    Two entry points:

    - :meth:`run(worker)` — block until the given pipeline worker finishes.
      The most common case for a single-pipeline bot.
    - :meth:`add_workers(*workers)` — register one or more workers on the
      runner's bus and start them in the background. Added workers run
      alongside the main worker and are cancelled when the main worker
      finishes (or when :meth:`end` / :meth:`cancel` is called).

    Event handlers available:

    - ``on_ready`` — fired after the runner has finished its
      initialization and any added workers have been started.
    - ``on_error`` — fired when starting an added worker fails.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        bus: WorkerBus | None = None,
        handle_sigint: bool = True,
        handle_sigterm: bool = False,
        force_gc: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        """Initialize the pipeline runner.

        Args:
            name: Optional name for the runner instance. Defaults to a
                UUID-based name. Must be unique across runners in a
                distributed setup.
            bus: Optional :class:`WorkerBus`. Defaults to a new
                in-process :class:`AsyncQueueBus`.
            handle_sigint: Whether to automatically handle SIGINT signals.
            handle_sigterm: Whether to automatically handle SIGTERM signals.
            force_gc: Whether to force garbage collection after the main
                worker completes.
            loop: Event loop to use. If None, uses the current running loop.
        """
        super().__init__(name=name or f"runner-{uuid.uuid4().hex[:8]}")

        self._bus: WorkerBus = bus or AsyncQueueBus()
        self._registry = WorkerRegistry(runner_name=self.name)

        self._entries: dict[str, _WorkerEntry] = {}
        self._known_runners: set[str] = set()
        self._running: bool = False
        self._shutdown_event = asyncio.Event()
        self._sig_task: asyncio.Task | None = None

        self._handle_sigint = handle_sigint
        self._handle_sigterm = handle_sigterm
        self._force_gc = force_gc
        self._loop = loop or asyncio.get_running_loop()

        self._register_event_handler("on_ready")
        self._register_event_handler("on_error")

    @property
    def bus(self) -> WorkerBus:
        """The bus this runner hosts; shared across launched workers."""
        return self._bus

    @property
    def registry(self) -> WorkerRegistry:
        """The worker registry this runner owns."""
        return self._registry

    async def add_workers(self, *workers: BaseWorker) -> None:
        """Add one or more workers to the runner.

        Adding a worker attaches it to the runner's bus and registry, and
        starts it in the background. If the runner is not yet running
        (``add_workers`` was called before :meth:`run`), workers are
        queued and started during run setup; if the runner is already
        running, each worker starts immediately.

        Added workers run alongside the main worker and are cancelled
        when the main worker finishes (or when :meth:`end` /
        :meth:`cancel` is called).

        Args:
            *workers: One or more workers to add.
        """
        for worker in workers:
            if worker.name in self._entries:
                logger.error(
                    f"PipelineRunner '{self}': worker '{worker.name}' already exists, skipping"
                )
                continue
            worker.attach(registry=self._registry, bus=self._bus)
            await self._registry.watch(worker.name, self._on_local_worker_ready)
            entry = _WorkerEntry(worker=worker)
            self._entries[worker.name] = entry
            logger.debug(f"PipelineRunner '{self}': added worker '{worker.name}'")

            if self._running:
                await self._start_worker(entry)

    async def run(self, worker: PipelineWorker | None = None) -> None:
        """Run a pipeline worker to completion (optionally alongside added workers).

        If ``worker`` is provided, blocks until that worker finishes. Any
        added workers are started in the background and cancelled
        when the main worker finishes.

        If ``worker`` is None, blocks until :meth:`end` or :meth:`cancel`
        is called (or until an incoming ``BusEndMessage`` /
        ``BusCancelMessage`` triggers the same path). Added workers
        finishing on their own does **not** unblock the runner — use
        this form for hosts that have no single "main" pipeline and
        want to stay up across many spawned sessions (e.g. a FastAPI
        server). If you want the runner to finish when a specific
        pipeline finishes, pass that pipeline as ``worker``.

        Args:
            worker: The pipeline worker to run, or None.
        """
        logger.debug(f"PipelineRunner '{self}': started running {worker}")
        self._shutdown_event.clear()

        # Treat the main worker as any other added worker: ``add_workers`` attaches
        # it to the bus and registry, and ``_setup_session`` then starts every
        # entry (main and pre-added) through the same code path.
        if worker is not None:
            await self.add_workers(worker)

        await self._setup_session()
        await self._call_event_handler("on_ready")

        # Wait for the main worker's background runner worker to finish
        # (or for an explicit shutdown when there's no main worker).
        try:
            if worker is not None:
                runner_task = self._entries[worker.name].runner_task
                if runner_task is not None:
                    await runner_task
            else:
                await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass

        # Cancel any remaining launched workers and wait for them to finish.
        await self._cancel_spawned_tasks()

        # Cleanup base object.
        await self.cleanup()

        # If we are cancelling through a signal, make sure we wait for it so
        # everything gets cleaned up nicely.
        if self._sig_task:
            await self._sig_task

        await self._bus.stop()
        self._running = False

        if self._force_gc:
            await self._gc_collect()

        logger.debug(f"PipelineRunner '{self}': finished running {worker}")

    async def stop_when_done(self) -> None:
        """Schedule all root pipeline workers to stop when their current processing is complete."""
        logger.debug(f"PipelineRunner '{self}': scheduled to stop when all workers are done")
        await asyncio.gather(
            *[
                entry.worker.stop_when_done()
                for entry in self._entries.values()
                if isinstance(entry.worker, PipelineWorker) and entry.worker.parent is None
            ]
        )

    async def end(self, reason: str | None = None) -> None:
        """Gracefully end all running workers.

        Idempotent; subsequent calls are ignored.

        Args:
            reason: Optional human-readable reason for ending.
        """
        if self._shutdown_event.is_set():
            return
        logger.debug(f"PipelineRunner '{self}': ending gracefully (reason={reason})")
        self._shutdown_event.set()
        for name, entry in self._entries.items():
            if entry.worker.parent is None:
                await self._bus.send(
                    BusEndWorkerMessage(source=self.name, target=name, reason=reason)
                )

    async def cancel(self, reason: str | None = None) -> None:
        """Immediately cancel all running workers.

        Idempotent; subsequent calls are ignored.

        Args:
            reason: Optional human-readable reason for cancelling.
        """
        if self._shutdown_event.is_set():
            return
        logger.debug(f"PipelineRunner '{self}': cancelling (reason={reason})")
        self._shutdown_event.set()
        for name, entry in self._entries.items():
            if entry.worker.parent is None:
                await self._bus.send(
                    BusCancelWorkerMessage(source=self.name, target=name, reason=reason)
                )

    async def on_bus_message(self, message: BusMessage) -> None:
        """Process incoming bus messages for runner-level concerns."""
        if message.source == self.name:
            return
        if isinstance(message, BusEndMessage):
            self.create_task(self.end(message.reason), "end")
        elif isinstance(message, BusCancelMessage):
            self.create_task(self.cancel(message.reason), "cancel")
        elif isinstance(message, BusAddWorkerMessage) and message.worker:
            await self.add_workers(message.worker)
        elif isinstance(message, BusWorkerRegistryMessage):
            await self._handle_worker_registry(message)

    async def _setup_session(self) -> None:
        """One-time per-run setup: worker manager, bus, signal handlers, launched workers."""
        if self._running:
            return
        task_manager = TaskManager()
        task_manager.setup(TaskManagerParams(loop=self._loop))
        await super().setup(task_manager)
        await self._bus.setup(task_manager)

        if self._handle_sigint:
            self._setup_sigint()
        if self._handle_sigterm:
            self._setup_sigterm()

        await self._bus.subscribe(self)
        await self._bus.start()

        await self._load_setup_files()

        for entry in self._entries.values():
            await self._start_worker(entry)

        self._running = True

    async def _cancel_spawned_tasks(self) -> None:
        """Wait for added workers' runner tasks to finish (or cancel them)."""
        remaining = [
            e.runner_task
            for e in self._entries.values()
            if e.runner_task and not e.runner_task.done()
        ]
        if not remaining:
            return
        for entry in self._entries.values():
            if entry.worker.parent is None:
                await self._bus.send(
                    BusCancelWorkerMessage(
                        source=self.name, target=entry.worker.name, reason="runner exiting"
                    )
                )
        await asyncio.gather(*remaining, return_exceptions=True)

    async def _load_setup_files(self) -> None:
        """Run ``setup_pipeline_runner`` from each file in ``PIPECAT_SETUP_FILES``.

        A setup file may define ``setup_pipeline_runner(runner)`` to add
        workers, attach event handlers, or wire other runner-level
        configuration.
        """
        await run_setup_hook(target=self, function_name="setup_pipeline_runner")

    async def _start_worker(self, entry: _WorkerEntry) -> None:
        """Run a registered worker as a background asyncio worker."""
        worker = entry.worker
        logger.debug(f"PipelineRunner '{self}': starting worker '{worker.name}'")

        entry.runner_task = self.create_task(
            self._run_worker(worker),
            f"task_{worker.name}",
        )
        # Add the worker to event loop right away without needing to `await`.
        await asyncio.sleep(0)

    async def _run_worker(self, worker: BaseWorker) -> None:
        """Drive a registered worker to completion."""
        try:
            params = PipelineWorkerParams(loop=self._loop)
            await worker.run(params)
        except asyncio.CancelledError:
            pass

    async def _on_local_worker_ready(self, data: WorkerReadyData) -> None:
        """Called when a local added worker registers as ready."""
        if data.runner != self.name:
            return
        entry = self._entries.get(data.worker_name)
        if not entry or entry.worker.parent is not None:
            return
        await self._send_registry()

    async def _send_registry(self) -> None:
        """Broadcast this runner's workers to the bus."""
        workers = [
            WorkerRegistryEntry(
                name=entry.worker.name,
                parent=entry.worker.parent,
                active=entry.worker.active,
                bridged=entry.worker.bridged,
                started_at=entry.worker.started_at,
            )
            for entry in self._entries.values()
        ]
        if workers:
            names = [w.name for w in workers]
            logger.debug(f"PipelineRunner '{self}': broadcasting registry: {names}")
            await self._bus.send(
                BusWorkerRegistryMessage(
                    source=self.name,
                    runner=self.name,
                    workers=workers,
                )
            )

    async def _handle_worker_registry(self, message: BusWorkerRegistryMessage) -> None:
        """Handle a registry message from a remote runner."""
        worker_names = [w.name for w in message.workers]
        logger.debug(
            f"PipelineRunner '{self}': received registry from '{message.runner}' "
            f"with workers: {worker_names}"
        )
        for entry in message.workers:
            await self._registry.register(
                WorkerReadyData(worker_name=entry.name, runner=message.runner)
            )
        if message.runner not in self._known_runners:
            self._known_runners.add(message.runner)
            logger.debug(
                f"PipelineRunner '{self}': new runner '{message.runner}', sending our registry back"
            )
            await self._send_registry()

    def _setup_sigint(self) -> None:
        """Set up SIGINT handler for graceful shutdown."""
        try:
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGINT, lambda *args: self._sig_handler())
        except NotImplementedError:
            # Windows fallback
            signal.signal(signal.SIGINT, lambda s, f: self._sig_handler())

    def _setup_sigterm(self) -> None:
        """Set up SIGTERM handler for graceful shutdown."""
        try:
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGTERM, lambda *args: self._sig_handler())
        except NotImplementedError:
            # Windows fallback
            signal.signal(signal.SIGTERM, lambda s, f: self._sig_handler())

    def _sig_handler(self) -> None:
        """Handle interrupt signals by cancelling the runner."""
        if not self._sig_task:
            self._sig_task = asyncio.create_task(self._sig_cancel())

    async def _sig_cancel(self) -> None:
        """Cancel the runner due to signal interruption."""
        logger.warning(f"PipelineRunner '{self}': interruption detected, cancelling.")
        await self.cancel(reason="interrupt signal")

    async def _gc_collect(self) -> None:
        """Force garbage collection and log results."""
        collected = await asyncio.to_thread(gc.collect)
        logger.debug(f"Garbage collector: collected {collected} objects.")
        logger.debug(f"Garbage collector: uncollectable objects {gc.garbage}")
