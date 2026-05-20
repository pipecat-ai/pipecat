#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipeline runner for managing pipeline task execution and orchestration.

This module provides the :class:`PipelineRunner` class. It runs
:class:`~pipecat.pipeline.task.PipelineTask` instances to completion and
also acts as the host for spawned :class:`~pipecat.pipeline.base_task.BaseTask`
instances — owning the shared :class:`~pipecat.bus.TaskBus`,
the task registry, and the task manager that backs the entire session.

For a typical single-pipeline bot, use :meth:`PipelineRunner.run` with the
task:

.. code-block:: python

    runner = PipelineRunner()
    await runner.run(task)

``run()`` returns when ``task`` finishes.

For multi-task setups, spawn the additional tasks alongside the main one:

.. code-block:: python

    runner = PipelineRunner()
    await runner.spawn(CodeWorker("code_worker", ...))
    await runner.run(task)

Optionally, ``spawn`` every task (including the main pipeline) and call
``run()`` with no argument. In that form ``run()`` blocks until
:meth:`PipelineRunner.end` / :meth:`PipelineRunner.cancel` is called (or an
incoming ``BusEndMessage`` / ``BusCancelMessage`` triggers the same path) —
spawned tasks finishing on their own does **not** unblock it.
"""

import asyncio
import gc
import signal
import uuid
from dataclasses import dataclass, field

from loguru import logger

from pipecat.bus import (
    AsyncQueueBus,
    BusAddTaskMessage,
    BusCancelMessage,
    BusCancelTaskMessage,
    BusEndMessage,
    BusEndTaskMessage,
    BusMessage,
    BusTaskRegistryMessage,
    TaskBus,
)
from pipecat.bus.subscriber import BusSubscriber
from pipecat.pipeline.base_task import BaseTask
from pipecat.pipeline.task import PipelineTask, PipelineTaskParams
from pipecat.pipeline.utils import run_setup_hook
from pipecat.registry import TaskRegistry
from pipecat.registry.types import TaskReadyData, TaskRegistryEntry
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams
from pipecat.utils.base_object import BaseObject


@dataclass
class _TaskEntry:
    """A task registered on the runner and its background asyncio task."""

    task: BaseTask
    runner_task: asyncio.Task | None = field(default=None, repr=False)


class PipelineRunner(BaseObject, BusSubscriber):
    """Manages pipeline task execution.

    Provides a high-level interface for running pipeline tasks with
    automatic signal handling (SIGINT/SIGTERM), optional garbage
    collection, proper cleanup of resources, and a task bus + registry
    for multi-task orchestration.

    Two entry points:

    - :meth:`run(task)` — block until the given pipeline task finishes.
      The most common case for a single-pipeline bot.
    - :meth:`spawn(task)` — fire-and-forget; register a child task on
      the runner's bus and start it in the background. Spawned tasks
      run alongside the main task and are cancelled when the main task
      finishes (or when :meth:`end` / :meth:`cancel` is called).

    Event handlers available:

    - ``on_ready`` — fired after the runner has finished its
      initialization and any spawned tasks have been started.
    - ``on_error`` — fired when starting a spawned task fails.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        bus: TaskBus | None = None,
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
            bus: Optional :class:`TaskBus`. Defaults to a new
                in-process :class:`AsyncQueueBus`.
            handle_sigint: Whether to automatically handle SIGINT signals.
            handle_sigterm: Whether to automatically handle SIGTERM signals.
            force_gc: Whether to force garbage collection after the main
                task completes.
            loop: Event loop to use. If None, uses the current running loop.
        """
        super().__init__(name=name or f"runner-{uuid.uuid4().hex[:8]}")

        self._bus: TaskBus = bus or AsyncQueueBus()
        self._registry = TaskRegistry(runner_name=self.name)

        self._entries: dict[str, _TaskEntry] = {}
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
    def bus(self) -> TaskBus:
        """The bus this runner hosts; shared across spawned tasks."""
        return self._bus

    @property
    def registry(self) -> TaskRegistry:
        """The task registry this runner owns."""
        return self._registry

    async def spawn(self, task: BaseTask) -> None:
        """Register a task with the runner and start it in the background.

        Can be called before or after :meth:`run`. When called after
        ``run()`` has started, the task is started immediately. Spawned
        tasks run alongside the main task and are cancelled when the
        main task finishes or when :meth:`end` / :meth:`cancel` is
        called.

        Args:
            task: The task to spawn.
        """
        if task.name in self._entries:
            logger.error(f"PipelineRunner '{self}': task '{task.name}' already exists, skipping")
            return
        task.attach(registry=self._registry, bus=self._bus)
        await self._registry.watch(task.name, self._on_local_task_ready)
        entry = _TaskEntry(task=task)
        self._entries[task.name] = entry
        logger.debug(f"PipelineRunner '{self}': spawned task '{task.name}'")

        if self._running:
            await self._start_task(entry)

    async def run(self, task: PipelineTask | None = None) -> None:
        """Run a pipeline task to completion (optionally alongside spawned tasks).

        If ``task`` is provided, blocks until that task finishes. Any
        spawned tasks are started in the background and cancelled
        when the main task finishes.

        If ``task`` is None, blocks until :meth:`end` or :meth:`cancel`
        is called (or until an incoming ``BusEndMessage`` /
        ``BusCancelMessage`` triggers the same path). Spawned tasks
        finishing on their own does **not** unblock the runner — use
        this form for hosts that have no single "main" pipeline and
        want to stay up across many spawned sessions (e.g. a FastAPI
        server). If you want the runner to finish when a specific
        pipeline finishes, pass that pipeline as ``task``.

        Args:
            task: The pipeline task to run, or None.
        """
        logger.debug(f"PipelineRunner '{self}': started running {task}")
        self._shutdown_event.clear()

        # Treat the main task as a spawned task: ``spawn`` attaches it
        # to the bus and registry, and ``_setup_session`` then starts
        # every entry (main and pre-spawned) through the same code path.
        if task is not None:
            await self.spawn(task)

        await self._setup_session()
        await self._call_event_handler("on_ready")

        # Wait for the main task's background runner task to finish
        # (or for an explicit shutdown when there's no main task).
        try:
            if task is not None:
                runner_task = self._entries[task.name].runner_task
                if runner_task is not None:
                    await runner_task
            else:
                await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass

        # Cancel any remaining spawned tasks and wait for them to finish.
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

        logger.debug(f"PipelineRunner '{self}': finished running {task}")

    async def stop_when_done(self) -> None:
        """Schedule all root pipeline tasks to stop when their current processing is complete."""
        logger.debug(f"PipelineRunner '{self}': scheduled to stop when all tasks are done")
        await asyncio.gather(
            *[
                entry.task.stop_when_done()
                for entry in self._entries.values()
                if isinstance(entry.task, PipelineTask) and entry.task.parent is None
            ]
        )

    async def end(self, reason: str | None = None) -> None:
        """Gracefully end all running tasks.

        Idempotent; subsequent calls are ignored.

        Args:
            reason: Optional human-readable reason for ending.
        """
        if self._shutdown_event.is_set():
            return
        logger.debug(f"PipelineRunner '{self}': ending gracefully (reason={reason})")
        self._shutdown_event.set()
        for name, entry in self._entries.items():
            if entry.task.parent is None:
                await self._bus.send(
                    BusEndTaskMessage(source=self.name, target=name, reason=reason)
                )

    async def cancel(self, reason: str | None = None) -> None:
        """Immediately cancel all running tasks.

        Idempotent; subsequent calls are ignored.

        Args:
            reason: Optional human-readable reason for cancelling.
        """
        if self._shutdown_event.is_set():
            return
        logger.debug(f"PipelineRunner '{self}': cancelling (reason={reason})")
        self._shutdown_event.set()
        for name, entry in self._entries.items():
            if entry.task.parent is None:
                await self._bus.send(
                    BusCancelTaskMessage(source=self.name, target=name, reason=reason)
                )

    async def on_bus_message(self, message: BusMessage) -> None:
        """Process incoming bus messages for runner-level concerns."""
        if message.source == self.name:
            return
        if isinstance(message, BusEndMessage):
            self.create_task(self.end(message.reason), "end")
        elif isinstance(message, BusCancelMessage):
            self.create_task(self.cancel(message.reason), "cancel")
        elif isinstance(message, BusAddTaskMessage) and message.task:
            await self.spawn(message.task)
        elif isinstance(message, BusTaskRegistryMessage):
            await self._handle_task_registry(message)

    async def _setup_session(self) -> None:
        """One-time per-run setup: task manager, bus, signal handlers, spawned tasks."""
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
            await self._start_task(entry)

        self._running = True

    async def _cancel_spawned_tasks(self) -> None:
        """Wait for spawned runner tasks to finish (or cancel them)."""
        remaining = [
            e.runner_task
            for e in self._entries.values()
            if e.runner_task and not e.runner_task.done()
        ]
        if not remaining:
            return
        for entry in self._entries.values():
            if entry.task.parent is None:
                await self._bus.send(
                    BusCancelTaskMessage(
                        source=self.name, target=entry.task.name, reason="runner exiting"
                    )
                )
        await asyncio.gather(*remaining, return_exceptions=True)

    async def _load_setup_files(self) -> None:
        """Run ``setup_pipeline_runner`` from each file in ``PIPECAT_SETUP_FILES``.

        A setup file may define ``setup_pipeline_runner(runner)`` to attach
        spawned tasks, event handlers, or other runner-level wiring.
        """
        await run_setup_hook(target=self, function_name="setup_pipeline_runner")

    async def _start_task(self, entry: _TaskEntry) -> None:
        """Run a registered task as a background asyncio task."""
        task = entry.task
        logger.debug(f"PipelineRunner '{self}': starting task '{task.name}'")

        entry.runner_task = self.create_task(
            self._run_task(task),
            f"task_{task.name}",
        )
        # Add the task to event loop right away without needing to `await`.
        await asyncio.sleep(0)

    async def _run_task(self, task: BaseTask) -> None:
        """Drive a registered task to completion."""
        try:
            params = PipelineTaskParams(loop=self._loop)
            await task.run(params)
        except asyncio.CancelledError:
            pass

    async def _on_local_task_ready(self, data: TaskReadyData) -> None:
        """Called when a local spawned task registers as ready."""
        if data.runner != self.name:
            return
        entry = self._entries.get(data.task_name)
        if not entry or entry.task.parent is not None:
            return
        await self._send_registry()

    async def _send_registry(self) -> None:
        """Broadcast this runner's tasks to the bus."""
        tasks = [
            TaskRegistryEntry(
                name=entry.task.name,
                parent=entry.task.parent,
                active=entry.task.active,
                bridged=entry.task.bridged,
                started_at=entry.task.started_at,
            )
            for entry in self._entries.values()
        ]
        if tasks:
            names = [t.name for t in tasks]
            logger.debug(f"PipelineRunner '{self}': broadcasting registry: {names}")
            await self._bus.send(
                BusTaskRegistryMessage(
                    source=self.name,
                    runner=self.name,
                    tasks=tasks,
                )
            )

    async def _handle_task_registry(self, message: BusTaskRegistryMessage) -> None:
        """Handle a registry message from a remote runner."""
        task_names = [t.name for t in message.tasks]
        logger.debug(
            f"PipelineRunner '{self}': received registry from '{message.runner}' "
            f"with tasks: {task_names}"
        )
        for entry in message.tasks:
            await self._registry.register(
                TaskReadyData(task_name=entry.name, runner=message.runner)
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
