#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Abstract base worker for the multi-worker framework.

Provides the `BaseWorker` class that all workers inherit from, handling
worker lifecycle, parent-child relationships, and long-running job
coordination on the bus.
"""

import asyncio
import dataclasses
import time
import uuid
from dataclasses import dataclass
from typing import Self

from loguru import logger

from pipecat.bus import (
    BusActivateWorkerMessage,
    BusAddWorkerMessage,
    BusCancelMessage,
    BusCancelWorkerMessage,
    BusDeactivateWorkerMessage,
    BusEndMessage,
    BusEndWorkerMessage,
    BusJobCancelMessage,
    BusJobRequestMessage,
    BusJobResponseMessage,
    BusJobResponseUrgentMessage,
    BusJobStreamDataMessage,
    BusJobStreamEndMessage,
    BusJobStreamStartMessage,
    BusJobUpdateMessage,
    BusJobUpdateRequestMessage,
    BusJobUpdateUrgentMessage,
    BusMessage,
    BusWorkerErrorMessage,
    BusWorkerLocalErrorMessage,
    BusWorkerReadyMessage,
    WorkerBus,
)
from pipecat.bus.messages import BusFrameMessage
from pipecat.bus.subscriber import BusSubscriber
from pipecat.pipeline.job_context import (
    JobContext,
    JobGroup,
    JobGroupContext,
    JobGroupError,
    JobGroupEvent,
    JobGroupResponse,
    JobStatus,
)
from pipecat.pipeline.job_decorator import _collect_job_handlers
from pipecat.pipeline.worker_ready_decorator import _collect_worker_ready_handlers
from pipecat.registry import WorkerRegistry
from pipecat.registry.types import WorkerErrorData, WorkerReadyData
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject


@dataclass
class WorkerParams:
    """Configuration parameters for worker execution.

    Parameters:
        task_manager: Task manager for handling asyncio tasks.
    """

    task_manager: BaseTaskManager


@dataclass
class WorkerActivationArgs:
    """Base activation arguments for any worker.

    Parameters:
        metadata: Optional structured data passed during activation.
    """

    metadata: dict | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create from a dict, ignoring unknown keys."""
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})

    def to_dict(self) -> dict:
        """Convert to a dict, excluding None values."""
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is not None
        }


class BaseWorker(BaseObject, BusSubscriber):
    """Abstract base for workers in framework.

    A worker connects to a `WorkerBus`, registers itself in the shared
    registry, accepts activation/deactivation, and exchanges job
    requests/responses with other workers. Concrete subclasses
    (e.g. `PipelineWorker`) provide the runtime that actually drives
    the worker's work.

    Overridable lifecycle methods (always call ``super()``):

    - ``on_activated(args)``: Called when this worker is activated.
    - ``on_deactivated()``: Called when this worker is deactivated.
    - ``on_worker_ready(data)``: Called when another worker is ready
      to receive messages. For local root workers, fires automatically.
      For children, fires only on the parent. For remote workers, fires
      only for workers watched via ``watch_workers()``.
    - ``on_job_request(message)``: Called when a job request is received.
    - ``on_job_response(message)``: Called when a worker sends a response.
    - ``on_job_update(message)``: Called when a worker sends a progress
      update.
    - ``on_job_update_requested(message)``: Called when the requester asks
      for a progress update.
    - ``on_job_completed(result)``: Called when all workers in a job group
      have responded.
    - ``on_job_error(message)``: Called when a worker errors and the group
      is cancelled (``cancel_on_error``).
    - ``on_job_stream_start(message)``: Called when a worker begins
      streaming.
    - ``on_job_stream_data(message)``: Called for each streaming chunk from
      a worker.
    - ``on_job_stream_end(message)``: Called when a worker finishes
      streaming.
    - ``on_job_cancelled(message)``: Called when this worker's job is
      cancelled by the requester.
    - ``on_bus_message(message)``: Called for bus messages after default
      lifecycle handling.

    Event handlers available:

    - on_activated: Worker was activated.
    - on_deactivated: Worker was deactivated.
    - on_worker_ready: Another worker is ready.
    - on_worker_failed: A child worker reported an error.
    - on_job_request: Received a job request.
    - on_job_response: A worker sent a response.
    - on_job_update: A worker sent a progress update.
    - on_job_update_requested: Requester asked for a progress update.
    - on_job_completed: All workers in a job group responded.
    - on_job_error: A worker errored and the group was cancelled.
    - on_job_stream_start: A worker started streaming.
    - on_job_stream_data: A worker sent a streaming chunk.
    - on_job_stream_end: A worker finished streaming.
    - on_job_cancelled: This worker's job was cancelled.
    - on_bus_message: A bus message was received.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        active: bool = True,
        check_dangling_tasks: bool = True,
        task_manager: BaseTaskManager | None = None,
    ):
        """Initialize the BaseWorker.

        Args:
            name: Unique name for this worker. If ``None``, an auto-generated
                name is used (useful for instances that don't participate
                in inter-worker communication).
            active: Whether the worker starts active. Defaults to True.
            check_dangling_tasks: Whether to warn about tasks left running when
                the worker finishes. Only applies when the worker owns its task
                manager; a worker sharing the runner's task manager leaves the
                report to the runner.
            task_manager: Optional task manager for handling asyncio tasks.
        """
        super().__init__(name=name, task_manager=task_manager)

        # Only the owner of a task manager reports its dangling tasks. A worker
        # that's handed its own task manager owns it; one that falls back to the
        # runner's shared task manager does not (the runner reports instead).
        self._check_dangling_tasks = check_dangling_tasks
        self._owns_task_manager = task_manager is not None

        # Runner-provided context. Populated by ``attach()`` before
        # ``run()`` is called. Accessing ``self.bus`` / ``self.registry``
        # before ``attach()`` raises.
        self._bus: WorkerBus | None = None
        self._registry: WorkerRegistry | None = None

        # Activation. Pending activation is deferred until the worker
        # starts, then on_activated fires.
        self._active = active
        self._pending_activation = active
        self._activation_args: dict | None = None

        # Worker lifecycle. Parent/children form a tree. Finished is set
        # when the worker stops.
        self._parent: str | None = None
        self._children: list[BaseWorker] = []
        self._started_at: float | None = None
        self._finished_event: asyncio.Event = asyncio.Event()

        # Job coordination. Worker state tracks active job requests
        # keyed by job_id, supporting multiple jobs in flight
        # (e.g. parallel @job handlers). Each running handler has a
        # tracked asyncio worker so it can be cancelled by system
        # messages. Requester state tracks job groups launched by
        # this worker. Job handlers are collected from @job decorated
        # methods at init.
        self._active_jobs: dict[str, BusJobRequestMessage] = {}
        self._job_handler_tasks: dict[str, asyncio.Task] = {}
        self._job_groups: dict[str, JobGroup] = {}
        self._job_handlers = _collect_job_handlers(self)
        self._job_locks: dict[str, asyncio.Lock] = {}

        # Worker-ready handlers collected from @worker_ready decorated methods.
        self._worker_ready_handlers = _collect_worker_ready_handlers(self)

        # Worker lifecycle events
        self._register_event_handler("on_activated")
        self._register_event_handler("on_deactivated")
        self._register_event_handler("on_bus_message")
        self._register_event_handler("on_job_request")
        self._register_event_handler("on_job_response")
        self._register_event_handler("on_job_update")
        self._register_event_handler("on_job_update_requested")
        self._register_event_handler("on_job_completed")
        self._register_event_handler("on_job_error")
        self._register_event_handler("on_job_stream_start")
        self._register_event_handler("on_job_stream_data")
        self._register_event_handler("on_job_stream_end")
        self._register_event_handler("on_job_cancelled")

        # Other workers
        self._register_event_handler("on_worker_ready")
        self._register_event_handler("on_worker_failed")

    @property
    def bus(self) -> WorkerBus:
        """The bus this worker is attached to.

        Raises:
            RuntimeError: If accessed before :meth:`attach` has been called.
        """
        if self._bus is None:
            raise RuntimeError(f"Worker '{self}': bus is not set; call attach() first.")
        return self._bus

    @property
    def active(self) -> bool:
        """Whether this worker is currently active."""
        return self._active

    @property
    def activation_args(self) -> dict | None:
        """The arguments from the most recent activation, or None if inactive."""
        return self._activation_args

    @property
    def parent(self) -> str | None:
        """The name of the parent worker, or None if this is a root worker."""
        return self._parent

    @property
    def registry(self) -> WorkerRegistry:
        """The shared worker registry this worker is attached to.

        Raises:
            RuntimeError: If accessed before :meth:`attach` has been called.
        """
        if self._registry is None:
            raise RuntimeError(f"Worker '{self}': registry is not set; call attach() first.")
        return self._registry

    @property
    def started_at(self) -> float | None:
        """Unix timestamp when this worker became ready, or None if not yet started."""
        return self._started_at

    @property
    def bridged(self) -> bool:
        """Whether this worker is bridged onto the bus.

        Subclasses (e.g. `PipelineWorker`) override when they auto-wrap
        their pipeline with bus edge processors.
        """
        return False

    @property
    def children(self) -> list["BaseWorker"]:
        """The list of child workers added via ``add_workers()``."""
        return self._children

    @property
    def active_jobs(self) -> dict[str, BusJobRequestMessage]:
        """Active job requests this worker is currently working on, keyed by job_id."""
        return self._active_jobs

    @property
    def job_groups(self) -> dict[str, JobGroup]:
        """Active job groups launched by this worker, keyed by job_id."""
        return self._job_groups

    async def attach(self, *, registry: WorkerRegistry, bus: WorkerBus) -> None:
        """Attach the worker to a runner-provided registry and bus.

        Called by the runner (typically from ``add_workers()``) before
        the worker is run. After this call, :attr:`registry` and
        :attr:`bus` return the provided instances, and the worker is
        subscribed to the bus — so workers added later are listening
        before any worker emits its first message.

        Args:
            registry: The shared worker registry.
            bus: The shared worker bus.
        """
        self._registry = registry
        self._bus = bus
        await self._bus.subscribe(self)

    async def cleanup(self) -> None:
        """Clean up the worker and release resources.

        Cancels running jobs, unsubscribes from the bus, and signals
        that the worker has stopped.
        """
        await super().cleanup()
        await self.stop()

    async def run(self, params: WorkerParams) -> None:
        """Run this worker until it finishes.

        The default implementation is for bus-only workers: it subscribes
        to the bus, marks the worker as started, then waits until
        :meth:`stop` (or :meth:`_finished_event`) is signalled. Subclasses
        with their own runtime (e.g. :class:`~pipecat.pipeline.worker.PipelineWorker`)
        override this method.

        Args:
            params: Configuration parameters for worker execution.
        """
        await super().setup(self._task_manager or params.task_manager)

        await self.start()
        try:
            await self._finished_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()
            self._print_dangling_tasks()

    async def start(self) -> None:
        """Mark the worker as started, register, and activate if requested."""
        self._started_at = time.time()
        await self._register_ready()
        await self._maybe_activate()
        await self._watch_decorated_workers()

    async def stop(self) -> None:
        """Clean up and signal that this worker has stopped.

        Cancels all running job groups and reports any still-active
        job requests back to their requesters as ``CANCELLED``, so
        parents aren't left waiting.
        """
        for job_id in list(self._job_groups.keys()):
            await self.cancel_job_group(job_id, reason=f"worker '{self}' stopped")
        for job_id in list(self._active_jobs.keys()):
            await self.send_job_response(job_id, status=JobStatus.CANCELLED)
        self._finished_event.set()

    async def end(self, *, reason: str | None = None) -> None:
        """Request a graceful end of the session.

        Args:
            reason: Optional human-readable reason for ending.
        """
        await self.send_bus_message(BusEndMessage(source=self.name, reason=reason))

    async def cancel(self, *, reason: str | None = None) -> None:
        """Request an immediate cancellation of all workers.

        Args:
            reason: Optional human-readable reason. Propagated through the
                runner to every root worker's ``BusCancelWorkerMessage``.
        """
        await self.send_bus_message(BusCancelMessage(source=self.name, reason=reason))

    async def wait(self) -> None:
        """Wait for this worker to finish."""
        await self._finished_event.wait()

    async def on_activated(self, args: dict | None) -> None:
        """Called when this worker is activated.

        Override in subclasses to react to activation.
        Always call ``super().on_activated(args)``.

        Args:
            args: Optional arguments from the caller.
        """
        pass

    async def on_deactivated(self) -> None:
        """Called when this worker is deactivated.

        Override in subclasses to react to deactivation.
        Always call ``super().on_deactivated()``.
        """
        pass

    async def on_worker_ready(self, data: WorkerReadyData) -> None:
        """Called when another worker is ready to receive messages.

        For local root workers this fires automatically. For remote workers
        it fires only for workers watched via ``watch_workers()``. For child
        workers it fires only on the parent that created them.

        Args:
            data: Information about the ready worker.
        """
        pass

    async def on_worker_failed(self, data: WorkerErrorData) -> None:
        """Called when a child worker reports an error.

        Args:
            data: Information about the error.
        """
        pass

    async def on_bus_message(self, message: BusMessage) -> None:
        """Called for every bus message after built-in lifecycle handling.

        Override to handle custom message types. Built-in message types
        (activation, end, cancel, job) are already dispatched to their
        respective hooks before this method is called.

        Args:
            message: The `BusMessage` to process.
        """
        # Frame messages are not handled by the base worker.
        if isinstance(message, BusFrameMessage):
            return

        # Ignore targeted messages for other workers
        if message.target and message.target != self.name:
            return

        if isinstance(message, (BusWorkerErrorMessage, BusWorkerLocalErrorMessage)):
            await self._handle_worker_error(message)
        elif isinstance(message, BusActivateWorkerMessage):
            await self._handle_worker_activate(message)
        elif isinstance(message, BusDeactivateWorkerMessage):
            await self._handle_worker_deactivate(message)
        elif isinstance(message, BusEndWorkerMessage):
            await self._handle_worker_end(message)
        elif isinstance(message, BusCancelWorkerMessage):
            await self._handle_worker_cancel(message)
        elif isinstance(message, BusJobRequestMessage):
            await self._handle_job_request(message)
        elif isinstance(message, (BusJobResponseMessage, BusJobResponseUrgentMessage)):
            await self._handle_job_response(message)
        elif isinstance(message, (BusJobUpdateMessage, BusJobUpdateUrgentMessage)):
            await self._handle_job_update(message)
        elif isinstance(message, BusJobUpdateRequestMessage):
            await self._handle_job_update_request(message)
        elif isinstance(message, BusJobCancelMessage):
            await self._handle_job_cancel(message)
        elif isinstance(message, BusJobStreamStartMessage):
            await self._handle_job_stream_start(message)
        elif isinstance(message, BusJobStreamDataMessage):
            await self._handle_job_stream_data(message)
        elif isinstance(message, BusJobStreamEndMessage):
            await self._handle_job_stream_end(message)

        await self._call_event_handler("on_bus_message", message)

    async def on_job_request(self, message: BusJobRequestMessage) -> None:
        """Called when this worker receives a job request.

        Override to perform work. Use ``send_job_update()`` to report
        progress and ``send_job_response()`` to return results.
        """
        pass

    async def on_job_response(
        self, message: BusJobResponseMessage | BusJobResponseUrgentMessage
    ) -> None:
        """Called when a worker sends a response.

        Override to process individual results as they arrive.
        """
        pass

    async def on_job_update(self, message: BusJobUpdateMessage | BusJobUpdateUrgentMessage) -> None:
        """Called when a worker sends a progress update."""
        pass

    async def on_job_update_requested(self, message: BusJobUpdateRequestMessage) -> None:
        """Called when the requester asks for a progress update.

        Override to send back a progress update via ``send_job_update()``.
        """
        pass

    async def on_job_completed(self, result: JobGroupResponse) -> None:
        """Called when all workers in a job group have responded."""
        pass

    async def on_job_error(
        self, message: BusJobResponseMessage | BusJobResponseUrgentMessage
    ) -> None:
        """Called when a job group is cancelled due to a worker error.

        Fires when a worker responds with ``ERROR`` or ``FAILED`` status
        and ``cancel_on_error`` is set. The group is cancelled and
        ``on_job_completed`` will not fire. Partial responses from
        workers that completed before the error are available in
        the job group's ``responses``.
        """
        pass

    async def on_job_stream_start(self, message: BusJobStreamStartMessage) -> None:
        """Called when a worker begins streaming."""
        pass

    async def on_job_stream_data(self, message: BusJobStreamDataMessage) -> None:
        """Called for each streaming chunk from a worker."""
        pass

    async def on_job_stream_end(self, message: BusJobStreamEndMessage) -> None:
        """Called when a worker finishes streaming."""
        pass

    async def on_job_cancelled(self, message: BusJobCancelMessage) -> None:
        """Called when this worker's job is cancelled by the requester.

        Override to clean up resources or stop in-progress work.
        """
        pass

    async def send_bus_message(self, message: BusMessage) -> None:
        """Send a message on the bus.

        Args:
            message: The `BusMessage` to send.
        """
        if self._bus:
            await self._bus.send(message)

    async def send_bus_error_message(self, error: str) -> None:
        """Report an error on the bus.

        Child workers send a local-only message to the parent.
        Root workers broadcast over the network.

        Args:
            error: Description of the error.
        """
        if self._parent:
            await self.send_bus_message(BusWorkerLocalErrorMessage(source=self.name, error=error))
        else:
            await self.send_bus_message(BusWorkerErrorMessage(source=self.name, error=error))

    async def add_workers(self, *workers: "BaseWorker", watch: bool = True) -> None:
        """Register one or more child workers under this parent.

        Each child's lifecycle (end, cancel) is automatically managed
        by this parent worker. By default, the children are also watched
        so the parent receives ``on_worker_ready`` when each one starts;
        pass ``watch=False`` to opt out (you can still call
        :meth:`watch_workers` later).

        Args:
            *workers: One or more child `BaseWorker` instances to add.
            watch: When ``True`` (the default), watch each newly added
                child so ``on_worker_ready`` fires once it registers.
                Workers that were skipped (already parented elsewhere)
                are not watched.
        """
        added_names: list[str] = []
        for worker in workers:
            if worker._parent is not None:
                logger.warning(
                    f"Worker '{worker.name}' already has parent '{worker._parent}', skipping"
                )
                continue
            worker._parent = self.name
            self._children.append(worker)
            added_names.append(worker.name)
            await self.send_bus_message(BusAddWorkerMessage(source=self.name, worker=worker))
        if watch and added_names:
            await self.watch_workers(*added_names)

    async def activate_worker(
        self,
        worker_name: str,
        *,
        args: WorkerActivationArgs | None = None,
        deactivate_self: bool = False,
    ) -> None:
        """Activate a worker by name.

        The target worker's ``on_activated`` hook will be called
        with the provided arguments.

        Args:
            worker_name: The name of the worker to activate.
            args: Optional ``WorkerActivationArgs`` forwarded to the
                target worker's ``on_activated``.
            deactivate_self: Whether to deactivate this worker before activating
                the target.
        """
        if self._active and deactivate_self:
            # Deactivate immediately (don't wait for the bus round-trip) so this
            # worker and the target are never briefly active at the same time.
            self._active = False
            await self.deactivate_worker(self.name)
        await self.send_bus_message(
            BusActivateWorkerMessage(
                source=self.name, target=worker_name, args=args.to_dict() if args else None
            )
        )

    async def deactivate_worker(self, worker_name: str) -> None:
        """Deactivate a worker by name.

        The target worker's ``on_deactivated`` hook will be called.

        Args:
            worker_name: The name of the worker to deactivate.
        """
        await self.send_bus_message(
            BusDeactivateWorkerMessage(source=self.name, target=worker_name)
        )

    async def watch_workers(self, *worker_names: str) -> None:
        """Request notification when one or more workers register.

        For each name: if the worker is already registered,
        ``on_worker_ready`` fires immediately. Otherwise
        ``on_worker_ready`` fires when the worker eventually registers.

        Args:
            *worker_names: Names of workers to watch for.
        """
        if not self._registry:
            return
        for worker_name in worker_names:
            logger.debug(f"Worker '{self}': watching for worker '{worker_name}'")
            await self._registry.watch(worker_name, self._on_watched_worker_ready)

    async def request_job(
        self,
        worker_name: str,
        *,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
    ) -> str:
        """Send a job request to a single worker (fire-and-forget).

        Waits for the worker to be ready before sending the request.
        Does not wait for the job to complete; use callbacks
        (``on_job_response``, ``on_job_completed``) or
        ``job()`` for that.

        Args:
            worker_name: Name of the worker to send the job to.
            name: Optional job name for routing to a named ``@job``
                handler on the worker.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds. If set, the job is
                automatically cancelled after this duration.

        Returns:
            The generated job_id.
        """
        group = await self.create_job_group_and_request_job(
            [worker_name],
            name=name,
            payload=payload,
            timeout=timeout,
            cancel_on_error=True,
        )
        return group.job_id

    def job(
        self,
        worker_name: str,
        *,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
    ) -> JobContext:
        """Create a single-worker job context manager.

        Waits for the worker to be ready, sends a job request, and
        waits for the response on exit. Supports ``async for`` inside
        the block to receive intermediate events (updates and streaming
        data) from the worker while waiting.

        On normal completion, the result is available via ``response``.
        On worker error or timeout, raises ``JobError``.

        Args:
            worker_name: Name of the worker to send the job to.
            name: Optional job name for routing to a named ``@job``
                handler on the worker.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds.

        Returns:
            A ``JobContext`` to use with ``async with``.

        Example::

            async with self.job("worker", payload=data) as t:
                async for event in t:
                    if event.type == JobEvent.UPDATE:
                        print(event.data)

            print(t.response)
        """
        return JobContext(
            self,
            worker_name,
            name=name,
            payload=payload,
            timeout=timeout,
        )

    async def request_job_group(
        self,
        *worker_names: str,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
    ) -> str:
        """Send a job request to multiple workers (fire-and-forget).

        Waits for all workers to be ready before sending requests.
        Does not wait for the job group to complete; use callbacks
        (``on_job_response``, ``on_job_completed``) or
        ``job_group()`` for that.

        Args:
            *worker_names: Names of the workers to send the job to.
            name: Optional job name for routing to named ``@job``
                handlers on the workers.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds. If set, the job is
                automatically cancelled after this duration.
            cancel_on_error: Whether to cancel the entire group if a
                worker responds with an error status. Defaults to True.

        Returns:
            The generated job_id shared by all workers in the group.
        """
        for worker_name in worker_names:
            if not isinstance(worker_name, str):
                raise TypeError(
                    f"{self} Expected worker name as str, got {type(worker_name).__name__}"
                )

        group = await self.create_job_group_and_request_job(
            list(worker_names),
            name=name,
            payload=payload,
            timeout=timeout,
            cancel_on_error=cancel_on_error,
        )
        return group.job_id

    def job_group(
        self,
        *worker_names: str,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
    ) -> JobGroupContext:
        """Create a job group context manager.

        Waits for workers to be ready, sends job requests, and waits
        for all responses on exit. Supports ``async for`` inside the
        block to receive intermediate events (updates and streaming
        data) from workers while waiting.

        On normal completion, results are available via ``responses``.
        On worker error (with ``cancel_on_error=True``) or timeout,
        raises ``JobGroupError``.

        Args:
            *worker_names: Names of the workers to send the job to.
            name: Optional job name for routing to named ``@job``
                handlers on the workers.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds.
            cancel_on_error: Whether to cancel the group if a worker
                errors. Defaults to True.

        Returns:
            A ``JobGroupContext`` to use with ``async with``.

        Example::

            async with self.job_group("w1", "w2", payload=data) as tg:
                async for event in tg:
                    if event.type == JobGroupEvent.UPDATE:
                        print(f"{event.worker_name}: {event.data}")

            for name, result in tg.responses.items():
                print(name, result)
        """
        for worker_name in worker_names:
            if not isinstance(worker_name, str):
                raise TypeError(
                    f"{self} Expected worker name as str, got {type(worker_name).__name__}"
                )

        return JobGroupContext(
            self,
            worker_names,
            name=name,
            payload=payload,
            timeout=timeout,
            cancel_on_error=cancel_on_error,
        )

    async def cancel_job_group(self, job_id: str, *, reason: str | None = None) -> None:
        """Cancel a running job group.

        Args:
            job_id: The job identifier to cancel.
            reason: Optional human-readable reason for cancellation.
        """
        group = self._job_groups.pop(job_id, None)
        if group:
            if group.timeout_task:
                await self.cancel_task(group.timeout_task)
            for worker_name in group.worker_names:
                await self.send_bus_message(
                    BusJobCancelMessage(
                        source=self.name, target=worker_name, job_id=job_id, reason=reason
                    )
                )
            group.fail(reason)

    async def request_job_update(self, job_id: str, worker_name: str) -> None:
        """Request a progress update from a worker.

        Args:
            job_id: The job identifier.
            worker_name: The name of the worker to request an update from.
        """
        await self.send_bus_message(
            BusJobUpdateRequestMessage(source=self.name, target=worker_name, job_id=job_id)
        )

    async def create_job_group_and_request_job(
        self,
        worker_names: list[str],
        *,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
    ) -> JobGroup:
        """Wait for workers to be ready, create a job group, and send requests.

        Waits for all workers to be registered as ready, then creates
        the group and sends a job request to each worker. Does not wait
        for the group to complete; call ``group.wait()`` or use
        ``job_group()`` for that.

        Args:
            worker_names: Names of the workers to send the job to.
            name: Optional job name for routing to named handlers.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds. Covers both the
                ready-wait and job execution.
            cancel_on_error: Whether to cancel the group if a worker
                errors. Defaults to True.

        Returns:
            The created ``JobGroup``.

        Raises:
            JobGroupError: If workers are not ready within the timeout.
        """
        all_ready = await self._wait_workers_ready(worker_names)
        try:
            await asyncio.wait_for(all_ready, timeout=timeout)
        except TimeoutError:
            raise JobGroupError("workers not ready within timeout")

        group = self._create_job_group(
            worker_names, timeout=timeout, cancel_on_error=cancel_on_error
        )

        for worker_name in worker_names:
            await self._send_job_request(worker_name, group.job_id, job_name=name, payload=payload)

        return group

    async def send_job_response(
        self,
        job_id: str,
        response: dict | None = None,
        *,
        status: JobStatus = JobStatus.COMPLETED,
        urgent: bool = False,
    ) -> None:
        """Send a job response back to the requester.

        After sending, the job is removed from the set of active jobs.

        Args:
            job_id: The identifier of the job being responded to.
            response: Optional result data.
            status: Completion status. Defaults to ``JobStatus.COMPLETED``.
            urgent: When True, the response is delivered with system
                priority, preempting queued data messages.

        Raises:
            RuntimeError: If there is no active job with this ``job_id``.
        """
        request = self._active_jobs.get(job_id)
        if request is None:
            raise RuntimeError(f"Worker '{self}': no active job '{job_id}' to respond to")
        msg_class = BusJobResponseUrgentMessage if urgent else BusJobResponseMessage
        await self.send_bus_message(
            msg_class(
                source=self.name,
                target=request.source,
                job_id=job_id,
                response=response,
                status=status,
            )
        )
        self._active_jobs.pop(job_id, None)

    async def send_job_update(
        self, job_id: str, update: dict | None = None, *, urgent: bool = False
    ) -> None:
        """Send a progress update to the requester.

        Args:
            job_id: The identifier of the job being updated.
            update: Optional progress data.
            urgent: When True, the update is delivered with system
                priority, preempting queued data messages.

        Raises:
            RuntimeError: If there is no active job with this ``job_id``.
        """
        request = self._active_jobs.get(job_id)
        if request is None:
            raise RuntimeError(f"Worker '{self}': no active job '{job_id}' to update")
        msg_class = BusJobUpdateUrgentMessage if urgent else BusJobUpdateMessage
        await self.send_bus_message(
            msg_class(
                source=self.name,
                target=request.source,
                job_id=job_id,
                update=update,
            )
        )

    async def send_job_stream_start(self, job_id: str, data: dict | None = None) -> None:
        """Begin streaming job results back to the requester.

        Args:
            job_id: The identifier of the job being streamed.
            data: Optional metadata about the stream.

        Raises:
            RuntimeError: If there is no active job with this ``job_id``.
        """
        request = self._active_jobs.get(job_id)
        if request is None:
            raise RuntimeError(f"Worker '{self}': no active job '{job_id}' to stream")
        await self.send_bus_message(
            BusJobStreamStartMessage(
                source=self.name,
                target=request.source,
                job_id=job_id,
                data=data,
            )
        )

    async def send_job_stream_data(self, job_id: str, data: dict | None = None) -> None:
        """Send a streaming chunk to the requester.

        Args:
            job_id: The identifier of the job being streamed.
            data: The chunk payload.

        Raises:
            RuntimeError: If there is no active job with this ``job_id``.
        """
        request = self._active_jobs.get(job_id)
        if request is None:
            raise RuntimeError(f"Worker '{self}': no active job '{job_id}' to stream")
        await self.send_bus_message(
            BusJobStreamDataMessage(
                source=self.name,
                target=request.source,
                job_id=job_id,
                data=data,
            )
        )

    async def send_job_stream_end(self, job_id: str, data: dict | None = None) -> None:
        """End the current stream and mark this worker's job as complete.

        Args:
            job_id: The identifier of the job being streamed.
            data: Optional final metadata.

        Raises:
            RuntimeError: If there is no active job with this ``job_id``.
        """
        request = self._active_jobs.get(job_id)
        if request is None:
            raise RuntimeError(f"Worker '{self}': no active job '{job_id}' to stream")
        await self.send_bus_message(
            BusJobStreamEndMessage(
                source=self.name,
                target=request.source,
                job_id=job_id,
                data=data,
            )
        )

    async def _register_ready(self) -> None:
        """Register this worker as ready in the shared registry.

        The registry notifies watchers (parent for children, runner
        for root workers).
        """
        if self._registry:
            # Send the bus message before registering. Registration
            # fires watchers synchronously, which may send additional
            # messages (e.g. ActivateWorker). Sending the ready message
            # first preserves correct chronological order for observers.
            await self.send_bus_message(
                BusWorkerReadyMessage(
                    source=self.name,
                    runner=self._registry.runner_name,
                    parent=self._parent,
                    active=self._active,
                    bridged=self.bridged,
                    started_at=self._started_at,
                )
            )
            await self._registry.register(
                WorkerReadyData(
                    worker_name=self.name,
                    runner=self._registry.runner_name,
                )
            )

    async def _maybe_activate(self) -> None:
        """Activate the worker, call on_activated, and fire event handlers."""
        if self._started_at is not None and self._pending_activation:
            logger.debug(f"Worker '{self}': activated")
            self._active = True
            self._pending_activation = False
            await self.on_activated(self._activation_args)
            await self._call_event_handler("on_activated", self._activation_args)

    async def _watch_decorated_workers(self) -> None:
        """Register watches for all ``@worker_ready`` decorated handlers."""
        await self.watch_workers(*self._worker_ready_handlers)

    async def _on_watched_worker_ready(self, data: WorkerReadyData) -> None:
        """Called when a watched worker is ready.

        Dispatches to the ``@worker_ready`` handler if one exists for this
        worker, otherwise proxies to ``on_worker_ready``.
        """
        logger.debug(f"Worker '{self}': worker '{data.worker_name}' ready")
        handler = self._worker_ready_handlers.get(data.worker_name)
        if handler:
            await handler(data)
        await self.on_worker_ready(data)
        await self._call_event_handler("on_worker_ready", data)

    async def _handle_worker_error(
        self, message: BusWorkerErrorMessage | BusWorkerLocalErrorMessage
    ) -> None:
        """Handle an error reported by a child or remote worker."""
        child_names = {child.name for child in self._children}
        if message.source in child_names:
            error_info = WorkerErrorData(worker_name=message.source, error=message.error)
            await self.on_worker_failed(error_info)
            await self._call_event_handler("on_worker_failed", error_info)

    async def _handle_worker_activate(self, message: BusActivateWorkerMessage) -> None:
        """Handle an activation message.

        Stores the activation arguments and marks the worker as pending
        activation, then delegates to ``_maybe_activate()``.

        Args:
            message: The ``BusActivateWorkerMessage`` requesting activation.
        """
        self._activation_args = message.args
        self._pending_activation = True
        await self._maybe_activate()

    async def _handle_worker_deactivate(self, message: BusDeactivateWorkerMessage) -> None:
        """Deactivate this worker.

        Args:
            message: The ``BusDeactivateWorkerMessage`` requesting deactivation.
        """
        logger.debug(f"Worker '{self}': deactivated")
        self._active = False
        self._activation_args = None
        await self.on_deactivated()
        await self._call_event_handler("on_deactivated")

    async def _handle_worker_end(self, message: BusEndWorkerMessage) -> None:
        """Propagate end to children, wait for them, then stop this worker.

        Subclasses with their own runtime (e.g. `PipelineWorker`) call
        :meth:`_propagate_end_to_children` directly and drive their own
        shutdown so ``_finished_event`` fires at the right moment.

        Args:
            message: The ``BusEndWorkerMessage`` requesting a graceful end.
        """
        logger.debug(f"Worker '{self}': received end")
        await self._propagate_end_to_children(message)
        await self.stop()

    async def _handle_worker_cancel(self, message: BusCancelWorkerMessage) -> None:
        """Propagate cancel to children, then stop this worker.

        Subclasses with their own runtime (e.g. `PipelineWorker`) call
        :meth:`_propagate_cancel_to_children` directly and drive their own
        shutdown so ``_finished_event`` fires at the right moment.

        Args:
            message: The ``BusCancelWorkerMessage`` requesting cancellation.
        """
        logger.debug(f"Worker '{self}': received cancel")
        await self._propagate_cancel_to_children(message)
        await self.stop()

    async def _propagate_end_to_children(self, message: BusEndWorkerMessage) -> None:
        """Forward a ``BusEndWorkerMessage`` to each child and wait for them."""
        for child in self._children:
            await self.send_bus_message(
                BusEndWorkerMessage(source=self.name, target=child.name, reason=message.reason)
            )
        await asyncio.gather(*(child.wait() for child in self._children))

    async def _propagate_cancel_to_children(self, message: BusCancelWorkerMessage) -> None:
        """Forward a ``BusCancelWorkerMessage`` to each child."""
        for child in self._children:
            await self.send_bus_message(
                BusCancelWorkerMessage(source=self.name, target=child.name, reason=message.reason)
            )

    async def _handle_job_request(self, message: BusJobRequestMessage) -> None:
        """Handle an incoming job request.

        Dispatches to @job handlers if any match, otherwise falls back
        to on_job_request. The handler always runs in its own asyncio
        worker so the bus message loop is never blocked. When the matched
        handler is marked ``sequential=True``, requests with the same
        job name are queued and run one at a time in FIFO order.
        """
        self._active_jobs[message.job_id] = message

        handler = self._job_handlers.get(message.job_name) if message.job_name else None
        if handler is None:
            handler = self.on_job_request

        lock: asyncio.Lock | None = None
        if message.job_name and getattr(handler, "job_sequential", False):
            lock = self._job_locks.setdefault(message.job_name, asyncio.Lock())

        task = self.create_task(
            self._run_job_handler(message.job_id, handler, message, lock),
            f"{self.name}::job_{message.job_name or 'default'}",
        )
        self._job_handler_tasks[message.job_id] = task

        await self._call_event_handler("on_job_request", message)

    async def _run_job_handler(
        self,
        job_id: str,
        handler,
        message,
        lock: asyncio.Lock | None = None,
    ) -> None:
        try:
            if lock is not None:
                async with lock:
                    await handler(message)
            else:
                await handler(message)
        except asyncio.CancelledError:
            pass
        finally:
            self._job_handler_tasks.pop(job_id, None)

    async def _handle_job_response(
        self, message: BusJobResponseMessage | BusJobResponseUrgentMessage
    ) -> None:
        """Handle a job response and track group completion."""
        await self.on_job_response(message)
        await self._call_event_handler("on_job_response", message)

        # Auto-cancel the group on error/failed if cancel_on_error is set
        if message.status in (JobStatus.ERROR, JobStatus.FAILED):
            group = self._job_groups.get(message.job_id)
            if group and group.cancel_on_error:
                group.responses[message.source] = message.response or {}
                await self.on_job_error(message)
                await self._call_event_handler("on_job_error", message)
                await self.cancel_job_group(
                    message.job_id, reason=f"worker '{message.source}' errored"
                )
                return

        await self._track_job_group_response(message.job_id, message.source, message.response)

    async def _handle_job_update(
        self, message: BusJobUpdateMessage | BusJobUpdateUrgentMessage
    ) -> None:
        """Handle a job progress update."""
        await self.on_job_update(message)
        await self._call_event_handler("on_job_update", message)
        self._push_job_group_event(
            message.job_id, JobGroupEvent(JobGroupEvent.UPDATE, message.source, message.update)
        )

    async def _handle_job_update_request(self, message: BusJobUpdateRequestMessage) -> None:
        """Handle a job update request from the requester."""
        if message.job_id in self._active_jobs:
            await self.on_job_update_requested(message)
            await self._call_event_handler("on_job_update_requested", message)

    async def _handle_job_cancel(self, message: BusJobCancelMessage) -> None:
        """Handle a job cancellation.

        Cancels the running handler worker (if any), calls the
        ``on_job_cancelled`` hook for cleanup, then automatically
        sends a cancelled response back to the requester. The
        requester receives ``on_job_response`` with
        ``status="cancelled"``, same path as completed or failed jobs.
        """
        if message.job_id in self._active_jobs:
            handler_task = self._job_handler_tasks.get(message.job_id)
            if handler_task:
                await self.cancel_task(handler_task)
            await self.on_job_cancelled(message)
            await self._call_event_handler("on_job_cancelled", message)
            await self.send_job_response(message.job_id, status=JobStatus.CANCELLED)

    async def _handle_job_stream_start(self, message: BusJobStreamStartMessage) -> None:
        """Handle the start of a streaming job response."""
        await self.on_job_stream_start(message)
        await self._call_event_handler("on_job_stream_start", message)
        self._push_job_group_event(
            message.job_id,
            JobGroupEvent(JobGroupEvent.STREAM_START, message.source, message.data),
        )

    async def _handle_job_stream_data(self, message: BusJobStreamDataMessage) -> None:
        """Handle a streaming job data chunk."""
        await self.on_job_stream_data(message)
        await self._call_event_handler("on_job_stream_data", message)
        self._push_job_group_event(
            message.job_id,
            JobGroupEvent(JobGroupEvent.STREAM_DATA, message.source, message.data),
        )

    async def _handle_job_stream_end(self, message: BusJobStreamEndMessage) -> None:
        """Handle the end of a streaming job response."""
        await self.on_job_stream_end(message)
        await self._call_event_handler("on_job_stream_end", message)
        self._push_job_group_event(
            message.job_id, JobGroupEvent(JobGroupEvent.STREAM_END, message.source, message.data)
        )
        await self._track_job_group_response(message.job_id, message.source, message.data)

    def _create_job_group(
        self,
        worker_names: list[str],
        *,
        timeout: float | None = None,
        cancel_on_error: bool = True,
    ) -> JobGroup:
        job_id = str(uuid.uuid4())
        group = JobGroup(
            job_id=job_id, worker_names=set(worker_names), cancel_on_error=cancel_on_error
        )
        self._job_groups[job_id] = group

        if timeout is not None:
            group.timeout_task = self.create_task(
                self._task_timeout(job_id, timeout), f"task_timeout_{job_id[:8]}"
            )

        return group

    async def _wait_workers_ready(self, worker_names: list[str]) -> asyncio.Future:
        """Return a future that resolves when all named workers are ready.

        Callers can race the returned future against a timeout or group
        done signal.

        Raises:
            RuntimeError: If the registry is not available.
        """
        if not self._registry:
            raise RuntimeError(f"Worker '{self}': registry not available")

        ready_events: dict[str, asyncio.Event] = {}
        for name in worker_names:
            event = asyncio.Event()
            ready_events[name] = event

            async def _on_ready(data, ev=event):
                ev.set()

            await self._registry.watch(name, _on_ready)

        return asyncio.ensure_future(asyncio.gather(*(ev.wait() for ev in ready_events.values())))

    async def _send_job_request(
        self,
        worker_name: str,
        job_id: str,
        job_name: str | None = None,
        payload: dict | None = None,
    ) -> None:
        await self.send_bus_message(
            BusJobRequestMessage(
                source=self.name,
                target=worker_name,
                job_id=job_id,
                job_name=job_name,
                payload=payload,
            )
        )

    async def _task_timeout(self, job_id: str, timeout: float) -> None:
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        await self.cancel_job_group(job_id, reason="timeout")

    def _push_job_group_event(self, job_id: str, event: JobGroupEvent) -> None:
        group = self._job_groups.get(job_id)
        if group and group.event_queue:
            group.event_queue.put_nowait(event)

    async def _track_job_group_response(
        self, job_id: str, source: str, response: dict | None
    ) -> None:
        """Record a worker's response and fire completion when all have responded."""
        group = self._job_groups.get(job_id)
        if group:
            group.responses[source] = response or {}
            if group.responses.keys() >= group.worker_names:
                if group.timeout_task:
                    await self.cancel_task(group.timeout_task)
                del self._job_groups[job_id]
                result = JobGroupResponse(job_id=job_id, responses=group.responses)
                await self.on_job_completed(result)
                await self._call_event_handler("on_job_completed", result)
                group.complete()

    def _print_dangling_tasks(self) -> None:
        """Warn about tasks left running on the task manager this worker owns."""
        if self._check_dangling_tasks and self._owns_task_manager:
            tasks = [t.get_name() for t in self.task_manager.current_tasks()]
            if tasks:
                logger.warning(f"{self} dangling tasks detected: {tasks}")
