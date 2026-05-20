#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Abstract base task for the multi-task framework.

Provides the `BaseTask` class that all tasks inherit from, handling
task lifecycle, parent-child relationships, and long-running job
coordination on the bus.
"""

import asyncio
import dataclasses
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from loguru import logger

if TYPE_CHECKING:
    from pipecat.pipeline.task import PipelineTaskParams

from pipecat.bus import (
    BusActivateTaskMessage,
    BusAddTaskMessage,
    BusCancelMessage,
    BusCancelTaskMessage,
    BusDeactivateTaskMessage,
    BusEndMessage,
    BusEndTaskMessage,
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
    BusTaskErrorMessage,
    BusTaskLocalErrorMessage,
    BusTaskReadyMessage,
    TaskBus,
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
from pipecat.pipeline.task_ready_decorator import _collect_task_ready_handlers
from pipecat.registry import TaskRegistry
from pipecat.registry.types import TaskErrorData, TaskReadyData
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams
from pipecat.utils.base_object import BaseObject


@dataclass
class TaskActivationArgs:
    """Base activation arguments for any task.

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


class BaseTask(BaseObject, BusSubscriber):
    """Abstract base for tasks in the multi-task framework.

    A task connects to a `TaskBus`, registers itself in the shared
    registry, accepts activation/deactivation, and exchanges job
    requests/responses with other tasks. Concrete subclasses
    (e.g. `PipelineTask`) provide the runtime that actually drives
    the task's work.

    Overridable lifecycle methods (always call ``super()``):

    - ``on_activated(args)``: Called when this task is activated.
    - ``on_deactivated()``: Called when this task is deactivated.
    - ``on_task_ready(data)``: Called when another task is ready
      to receive messages. For local root tasks, fires automatically.
      For children, fires only on the parent. For remote tasks, fires
      only for tasks watched via ``watch_task()``.
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
    - ``on_job_cancelled(message)``: Called when this task's job is
      cancelled by the requester.
    - ``on_bus_message(message)``: Called for bus messages after default
      lifecycle handling.

    Event handlers available:

    - on_activated: Task was activated.
    - on_deactivated: Task was deactivated.
    - on_task_ready: Another task is ready.
    - on_task_failed: A child task reported an error.
    - on_job_request: Received a job request.
    - on_job_response: A worker sent a response.
    - on_job_update: A worker sent a progress update.
    - on_job_update_requested: Requester asked for a progress update.
    - on_job_completed: All workers in a job group responded.
    - on_job_error: A worker errored and the group was cancelled.
    - on_job_stream_start: A worker started streaming.
    - on_job_stream_data: A worker sent a streaming chunk.
    - on_job_stream_end: A worker finished streaming.
    - on_job_cancelled: This task's job was cancelled.
    - on_bus_message: A bus message was received.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        active: bool = True,
    ):
        """Initialize the BaseTask.

        Args:
            name: Unique name for this task. If ``None``, an auto-generated
                name is used (useful for instances that don't participate
                in inter-task communication).
            active: Whether the task starts active. Defaults to True.
        """
        super().__init__(name=name)

        # Runner-provided context. Populated by ``attach()`` before
        # ``run()`` is called. Accessing ``self.bus`` / ``self.registry``
        # before ``attach()`` raises.
        self._bus: TaskBus | None = None
        self._registry: TaskRegistry | None = None

        # Activation. Pending activation is deferred until the task
        # starts, then on_activated fires.
        self._active = active
        self._pending_activation = active
        self._activation_args: dict | None = None

        # Task lifecycle. Parent/children form a tree. Finished is set
        # when the task stops.
        self._parent: str | None = None
        self._children: list[BaseTask] = []
        self._started_at: float | None = None
        self._finished_event: asyncio.Event = asyncio.Event()

        # Job coordination. Worker state tracks active job requests
        # keyed by job_id, supporting multiple jobs in flight
        # (e.g. parallel @job handlers). Each running handler has a
        # tracked asyncio task so it can be cancelled by system
        # messages. Requester state tracks job groups launched by
        # this task. Job handlers are collected from @job decorated
        # methods at init.
        self._active_jobs: dict[str, BusJobRequestMessage] = {}
        self._job_handler_tasks: dict[str, asyncio.Task] = {}
        self._job_groups: dict[str, JobGroup] = {}
        self._job_handlers = _collect_job_handlers(self)
        self._job_locks: dict[str, asyncio.Lock] = {}

        # Task-ready handlers collected from @task_ready decorated methods.
        self._task_ready_handlers = _collect_task_ready_handlers(self)

        # Task lifecycle events
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

        # Other tasks
        self._register_event_handler("on_task_ready")
        self._register_event_handler("on_task_failed")

    @property
    def bus(self) -> TaskBus:
        """The bus this task is attached to.

        Raises:
            RuntimeError: If accessed before :meth:`attach` has been called.
        """
        if self._bus is None:
            raise RuntimeError(f"Task '{self}': bus is not set; call attach() first.")
        return self._bus

    @property
    def active(self) -> bool:
        """Whether this task is currently active."""
        return self._active

    @property
    def activation_args(self) -> dict | None:
        """The arguments from the most recent activation, or None if inactive."""
        return self._activation_args

    @property
    def parent(self) -> str | None:
        """The name of the parent task, or None if this is a root task."""
        return self._parent

    @property
    def registry(self) -> TaskRegistry:
        """The shared task registry this task is attached to.

        Raises:
            RuntimeError: If accessed before :meth:`attach` has been called.
        """
        if self._registry is None:
            raise RuntimeError(f"Task '{self}': registry is not set; call attach() first.")
        return self._registry

    @property
    def started_at(self) -> float | None:
        """Unix timestamp when this task became ready, or None if not yet started."""
        return self._started_at

    @property
    def bridged(self) -> bool:
        """Whether this task is bridged onto the bus.

        Subclasses (e.g. `PipelineTask`) override when they auto-wrap
        their pipeline with bus edge processors.
        """
        return False

    @property
    def children(self) -> list["BaseTask"]:
        """The list of child tasks added via ``add_task()``."""
        return self._children

    @property
    def active_jobs(self) -> dict[str, BusJobRequestMessage]:
        """Active job requests this task is currently working on, keyed by job_id."""
        return self._active_jobs

    @property
    def job_groups(self) -> dict[str, JobGroup]:
        """Active job groups launched by this task, keyed by job_id."""
        return self._job_groups

    def attach(self, *, registry: TaskRegistry, bus: TaskBus) -> None:
        """Attach the task to a runner-provided registry and bus.

        Called by the runner (typically from ``spawn()``) before the
        task is run. After this call, :attr:`registry` and :attr:`bus`
        return the provided instances.

        Args:
            registry: The shared task registry.
            bus: The shared task bus.
        """
        self._registry = registry
        self._bus = bus

    async def cleanup(self) -> None:
        """Clean up the task and release resources.

        Cancels running jobs, unsubscribes from the bus, and signals
        that the task has stopped.
        """
        await super().cleanup()
        await self.stop()

    async def run(self, params: "PipelineTaskParams") -> None:
        """Run this task until it finishes.

        The default implementation is for bus-only tasks: it subscribes
        to the bus, marks the task as started, then waits until
        :meth:`stop` (or :meth:`_finished_event`) is signalled. Subclasses
        with their own runtime (e.g. :class:`~pipecat.pipeline.task.PipelineTask`)
        override this method.

        Args:
            params: Configuration parameters for task execution.
        """
        task_manager = TaskManager()
        task_manager.setup(TaskManagerParams(loop=params.loop))
        await super().setup(task_manager)

        if self._bus is not None:
            await self._bus.subscribe(self)

        await self.start()
        try:
            await self._finished_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def start(self) -> None:
        """Mark the task as started, register, and activate if requested."""
        self._started_at = time.time()
        await self._register_ready()
        await self._maybe_activate()
        await self._watch_decorated_tasks()

    async def stop(self) -> None:
        """Clean up and signal that this task has stopped.

        Cancels all running job groups and reports any still-active
        job requests back to their requesters as ``CANCELLED``, so
        parents aren't left waiting.
        """
        for job_id in list(self._job_groups.keys()):
            await self.cancel_job_group(job_id, reason=f"task '{self}' stopped")
        for job_id in list(self._active_jobs.keys()):
            await self.send_job_response(job_id, status=JobStatus.CANCELLED)
        self._finished_event.set()

    async def end(self, *, reason: str | None = None) -> None:
        """Request a graceful end of the session.

        Args:
            reason: Optional human-readable reason for ending.
        """
        await self.send_bus_message(BusEndMessage(source=self.name, reason=reason))

    async def cancel(self) -> None:
        """Request an immediate cancellation of all tasks."""
        await self.send_bus_message(BusCancelMessage(source=self.name))

    async def wait(self) -> None:
        """Wait for this task to finish."""
        await self._finished_event.wait()

    async def on_activated(self, args: dict | None) -> None:
        """Called when this task is activated.

        Override in subclasses to react to activation.
        Always call ``super().on_activated(args)``.

        Args:
            args: Optional arguments from the caller.
        """
        pass

    async def on_deactivated(self) -> None:
        """Called when this task is deactivated.

        Override in subclasses to react to deactivation.
        Always call ``super().on_deactivated()``.
        """
        pass

    async def on_task_ready(self, data: TaskReadyData) -> None:
        """Called when another task is ready to receive messages.

        For local root tasks this fires automatically. For remote tasks
        it fires only for tasks watched via ``watch_task()``. For child
        tasks it fires only on the parent that created them.

        Args:
            data: Information about the ready task.
        """
        pass

    async def on_task_failed(self, data: TaskErrorData) -> None:
        """Called when a child task reports an error.

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
        # Frame messages are not handled by the base task.
        if isinstance(message, BusFrameMessage):
            return

        # Ignore targeted messages for other tasks
        if message.target and message.target != self.name:
            return

        if isinstance(message, (BusTaskErrorMessage, BusTaskLocalErrorMessage)):
            await self._handle_task_error(message)
        elif isinstance(message, BusActivateTaskMessage):
            await self._handle_task_activate(message)
        elif isinstance(message, BusDeactivateTaskMessage):
            await self._handle_task_deactivate(message)
        elif isinstance(message, BusEndTaskMessage):
            await self._handle_task_end(message)
        elif isinstance(message, BusCancelTaskMessage):
            await self._handle_task_cancel(message)
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
        """Called when this task receives a job request.

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
        """Called when this task's job is cancelled by the requester.

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

        Child tasks send a local-only message to the parent.
        Root tasks broadcast over the network.

        Args:
            error: Description of the error.
        """
        if self._parent:
            await self.send_bus_message(BusTaskLocalErrorMessage(source=self.name, error=error))
        else:
            await self.send_bus_message(BusTaskErrorMessage(source=self.name, error=error))

    async def add_task(self, task: "BaseTask") -> None:
        """Register a child task under this parent.

        The child's lifecycle (end, cancel) is automatically managed
        by this parent task. To receive ``on_task_ready`` when the
        child starts, call ``watch_task(task.name)``.

        Args:
            task: The child `BaseTask` instance to add.
        """
        if task._parent is not None:
            logger.error(f"Task '{task.name}' already has parent '{task._parent}', skipping")
            return
        task._parent = self.name
        self._children.append(task)
        await self.send_bus_message(BusAddTaskMessage(source=self.name, task=task))

    async def activate_task(
        self,
        task_name: str,
        *,
        args: TaskActivationArgs | None = None,
        deactivate_self: bool = False,
    ) -> None:
        """Activate a task by name.

        The target task's ``on_activated`` hook will be called
        with the provided arguments.

        Args:
            task_name: The name of the task to activate.
            args: Optional ``TaskActivationArgs`` forwarded to the
                target task's ``on_activated``.
            deactivate_self: Whether to deactivate this task before activating
                the target.
        """
        if self._active and deactivate_self:
            await self.deactivate_task(self.name)
        await self.send_bus_message(
            BusActivateTaskMessage(
                source=self.name, target=task_name, args=args.to_dict() if args else None
            )
        )

    async def deactivate_task(self, task_name: str) -> None:
        """Deactivate a task by name.

        The target task's ``on_deactivated`` hook will be called.

        Args:
            task_name: The name of the task to deactivate.
        """
        await self.send_bus_message(BusDeactivateTaskMessage(source=self.name, target=task_name))

    async def watch_task(self, task_name: str) -> None:
        """Request notification when a task registers.

        If the task is already registered, ``on_task_ready`` fires
        immediately. Otherwise ``on_task_ready`` fires when the
        task eventually registers.

        Args:
            task_name: The name of the task to watch for.
        """
        if self._registry:
            logger.debug(f"Task '{self}': watching for task '{task_name}'")
            await self._registry.watch(task_name, self._on_watched_task_ready)

    async def request_job(
        self,
        task_name: str,
        *,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
    ) -> str:
        """Send a job request to a single task (fire-and-forget).

        Waits for the task to be ready before sending the request.
        Does not wait for the job to complete; use callbacks
        (``on_job_response``, ``on_job_completed``) or
        ``job()`` for that.

        Args:
            task_name: Name of the task to send the job to.
            name: Optional job name for routing to a named ``@job``
                handler on the worker.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds. If set, the job is
                automatically cancelled after this duration.

        Returns:
            The generated job_id.
        """
        group = await self.create_job_group_and_request_job(
            [task_name],
            name=name,
            payload=payload,
            timeout=timeout,
            cancel_on_error=True,
        )
        return group.job_id

    def job(
        self,
        task_name: str,
        *,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
    ) -> JobContext:
        """Create a single-task job context manager.

        Waits for the task to be ready, sends a job request, and
        waits for the response on exit. Supports ``async for`` inside
        the block to receive intermediate events (updates and streaming
        data) from the worker while waiting.

        On normal completion, the result is available via ``response``.
        On worker error or timeout, raises ``JobError``.

        Args:
            task_name: Name of the task to send the job to.
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
            task_name,
            name=name,
            payload=payload,
            timeout=timeout,
        )

    async def request_job_group(
        self,
        *task_names: str,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
    ) -> str:
        """Send a job request to multiple tasks (fire-and-forget).

        Waits for all tasks to be ready before sending requests.
        Does not wait for the job group to complete; use callbacks
        (``on_job_response``, ``on_job_completed``) or
        ``job_group()`` for that.

        Args:
            *task_names: Names of the tasks to send the job to.
            name: Optional job name for routing to named ``@job``
                handlers on the workers.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds. If set, the job is
                automatically cancelled after this duration.
            cancel_on_error: Whether to cancel the entire group if a
                worker responds with an error status. Defaults to True.

        Returns:
            The generated job_id shared by all tasks in the group.
        """
        for task_name in task_names:
            if not isinstance(task_name, str):
                raise TypeError(f"{self} Expected task name as str, got {type(task_name).__name__}")

        group = await self.create_job_group_and_request_job(
            list(task_names),
            name=name,
            payload=payload,
            timeout=timeout,
            cancel_on_error=cancel_on_error,
        )
        return group.job_id

    def job_group(
        self,
        *task_names: str,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
    ) -> JobGroupContext:
        """Create a job group context manager.

        Waits for tasks to be ready, sends job requests, and waits
        for all responses on exit. Supports ``async for`` inside the
        block to receive intermediate events (updates and streaming
        data) from workers while waiting.

        On normal completion, results are available via ``responses``.
        On worker error (with ``cancel_on_error=True``) or timeout,
        raises ``JobGroupError``.

        Args:
            *task_names: Names of the tasks to send the job to.
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
                        print(f"{event.task_name}: {event.data}")

            for name, result in tg.responses.items():
                print(name, result)
        """
        for task_name in task_names:
            if not isinstance(task_name, str):
                raise TypeError(f"{self} Expected task name as str, got {type(task_name).__name__}")

        return JobGroupContext(
            self,
            task_names,
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
            for task_name in group.task_names:
                await self.send_bus_message(
                    BusJobCancelMessage(
                        source=self.name, target=task_name, job_id=job_id, reason=reason
                    )
                )
            group.fail(reason)

    async def request_job_update(self, job_id: str, task_name: str) -> None:
        """Request a progress update from a worker.

        Args:
            job_id: The job identifier.
            task_name: The name of the worker to request an update from.
        """
        await self.send_bus_message(
            BusJobUpdateRequestMessage(source=self.name, target=task_name, job_id=job_id)
        )

    async def create_job_group_and_request_job(
        self,
        task_names: list[str],
        *,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
    ) -> JobGroup:
        """Wait for tasks to be ready, create a job group, and send requests.

        Waits for all tasks to be registered as ready, then creates
        the group and sends a job request to each task. Does not wait
        for the group to complete; call ``group.wait()`` or use
        ``job_group()`` for that.

        Args:
            task_names: Names of the tasks to send the job to.
            name: Optional job name for routing to named handlers.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds. Covers both the
                ready-wait and job execution.
            cancel_on_error: Whether to cancel the group if a worker
                errors. Defaults to True.

        Returns:
            The created ``JobGroup``.

        Raises:
            JobGroupError: If tasks are not ready within the timeout.
        """
        all_ready = await self._wait_tasks_ready(task_names)
        try:
            await asyncio.wait_for(all_ready, timeout=timeout)
        except TimeoutError:
            raise JobGroupError("tasks not ready within timeout")

        group = self._create_job_group(task_names, timeout=timeout, cancel_on_error=cancel_on_error)

        for task_name in task_names:
            await self._send_job_request(task_name, group.job_id, job_name=name, payload=payload)

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
            raise RuntimeError(f"Task '{self}': no active job '{job_id}' to respond to")
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
            raise RuntimeError(f"Task '{self}': no active job '{job_id}' to update")
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
            raise RuntimeError(f"Task '{self}': no active job '{job_id}' to stream")
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
            raise RuntimeError(f"Task '{self}': no active job '{job_id}' to stream")
        await self.send_bus_message(
            BusJobStreamDataMessage(
                source=self.name,
                target=request.source,
                job_id=job_id,
                data=data,
            )
        )

    async def send_job_stream_end(self, job_id: str, data: dict | None = None) -> None:
        """End the current stream and mark this task's job as complete.

        Args:
            job_id: The identifier of the job being streamed.
            data: Optional final metadata.

        Raises:
            RuntimeError: If there is no active job with this ``job_id``.
        """
        request = self._active_jobs.get(job_id)
        if request is None:
            raise RuntimeError(f"Task '{self}': no active job '{job_id}' to stream")
        await self.send_bus_message(
            BusJobStreamEndMessage(
                source=self.name,
                target=request.source,
                job_id=job_id,
                data=data,
            )
        )

    async def _register_ready(self) -> None:
        """Register this task as ready in the shared registry.

        The registry notifies watchers (parent for children, runner
        for root tasks).
        """
        if self._registry:
            # Send the bus message before registering. Registration
            # fires watchers synchronously, which may send additional
            # messages (e.g. ActivateTask). Sending the ready message
            # first preserves correct chronological order for observers.
            await self.send_bus_message(
                BusTaskReadyMessage(
                    source=self.name,
                    runner=self._registry.runner_name,
                    parent=self._parent,
                    active=self._active,
                    bridged=self.bridged,
                    started_at=self._started_at,
                )
            )
            await self._registry.register(
                TaskReadyData(
                    task_name=self.name,
                    runner=self._registry.runner_name,
                )
            )

    async def _maybe_activate(self) -> None:
        """Activate the task, call on_activated, and fire event handlers."""
        if self._started_at is not None and self._pending_activation:
            logger.debug(f"Task '{self}': activated")
            self._active = True
            self._pending_activation = False
            await self.on_activated(self._activation_args)
            await self._call_event_handler("on_activated", self._activation_args)

    async def _watch_decorated_tasks(self) -> None:
        """Register watches for all ``@task_ready`` decorated handlers."""
        for task_name in self._task_ready_handlers:
            await self.watch_task(task_name)

    async def _on_watched_task_ready(self, data: TaskReadyData) -> None:
        """Called when a watched task is ready.

        Dispatches to the ``@task_ready`` handler if one exists for this
        task, otherwise proxies to ``on_task_ready``.
        """
        logger.debug(f"Task '{self}': task '{data.task_name}' ready")
        handler = self._task_ready_handlers.get(data.task_name)
        if handler:
            await handler(data)
        await self.on_task_ready(data)
        await self._call_event_handler("on_task_ready", data)

    async def _handle_task_error(
        self, message: BusTaskErrorMessage | BusTaskLocalErrorMessage
    ) -> None:
        """Handle an error reported by a child or remote task."""
        child_names = {child.name for child in self._children}
        if message.source in child_names:
            error_info = TaskErrorData(task_name=message.source, error=message.error)
            await self.on_task_failed(error_info)
            await self._call_event_handler("on_task_failed", error_info)

    async def _handle_task_activate(self, message: BusActivateTaskMessage) -> None:
        """Handle an activation message.

        Stores the activation arguments and marks the task as pending
        activation, then delegates to ``_maybe_activate()``.

        Args:
            message: The ``BusActivateTaskMessage`` requesting activation.
        """
        self._activation_args = message.args
        self._pending_activation = True
        await self._maybe_activate()

    async def _handle_task_deactivate(self, message: BusDeactivateTaskMessage) -> None:
        """Deactivate this task.

        Args:
            message: The ``BusDeactivateTaskMessage`` requesting deactivation.
        """
        logger.debug(f"Task '{self}': deactivated")
        self._active = False
        self._activation_args = None
        await self.on_deactivated()
        await self._call_event_handler("on_deactivated")

    async def _handle_task_end(self, message: BusEndTaskMessage) -> None:
        """Propagate end to children, wait for them, then stop this task.

        Subclasses with their own runtime (e.g. `PipelineTask`) call
        :meth:`_propagate_end_to_children` directly and drive their own
        shutdown so ``_finished_event`` fires at the right moment.

        Args:
            message: The ``BusEndTaskMessage`` requesting a graceful end.
        """
        logger.debug(f"Task '{self}': received end")
        await self._propagate_end_to_children(message)
        await self.stop()

    async def _handle_task_cancel(self, message: BusCancelTaskMessage) -> None:
        """Propagate cancel to children, then stop this task.

        Subclasses with their own runtime (e.g. `PipelineTask`) call
        :meth:`_propagate_cancel_to_children` directly and drive their own
        shutdown so ``_finished_event`` fires at the right moment.

        Args:
            message: The ``BusCancelTaskMessage`` requesting cancellation.
        """
        logger.debug(f"Task '{self}': received cancel")
        await self._propagate_cancel_to_children(message)
        await self.stop()

    async def _propagate_end_to_children(self, message: BusEndTaskMessage) -> None:
        """Forward a ``BusEndTaskMessage`` to each child and wait for them."""
        for child in self._children:
            await self.send_bus_message(
                BusEndTaskMessage(source=self.name, target=child.name, reason=message.reason)
            )
        await asyncio.gather(*(child.wait() for child in self._children))

    async def _propagate_cancel_to_children(self, message: BusCancelTaskMessage) -> None:
        """Forward a ``BusCancelTaskMessage`` to each child."""
        for child in self._children:
            await self.send_bus_message(
                BusCancelTaskMessage(source=self.name, target=child.name, reason=message.reason)
            )

    async def _handle_job_request(self, message: BusJobRequestMessage) -> None:
        """Handle an incoming job request.

        Dispatches to @job handlers if any match, otherwise falls back
        to on_job_request. The handler always runs in its own asyncio
        task so the bus message loop is never blocked. When the matched
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

        Cancels the running handler task (if any), calls the
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
        task_names: list[str],
        *,
        timeout: float | None = None,
        cancel_on_error: bool = True,
    ) -> JobGroup:
        job_id = str(uuid.uuid4())
        group = JobGroup(job_id=job_id, task_names=set(task_names), cancel_on_error=cancel_on_error)
        self._job_groups[job_id] = group

        if timeout is not None:
            group.timeout_task = self.create_task(
                self._task_timeout(job_id, timeout), f"task_timeout_{job_id[:8]}"
            )

        return group

    async def _wait_tasks_ready(self, task_names: list[str]) -> asyncio.Future:
        """Return a future that resolves when all named tasks are ready.

        Callers can race the returned future against a timeout or group
        done signal.

        Raises:
            RuntimeError: If the registry is not available.
        """
        if not self._registry:
            raise RuntimeError(f"Task '{self}': registry not available")

        ready_events: dict[str, asyncio.Event] = {}
        for name in task_names:
            event = asyncio.Event()
            ready_events[name] = event

            async def _on_ready(data, ev=event):
                ev.set()

            await self._registry.watch(name, _on_ready)

        return asyncio.ensure_future(asyncio.gather(*(ev.wait() for ev in ready_events.values())))

    async def _send_job_request(
        self,
        task_name: str,
        job_id: str,
        job_name: str | None = None,
        payload: dict | None = None,
    ) -> None:
        await self.send_bus_message(
            BusJobRequestMessage(
                source=self.name,
                target=task_name,
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
            if group.responses.keys() >= group.task_names:
                if group.timeout_task:
                    await self.cancel_task(group.timeout_task)
                del self._job_groups[job_id]
                result = JobGroupResponse(job_id=job_id, responses=group.responses)
                await self.on_job_completed(result)
                await self._call_event_handler("on_job_completed", result)
                group.complete()
