#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Worker group types for structured concurrent worker execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pipecat.workers.base_worker import BaseWorker


class JobStatus(StrEnum):
    """Status of a completed worker.

    Inherits from ``str`` so values compare naturally with plain strings
    and serialize without extra handling.

    Attributes:
        COMPLETED: The worker finished successfully.
        CANCELLED: The worker was cancelled by the requester.
        FAILED: The worker failed due to a logical or business error.
        ERROR: The worker encountered an unexpected runtime error.
    """

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    ERROR = "error"


class JobError(Exception):
    """Raised when a worker is cancelled due to a worker error or timeout."""

    pass


class JobGroupError(Exception):
    """Raised when a worker group is cancelled due to a worker error or timeout."""

    pass


@dataclass
class JobGroupResponse:
    """Collected results from a completed job group.

    Parameters:
        job_id: The shared job identifier.
        responses: Collected responses keyed by worker name.
    """

    job_id: str
    responses: dict[str, dict]


@dataclass
class JobEvent:
    """An event received from a worker during a single-worker job.

    Parameters:
        type: The event type.
        data: Optional event payload.
    """

    UPDATE: ClassVar[str] = "update"
    STREAM_START: ClassVar[str] = "stream_start"
    STREAM_DATA: ClassVar[str] = "stream_data"
    STREAM_END: ClassVar[str] = "stream_end"

    type: str
    data: dict | None = None


@dataclass
class JobGroupEvent:
    """An event received from a worker during job group execution.

    Parameters:
        type: The event type.
        worker_name: The name of the worker that sent the event.
        data: Optional event payload.
    """

    UPDATE: ClassVar[str] = "update"
    STREAM_START: ClassVar[str] = "stream_start"
    STREAM_DATA: ClassVar[str] = "stream_data"
    STREAM_END: ClassVar[str] = "stream_end"

    type: str
    worker_name: str
    data: dict | None = None


@dataclass
class JobGroup:
    """Tracks a group of workers launched together.

    Parameters:
        job_id: Shared identifier for all workers in this group.
        worker_names: Names of the workers in the group.
        responses: Collected responses keyed by worker name.
        timeout_task: Optional asyncio worker that cancels the group on timeout.
        cancel_on_error: Whether to cancel the group if a worker errors.
        event_queue: Optional queue for streaming events to a
            ``JobGroupContext`` async iterator.
    """

    job_id: str
    worker_names: set[str]
    responses: dict[str, dict] = field(default_factory=dict)
    timeout_task: asyncio.Task | None = None
    cancel_on_error: bool = True
    event_queue: asyncio.Queue | None = field(default=None, repr=False)
    _done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _error: str | None = field(default=None, repr=False)

    @property
    def is_done(self) -> bool:
        """Whether the group has completed or failed."""
        return self._done.is_set()

    async def wait(self) -> None:
        """Wait for all workers in the group to respond.

        Raises:
            JobGroupError: If the group was cancelled due to error or timeout.
        """
        await self._done.wait()
        if self._error:
            raise JobGroupError(self._error)

    def complete(self) -> None:
        """Signal that all workers have responded."""
        self._done.set()
        if self.event_queue:
            self.event_queue.put_nowait(None)

    def fail(self, reason: str | None = None) -> None:
        """Signal that the group was cancelled.

        Args:
            reason: Human-readable reason for the failure.
        """
        self._error = reason
        self._done.set()
        if self.event_queue:
            self.event_queue.put_nowait(None)


class JobGroupContext:
    """Async context manager and iterator for structured job group execution.

    Sends job requests on enter, waits for all responses on exit.
    Supports ``async for`` to receive intermediate events (updates
    and streaming data) from workers while waiting for completion.

    On normal completion, results are available via ``responses``.
    On worker error (with ``cancel_on_error=True``) or timeout, raises
    ``JobGroupError``. If the ``async with`` block raises, remaining
    jobs are cancelled.

    Example::

        async with self.job_group("w1", "w2", payload=data) as tg:
            async for event in tg:
                print(f"{event.worker_name} [{event.type}]: {event.data}")

        for name, result in tg.responses.items():
            print(name, result)
    """

    def __init__(
        self,
        worker: BaseWorker,
        worker_names: tuple[str, ...],
        *,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
    ):
        """Initialize the JobGroupContext.

        Args:
            worker: The parent `BaseWorker` that owns this job group.
            worker_names: Names of the workers to send the job to.
            name: Optional job name for routing to named ``@job`` handlers.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds covering both the
                ready-wait and job execution.
            cancel_on_error: Whether to cancel the group if a worker
                errors. Defaults to True.
        """
        self._worker = worker
        self._worker_names = worker_names
        self._name = name
        self._payload = payload
        self._timeout = timeout
        self._cancel_on_error = cancel_on_error
        self._group: JobGroup | None = None

    @property
    def job_id(self) -> str:
        """The shared job identifier for this group."""
        if not self._group:
            raise RuntimeError("Job group has not been started")
        return self._group.job_id

    @property
    def responses(self) -> dict[str, dict]:
        """Collected responses keyed by worker name."""
        if not self._group:
            raise RuntimeError("Job group has not been started")
        return self._group.responses

    def __aiter__(self):
        return self

    async def __anext__(self) -> JobGroupEvent:
        if not self._group or not self._group.event_queue:
            raise StopAsyncIteration
        event = await self._group.event_queue.get()
        if event is None:
            raise StopAsyncIteration
        return event

    async def __aenter__(self) -> JobGroupContext:
        self._group = await self._worker.create_job_group_and_request_job(
            list(self._worker_names),
            name=self._name,
            payload=self._payload,
            timeout=self._timeout,
            cancel_on_error=self._cancel_on_error,
        )
        self._group.event_queue = asyncio.Queue()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            if self._group and self._group.job_id in self._worker.job_groups:
                # Shield the cleanup so it completes even if the
                # surrounding worker is being cancelled (e.g. tool
                # interruption).
                await asyncio.shield(
                    self._worker.cancel_job_group(
                        self._group.job_id, reason="context exited with error"
                    )
                )
            return False

        assert self._group is not None
        await self._group.wait()
        return False


class JobContext:
    """Async context manager and iterator for a single-worker job.

    Sends a job request on enter, waits for the response on exit.
    Supports ``async for`` to receive intermediate events (updates
    and streaming data) from the worker while waiting for completion.

    On normal completion, the result is available via ``response``.
    On worker error or timeout, raises ``JobError``. If the
    ``async with`` block raises, the job is cancelled.

    Example::

        async with self.job("worker", payload=data) as t:
            async for event in t:
                print(f"[{event.type}]: {event.data}")

        print(t.response)
    """

    def __init__(
        self,
        worker: BaseWorker,
        worker_name: str,
        *,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
    ):
        """Initialize the JobContext.

        Args:
            worker: The parent `BaseWorker` that owns this job.
            worker_name: Name of the worker to send the job to.
            name: Optional job name for routing to a named ``@job`` handler.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds covering both the
                ready-wait and job execution.
        """
        self._worker = worker
        self._worker_name = worker_name
        self._name = name
        self._payload = payload
        self._timeout = timeout
        self._group: JobGroup | None = None

    @property
    def job_id(self) -> str:
        """The job identifier."""
        if not self._group:
            raise RuntimeError("Job has not been started")
        return self._group.job_id

    @property
    def response(self) -> dict:
        """The worker's response payload."""
        if not self._group:
            raise RuntimeError("Job has not been started")
        return self._group.responses.get(self._worker_name, {})

    def __aiter__(self):
        return self

    async def __anext__(self) -> JobEvent:
        if not self._group or not self._group.event_queue:
            raise StopAsyncIteration
        event = await self._group.event_queue.get()
        if event is None:
            raise StopAsyncIteration
        return JobEvent(type=event.type, data=event.data)

    async def __aenter__(self) -> JobContext:
        self._group = await self._worker.create_job_group_and_request_job(
            [self._worker_name],
            name=self._name,
            payload=self._payload,
            timeout=self._timeout,
            cancel_on_error=True,
        )
        self._group.event_queue = asyncio.Queue()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            if self._group and self._group.job_id in self._worker.job_groups:
                # Shield the cleanup so it completes even if the
                # surrounding worker is being cancelled (e.g. tool
                # interruption).
                await asyncio.shield(
                    self._worker.cancel_job_group(
                        self._group.job_id, reason="context exited with error"
                    )
                )
            return False

        assert self._group is not None
        try:
            await self._group.wait()
        except JobGroupError as e:
            raise JobError(str(e)) from e
        return False
