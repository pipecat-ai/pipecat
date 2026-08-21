#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Worker group types for structured concurrent worker execution."""

from __future__ import annotations

import asyncio
import warnings
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from pipecat.workers.base_worker import BaseWorker


class JobParams(BaseModel):
    """Configuration for a job sent to a single worker.

    Parameters:
        name: Optional job name, routing the request to the matching
            ``@job(name=...)`` handler on the worker.
        payload: Optional structured data describing the work.
        timeout: Optional timeout in seconds, covering both the wait for
            the worker to become ready and the job itself.
        label: Optional human-readable description of the work, e.g.
            ``"Research: Radiohead"``. A
            :class:`~pipecat.workers.base_ui_worker.BaseUIWorker` titles
            the client's progress card with it.
        cancellable: Whether an external requester, such as the client UI,
            may ask for the job to be cancelled. Cancellation the worker
            initiates itself (shutdown, timeout, ``cancel_on_error``)
            proceeds either way.
    """

    name: str | None = None
    payload: dict | None = None
    timeout: float | None = None
    label: str | None = None
    cancellable: bool = True


class JobGroupParams(JobParams):
    """Configuration for a job sent to several workers at once.

    Carries every :class:`JobParams` field, which apply to the group as a
    whole, plus how the group reacts to one of its workers failing.

    Parameters:
        cancel_on_error: Whether a worker responding with an error status
            cancels the rest of the group.
    """

    cancel_on_error: bool = True


#: Either params class, so ``resolve_job_params`` hands back what it was asked for.
JobParamsT = TypeVar("JobParamsT", bound=JobParams)


def resolve_job_params(
    params: JobParamsT | None,
    params_class: type[JobParamsT],
    **deprecated,
) -> JobParamsT:
    """Fold the deprecated per-argument job spelling into a params object.

    Dispatch methods accept a params object and, until 2.0.0, the individual
    arguments it replaced. Call this once at the entry point a caller reaches,
    then pass the result down as ``params`` so nothing warns twice.

    .. deprecated:: 1.8.0
        No replacement. Will be removed in 2.0.0, along with the individual
        arguments it exists to absorb.

    Args:
        params: The params object the caller passed, if any.
        params_class: The class to build when only individual arguments came in.
        **deprecated: The individual arguments. Each defaults to None, so a
            non-None value means the caller passed it explicitly.

    Returns:
        The params to dispatch with.

    Raises:
        TypeError: If both ``params`` and individual arguments were passed.
    """
    passed = {key: value for key, value in deprecated.items() if value is not None}
    if not passed:
        return params or params_class()
    if params is not None:
        raise TypeError(f"Pass either `params` or `{'`, `'.join(sorted(passed))}`, not both.")
    warnings.warn(
        f"Passing `{'`, `'.join(sorted(passed))}` to a job dispatch method is deprecated since "
        f"1.8.0 and will be removed in 2.0.0. Use `params={params_class.__name__}(...)` instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    return params_class(**passed)


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
        worker_names: Names of the workers in the group, in dispatch order.
        responses: Collected responses keyed by worker name.
        timeout_task: Optional asyncio worker that cancels the group on timeout.
        cancel_on_error: Whether to cancel the group if a worker errors.
        label: Optional human-readable description of the work, from
            :attr:`JobGroupParams.label`.
        cancellable: Whether an external requester may ask for the group to be
            cancelled, from :attr:`JobGroupParams.cancellable`.
        terminated: Names of the workers that have reached a terminal state,
            whether by responding or by ending their stream.
        event_queue: Optional queue for streaming events to a
            ``JobGroupContext`` async iterator.
    """

    job_id: str
    worker_names: list[str]
    responses: dict[str, dict] = field(default_factory=dict)
    timeout_task: asyncio.Task | None = None
    cancel_on_error: bool = True
    label: str | None = None
    cancellable: bool = True
    terminated: set[str] = field(default_factory=set)
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

        async with self.job_group(
            "w1", "w2", params=JobGroupParams(payload=data)
        ) as tg:
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
        params: JobGroupParams | None = None,
    ):
        """Initialize the JobGroupContext.

        Args:
            worker: The parent `BaseWorker` that owns this job group.
            worker_names: Names of the workers to send the job to.
            params: How to run the group. Defaults to
                :class:`JobGroupParams` defaults.
        """
        self._worker = worker
        self._worker_names = worker_names
        self._params = params or JobGroupParams()
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
            params=self._params,
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

        async with self.job("worker", params=JobParams(payload=data)) as t:
            async for event in t:
                print(f"[{event.type}]: {event.data}")

        print(t.response)
    """

    def __init__(
        self,
        worker: BaseWorker,
        worker_name: str,
        *,
        params: JobParams | None = None,
    ):
        """Initialize the JobContext.

        Args:
            worker: The parent `BaseWorker` that owns this job.
            worker_name: Name of the worker to send the job to.
            params: How to run the job. Defaults to :class:`JobParams`
                defaults.
        """
        self._worker = worker
        self._worker_name = worker_name
        self._params = params or JobParams()
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
            params=JobGroupParams(**self._params.model_dump()),
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
