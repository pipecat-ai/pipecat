#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""BaseUIWorker: a worker whose jobs and job groups can surface on the client UI.

Extends ``BaseWorker`` with client-visible job dispatch: pass
``ui=UIJobGroupOptions(...)`` to ``job(...)``, ``job_group(...)``,
``request_job(...)``, or ``request_job_group(...)`` and the work's lifecycle
-- start, per-worker progress, and completion -- streams to the UI client as
``ui-job-group`` envelopes, with the client's reserved ``__cancel_job_group``
event honored for groups registered as cancellable. No LLM is involved:
``BaseUIWorker`` is instantiable as-is (its inherited ``run()`` is a bus-only
loop), so a plain pipeline app can register one on the runner as a dispatcher
and call it from tools. ``UIWorker`` inherits this class and adds the
LLM-driven page interaction (snapshots, UI events, commands).
"""

import time
from dataclasses import dataclass, field

from loguru import logger

from pipecat.bus.messages import (
    BusJobResponseMessage,
    BusJobResponseUrgentMessage,
    BusJobStreamEndMessage,
    BusJobUpdateMessage,
    BusJobUpdateUrgentMessage,
    BusMessage,
)
from pipecat.bus.ui.messages import (
    _UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME,
    BusUIEventMessage,
    BusUIJobCompletedMessage,
    BusUIJobGroupCompletedMessage,
    BusUIJobGroupStartedMessage,
    BusUIJobUpdateMessage,
)
from pipecat.pipeline.job_context import (
    JobContext,
    JobGroup,
    JobGroupContext,
    JobStatus,
)
from pipecat.workers.base_worker import BaseWorker


@dataclass
class UIJobGroupOptions:
    """Client-visibility options for a job or job group.

    Passing one to ``BaseUIWorker.job(...)``, ``job_group(...)``,
    ``request_job(...)``, or ``request_job_group(...)`` surfaces the work's
    lifecycle -- start, per-worker progress, and completion -- to the UI
    client as ``ui-job-group`` envelopes, so the client can show a
    (optionally cancellable) progress card. Workers need not know about the
    UI surface: any ``send_job_update`` they emit against the group's
    ``job_id`` is forwarded automatically.

    Parameters:
        label: Optional human-readable label surfaced to the client
            (e.g. ``"Research: Radiohead"``). The client UI uses it to
            title the in-flight job-group card.
        cancellable: Whether the client may request cancellation of the
            group via the reserved ``__cancel_job_group`` event.
            Defaults to True.
    """

    label: str | None = None
    cancellable: bool = True


@dataclass
class _UIJobGroupRegistration:
    """Per-group metadata a worker keeps for each in-flight user job group.

    Consulted by ``on_bus_message`` to decide which bus job messages to forward
    to the client and whether a ``__cancel_job_group`` event should be honored.

    Parameters:
        worker_names: Names of the workers the group was dispatched to.
        label: Optional human-readable label shown on the client job-group card.
        cancellable: Whether the client may cancel the group via ``__cancel_job_group``.
        reported: Workers that have already had a terminal ``job_completed``
            envelope forwarded to the client; consulted at cancellation so
            the remaining workers get a synthesized ``cancelled`` envelope.
    """

    worker_names: list[str]
    label: str | None
    cancellable: bool
    reported: set[str] = field(default_factory=set)


class BaseUIWorker(BaseWorker):
    """Worker that can surface its jobs and job groups on the client UI.

    Overrides the ``BaseWorker`` dispatch methods with an optional ``ui``
    parameter. When set, the group is registered for lifecycle forwarding:
    a ``group_started`` envelope is published at dispatch, worker updates
    and responses are forwarded as ``job_update`` / ``job_completed``
    envelopes, ``group_completed`` is published at group teardown (normal
    completion, cancellation, or timeout), and the client's reserved
    ``__cancel_job_group`` event is translated into ``cancel_job_group``
    for groups registered as cancellable. Without ``ui``, behavior is
    identical to ``BaseWorker``.

    Instantiable directly (no LLM): register one on the runner as a
    dispatcher when a pipeline app wants client-visible background work::

        ui_jobs = BaseUIWorker("ui-jobs")
        job_id = await ui_jobs.request_job_group(
            "wikipedia", "news",
            payload={"query": query},
            ui=UIJobGroupOptions(label=f"Research: {query}"),
        )
    """

    def __init__(self, *args, **kwargs):
        """Initialize the BaseUIWorker.

        Args:
            *args: Positional arguments forwarded to the next class in the
                MRO (ultimately ``BaseWorker``).
            **kwargs: Keyword arguments forwarded to the next class in the
                MRO (ultimately ``BaseWorker``).
        """
        super().__init__(*args, **kwargs)
        # Registry of in-flight user-facing job groups dispatched by this
        # worker. Keyed by ``job_id``; ``on_bus_message`` consults it to
        # decide which job update / response messages to forward to the
        # client as ``ui-job-group`` envelopes.
        self._ui_job_groups: dict[str, _UIJobGroupRegistration] = {}

    def job(
        self,
        worker_name: str,
        *,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        ui: UIJobGroupOptions | None = None,
    ) -> JobContext:
        """Create a single-worker job context manager.

        Like ``BaseWorker.job(...)``, plus optional client visibility.

        Args:
            worker_name: Name of the worker to send the job to.
            name: Optional job name for routing to a named ``@job``
                handler on the worker.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds.
            ui: Optional client-visibility options; when set, the job's
                lifecycle streams to the UI client as ``ui-job-group``
                envelopes (see ``UIJobGroupOptions``).

        Returns:
            A ``JobContext`` to use with ``async with``.
        """
        return JobContext(
            self,
            worker_name,
            name=name,
            payload=payload,
            timeout=timeout,
            ui=ui,
        )

    def job_group(
        self,
        *worker_names: str,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
        ui: UIJobGroupOptions | None = None,
    ) -> JobGroupContext:
        """Create a job group context manager.

        Like ``BaseWorker.job_group(...)``, plus optional client visibility.

        Args:
            *worker_names: Names of the workers to send the job to.
            name: Optional job name for routing to named ``@job``
                handlers on the workers.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds.
            cancel_on_error: Whether to cancel the group if a worker
                errors. Defaults to True.
            ui: Optional client-visibility options; when set, the group's
                lifecycle streams to the UI client as ``ui-job-group``
                envelopes (see ``UIJobGroupOptions``).

        Returns:
            A ``JobGroupContext`` to use with ``async with``.
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
            ui=ui,
        )

    async def request_job(
        self,
        worker_name: str,
        *,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        ui: UIJobGroupOptions | None = None,
    ) -> str:
        """Send a job request to a single worker (fire-and-forget).

        Like ``BaseWorker.request_job(...)``, plus optional client visibility.

        Args:
            worker_name: Name of the worker to send the job to.
            name: Optional job name for routing to a named ``@job``
                handler on the worker.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds. If set, the job is
                automatically cancelled after this duration.
            ui: Optional client-visibility options; when set, the job's
                lifecycle streams to the UI client as ``ui-job-group``
                envelopes (see ``UIJobGroupOptions``).

        Returns:
            The generated job_id.
        """
        group = await self.create_job_group_and_request_job(
            [worker_name],
            name=name,
            payload=payload,
            timeout=timeout,
            cancel_on_error=True,
            ui=ui,
        )
        return group.job_id

    async def request_job_group(
        self,
        *worker_names: str,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
        ui: UIJobGroupOptions | None = None,
    ) -> str:
        """Send a job request to multiple workers (fire-and-forget).

        Like ``BaseWorker.request_job_group(...)``, plus optional client
        visibility.

        Args:
            *worker_names: Names of the workers to send the job to.
            name: Optional job name for routing to named ``@job``
                handlers on the workers.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds. If set, the job is
                automatically cancelled after this duration.
            cancel_on_error: Whether to cancel the entire group if a
                worker responds with an error status. Defaults to True.
            ui: Optional client-visibility options; when set, the group's
                lifecycle streams to the UI client as ``ui-job-group``
                envelopes (see ``UIJobGroupOptions``).

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
            ui=ui,
        )
        return group.job_id

    async def create_job_group_and_request_job(
        self,
        worker_names: list[str],
        *,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
        ui: UIJobGroupOptions | None = None,
    ) -> JobGroup:
        """Create a job group and send requests, optionally client-visible.

        Like ``BaseWorker.create_job_group_and_request_job(...)``; when
        ``ui`` is set, the group is additionally registered for lifecycle
        forwarding and a ``group_started`` envelope is published before
        this method returns. The matching ``group_completed`` is published
        when the group completes, is cancelled, or times out.

        Args:
            worker_names: Names of the workers to send the job to.
            name: Optional job name for routing to named handlers.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds. Covers both the
                ready-wait and job execution.
            cancel_on_error: Whether to cancel the group if a worker
                errors. Defaults to True.
            ui: Optional client-visibility options (see
                ``UIJobGroupOptions``).

        Returns:
            The created ``JobGroup``.
        """
        group = await super().create_job_group_and_request_job(
            worker_names,
            name=name,
            payload=payload,
            timeout=timeout,
            cancel_on_error=cancel_on_error,
        )
        if ui:
            self._register_ui_job_group(
                job_id=group.job_id,
                worker_names=list(worker_names),
                label=ui.label,
                cancellable=ui.cancellable,
            )
            await self.send_bus_message(
                BusUIJobGroupStartedMessage(
                    source=self.name,
                    target=None,
                    job_id=group.job_id,
                    workers=list(worker_names),
                    label=ui.label,
                    cancellable=ui.cancellable,
                    at=int(time.time() * 1000),
                )
            )
        return group

    async def cancel_job_group(self, job_id: str, *, reason: str | None = None) -> None:
        """Cancel a running job group and complete its client card, if any.

        Args:
            job_id: The job identifier to cancel.
            reason: Optional human-readable reason for cancellation.
        """
        await super().cancel_job_group(job_id, reason=reason)
        # The workers' own CANCELLED responses arrive after the registration
        # is gone below, so synthesize the terminal envelope for every worker
        # that hasn't already reported — deterministically, instead of racing
        # the round trip. Workers that already completed or errored keep the
        # status the client saw.
        registration = self._ui_job_groups.get(job_id)
        if registration:
            for worker_name in registration.worker_names:
                if worker_name in registration.reported:
                    continue
                await self.send_bus_message(
                    BusUIJobCompletedMessage(
                        source=self.name,
                        target=None,
                        job_id=job_id,
                        worker_name=worker_name,
                        status=str(JobStatus.CANCELLED),
                        response=None,
                        at=int(time.time() * 1000),
                    )
                )
        await self._maybe_complete_ui_job_group(job_id)

    async def on_bus_message(self, message: BusMessage) -> None:
        """Forward registered job-group lifecycle to the client.

        Runs after base lifecycle handling. Worker updates and responses
        for registered groups are forwarded as ``ui-job-group`` envelopes;
        a response that tears the group down also publishes
        ``group_completed``. The client's reserved ``__cancel_job_group``
        event is honored for registered, cancellable groups.

        Args:
            message: The ``BusMessage`` to process.
        """
        # Forward a worker's terminal BEFORE base handling: on error
        # statuses (with ``cancel_on_error``) the base handler cancels the
        # group inside ``super()``, and the cancel override below completes
        # the card — the client must first learn which worker failed. A
        # stream end is a worker's successful terminal, reported like a
        # completed response carrying the final stream data.
        if isinstance(message, (BusJobResponseMessage, BusJobResponseUrgentMessage)):
            await self._maybe_forward_job_completed(message)
        elif isinstance(message, BusJobStreamEndMessage):
            await self._maybe_forward_stream_end_completed(message)

        await super().on_bus_message(message)

        if isinstance(message, (BusJobUpdateMessage, BusJobUpdateUrgentMessage)):
            await self._maybe_forward_job_update(message)
        elif isinstance(
            message,
            (BusJobResponseMessage, BusJobResponseUrgentMessage, BusJobStreamEndMessage),
        ):
            # A response and a stream end are both terminal paths that can
            # tear the group down inside ``super()``; once the group is
            # gone, complete the card (after any per-worker envelope,
            # preserving client ordering).
            if message.job_id not in self._job_groups:
                await self._maybe_complete_ui_job_group(message.job_id)
        elif (
            isinstance(message, BusUIEventMessage)
            and message.event_name == _UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME
        ):
            await self._handle_cancel_job_event(message)

    def _register_ui_job_group(
        self,
        *,
        job_id: str,
        worker_names: list[str],
        label: str | None,
        cancellable: bool,
    ) -> None:
        """Register an in-flight user job group for lifecycle forwarding.

        Subsequent ``BusJobUpdateMessage`` / ``BusJobResponseMessage`` whose
        ``job_id`` matches this entry will be forwarded to the client.
        """
        if job_id in self._ui_job_groups:
            logger.warning(
                f"Worker '{self.name}': user job group {job_id} already registered; overwriting"
            )
        self._ui_job_groups[job_id] = _UIJobGroupRegistration(
            worker_names=list(worker_names),
            label=label,
            cancellable=cancellable,
        )

    async def _maybe_complete_ui_job_group(self, job_id: str) -> None:
        """Publish ``group_completed`` and unregister, once, at group teardown.

        Called from every group teardown path -- normal completion and
        ``cancel_job_group`` (which the timeout and context-error paths
        also route through). Popping the registration makes it idempotent,
        and late-arriving updates or responses for the group are no longer
        forwarded.
        """
        if self._ui_job_groups.pop(job_id, None) is None:
            return
        await self.send_bus_message(
            BusUIJobGroupCompletedMessage(
                source=self.name,
                target=None,
                job_id=job_id,
                at=int(time.time() * 1000),
            )
        )

    async def _maybe_forward_job_update(
        self, message: BusJobUpdateMessage | BusJobUpdateUrgentMessage
    ) -> None:
        """Forward a worker update for a registered user job group.

        No-op if the message's ``job_id`` is not registered.
        """
        if message.job_id not in self._ui_job_groups:
            return
        await self.send_bus_message(
            BusUIJobUpdateMessage(
                source=self.name,
                target=None,
                job_id=message.job_id,
                worker_name=message.source,
                data=message.update,
                at=int(time.time() * 1000),
            )
        )

    async def _maybe_forward_job_completed(
        self, message: BusJobResponseMessage | BusJobResponseUrgentMessage
    ) -> None:
        """Forward a worker response for a registered user job group.

        No-op if the message's ``job_id`` is not registered.
        """
        registration = self._ui_job_groups.get(message.job_id)
        if registration is None:
            return
        registration.reported.add(message.source)
        await self.send_bus_message(
            BusUIJobCompletedMessage(
                source=self.name,
                target=None,
                job_id=message.job_id,
                worker_name=message.source,
                status=str(message.status),
                response=message.response,
                at=int(time.time() * 1000),
            )
        )

    async def _maybe_forward_stream_end_completed(self, message: BusJobStreamEndMessage) -> None:
        """Forward a worker's stream end as its terminal ``job_completed``.

        A worker may finish via ``send_job_stream_end`` instead of a
        response; the client is told it completed, with the final stream
        data as the response payload. No-op if the message's ``job_id`` is
        not registered.
        """
        registration = self._ui_job_groups.get(message.job_id)
        if registration is None:
            return
        registration.reported.add(message.source)
        await self.send_bus_message(
            BusUIJobCompletedMessage(
                source=self.name,
                target=None,
                job_id=message.job_id,
                worker_name=message.source,
                status=str(JobStatus.COMPLETED),
                response=message.data,
                at=int(time.time() * 1000),
            )
        )

    async def _handle_cancel_job_event(self, message: BusUIEventMessage) -> None:
        """Translate a client ``__cancel_job_group`` event into ``cancel_job_group``.

        Looks up the registered group and calls
        ``cancel_job_group(job_id, reason)``. Ignores the request
        silently if the group is unknown or was registered with
        ``cancellable=False``.
        """
        payload = message.payload if isinstance(message.payload, dict) else {}
        job_id = payload.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            logger.warning(
                f"Worker '{self.name}': received {_UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME} "
                "with no job_id; ignoring"
            )
            return
        registration = self._ui_job_groups.get(job_id)
        if registration is None:
            logger.debug(
                f"Worker '{self.name}': {_UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME} for "
                f"unknown job_id {job_id}; ignoring"
            )
            return
        if not registration.cancellable:
            logger.debug(
                f"Worker '{self.name}': {_UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME} for "
                f"non-cancellable group {job_id}; ignoring"
            )
            return
        reason = payload.get("reason")
        if reason is not None and not isinstance(reason, str):
            reason = None
        await self.cancel_job_group(job_id, reason=reason or "cancelled by user")
