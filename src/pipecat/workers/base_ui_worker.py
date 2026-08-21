#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""BaseUIWorker: a worker whose jobs and job groups surface on the client UI.

Every group a ``BaseUIWorker`` dispatches streams its lifecycle -- start,
per-worker progress, and completion -- to the UI client as ``ui-job-group``
envelopes, and the client's reserved ``__cancel_job_group`` event is honored
for groups dispatched as cancellable. ``JobGroupParams.label`` titles the
client's progress card. Dispatch from a plain ``BaseWorker`` instead when the
work should stay invisible.

No LLM is involved: ``BaseUIWorker`` is instantiable as-is (its inherited
``run()`` is a bus-only loop), so a plain pipeline app can register one on the
runner as a dispatcher and call it from tools. ``UIWorker`` inherits this class
and adds the LLM-driven page interaction (snapshots, UI events, commands).
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
    JobGroup,
    JobGroupParams,
    JobStatus,
    resolve_job_params,
)
from pipecat.workers.base_worker import BaseWorker


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
    """Worker that surfaces its jobs and job groups on the client UI.

    Every group this worker dispatches is registered for lifecycle
    forwarding: a ``group_started`` envelope is published at dispatch,
    worker updates and responses are forwarded as ``job_update`` /
    ``job_completed`` envelopes, ``group_completed`` is published at group
    teardown (normal completion, cancellation, or timeout), and the
    client's reserved ``__cancel_job_group`` event is translated into
    ``cancel_job_group`` for groups dispatched as cancellable.

    Instantiable directly (no LLM): register one on the runner as a
    dispatcher when a pipeline app wants client-visible background work::

        ui_jobs = BaseUIWorker("ui-jobs")
        job_id = await ui_jobs.request_job_group(
            "wikipedia", "news",
            params=JobGroupParams(
                payload={"query": query},
                label=f"Research: {query}",
            ),
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

    async def create_job_group_and_request_job(
        self,
        worker_names: list[str],
        *,
        params: JobGroupParams | None = None,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool | None = None,
    ) -> JobGroup:
        """Dispatch a job group and announce it to the client.

        Like ``BaseWorker.create_job_group_and_request_job(...)``, and then
        registers the group for lifecycle forwarding and publishes
        ``group_started``. The matching ``group_completed`` follows when the
        group completes, is cancelled, or times out.

        Args:
            worker_names: Names of the workers to send the job to.
            params: How to run the group. See :class:`JobGroupParams`.
            name: Job name.

                .. deprecated:: 1.8.0
                    Use ``params=JobGroupParams(name=...)`` instead. Will be
                    removed in 2.0.0.
            payload: Structured data describing the work.

                .. deprecated:: 1.8.0
                    Use ``params=JobGroupParams(payload=...)`` instead. Will
                    be removed in 2.0.0.
            timeout: Timeout in seconds.

                .. deprecated:: 1.8.0
                    Use ``params=JobGroupParams(timeout=...)`` instead. Will
                    be removed in 2.0.0.
            cancel_on_error: Whether a worker error cancels the group.

                .. deprecated:: 1.8.0
                    Use ``params=JobGroupParams(cancel_on_error=...)``
                    instead. Will be removed in 2.0.0.

        Returns:
            The created ``JobGroup``.
        """
        group_params = resolve_job_params(
            params,
            JobGroupParams,
            name=name,
            payload=payload,
            timeout=timeout,
            cancel_on_error=cancel_on_error,
        )
        group = await super().create_job_group_and_request_job(worker_names, params=group_params)
        self._register_ui_job_group(
            job_id=group.job_id,
            worker_names=list(worker_names),
            label=group_params.label,
            cancellable=group_params.cancellable,
        )
        await self.send_bus_message(
            BusUIJobGroupStartedMessage(
                source=self.name,
                target=None,
                job_id=group.job_id,
                workers=list(worker_names),
                label=group_params.label,
                cancellable=group_params.cancellable,
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
