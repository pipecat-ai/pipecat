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
    JobGroupResponse,
    JobStatus,
)
from pipecat.workers.base_worker import BaseWorker


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

    async def create_job_group_and_request_job(self, worker_names: list[str], **kwargs) -> JobGroup:
        """Dispatch a job group and announce it to the client.

        Args:
            worker_names: Names of the workers to send the job to.
            **kwargs: Everything
                :meth:`~pipecat.workers.base_worker.BaseWorker.create_job_group_and_request_job`
                takes, forwarded unchanged.

        Returns:
            The created ``JobGroup``.
        """
        group = await super().create_job_group_and_request_job(worker_names, **kwargs)
        await self.send_bus_message(
            BusUIJobGroupStartedMessage(
                source=self.name,
                target=None,
                job_id=group.job_id,
                workers=list(group.worker_names),
                label=group.label,
                cancellable=group.cancellable,
                at=int(time.time() * 1000),
            )
        )
        return group

    async def cancel_job_group(self, job_id: str, *, reason: str | None = None) -> None:
        """Cancel a running job group and complete its client card.

        Args:
            job_id: The job identifier to cancel.
            reason: Optional human-readable reason for cancellation.
        """
        # Capture the group before ``super()`` tears it down: the client's
        # card is completed from it below.
        group = self._job_groups.get(job_id)
        await super().cancel_job_group(job_id, reason=reason)
        if group is None:
            return
        # The workers' own CANCELLED responses arrive after the group is
        # gone, so synthesize the terminal envelope for every worker the
        # cancellation actually cut short, deterministically instead of
        # racing the round trip. Workers that already finished keep the
        # status the client saw.
        for worker_name in group.worker_names:
            if worker_name in group.terminated:
                continue
            await self._send_job_completed(
                job_id=job_id,
                worker_name=worker_name,
                status=str(JobStatus.CANCELLED),
                response=None,
            )
        await self._send_group_completed(job_id)

    async def on_bus_message(self, message: BusMessage) -> None:
        """Handle the client's reserved ``__cancel_job_group`` event.

        Everything else this worker forwards to the client hangs off the
        job hooks (:meth:`on_job_update`, :meth:`on_job_response`,
        :meth:`on_job_stream_end`, :meth:`on_job_completed`), which the
        base class calls at the right point in a group's lifecycle.

        Args:
            message: The ``BusMessage`` to process.
        """
        await super().on_bus_message(message)
        if (
            isinstance(message, BusUIEventMessage)
            and message.event_name == _UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME
        ):
            await self._handle_cancel_job_event(message)

    async def on_job_update(self, message: BusJobUpdateMessage | BusJobUpdateUrgentMessage) -> None:
        """Forward a worker's progress update to the client."""
        await super().on_job_update(message)
        # A group torn down by a cancellation still has messages in flight
        # from its workers; the client's card is already closed.
        if message.job_id not in self._job_groups:
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

    async def on_job_response(
        self, message: BusJobResponseMessage | BusJobResponseUrgentMessage
    ) -> None:
        """Forward a worker's response to the client as its terminal envelope.

        Runs before the group is torn down, so on an error status (with
        ``cancel_on_error``) the client learns which worker failed before
        the card closes.
        """
        await super().on_job_response(message)
        # A cancelled group is already gone, and every worker it cut short
        # was reported at cancellation; their own CANCELLED responses land
        # here afterwards and would double up.
        if message.job_id not in self._job_groups:
            return
        await self._send_job_completed(
            job_id=message.job_id,
            worker_name=message.source,
            status=str(message.status),
            response=message.response,
        )

    async def on_job_stream_end(self, message: BusJobStreamEndMessage) -> None:
        """Forward a worker's stream end as its terminal envelope.

        A worker may finish by ending its stream instead of responding; the
        client is told it completed, with the final stream data as the
        response payload.
        """
        await super().on_job_stream_end(message)
        if message.job_id not in self._job_groups:
            return
        await self._send_job_completed(
            job_id=message.job_id,
            worker_name=message.source,
            status=str(JobStatus.COMPLETED),
            response=message.data,
        )

    async def on_job_completed(self, result: JobGroupResponse) -> None:
        """Complete the client's card for a group whose workers all finished."""
        await super().on_job_completed(result)
        await self._send_group_completed(result.job_id)

    async def _send_job_completed(
        self,
        *,
        job_id: str,
        worker_name: str,
        status: str,
        response: dict | None,
    ) -> None:
        """Publish one worker's terminal envelope."""
        await self.send_bus_message(
            BusUIJobCompletedMessage(
                source=self.name,
                target=None,
                job_id=job_id,
                worker_name=worker_name,
                status=status,
                response=response,
                at=int(time.time() * 1000),
            )
        )

    async def _send_group_completed(self, job_id: str) -> None:
        """Publish the envelope that closes the client's card.

        Reached once per group: a group either completes with every worker
        having finished, or is cancelled, and the two paths are exclusive.
        """
        await self.send_bus_message(
            BusUIJobGroupCompletedMessage(
                source=self.name,
                target=None,
                job_id=job_id,
                at=int(time.time() * 1000),
            )
        )

    async def _handle_cancel_job_event(self, message: BusUIEventMessage) -> None:
        """Translate a client ``__cancel_job_group`` event into a cancel request.

        Hands the request to
        :meth:`~pipecat.workers.base_worker.BaseWorker.request_cancel_job_group`,
        which refuses it for a group that is unknown or was dispatched as
        non-cancellable.
        """
        payload = message.payload if isinstance(message.payload, dict) else {}
        job_id = payload.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            logger.warning(
                f"Worker '{self.name}': received {_UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME} "
                "with no job_id; ignoring"
            )
            return
        reason = payload.get("reason")
        if reason is not None and not isinstance(reason, str):
            reason = None
        cancelled = await self.request_cancel_job_group(
            job_id, reason=reason or "cancelled by user"
        )
        if not cancelled:
            logger.debug(
                f"Worker '{self.name}': {_UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME} for "
                f"unknown or non-cancellable group {job_id}; ignoring"
            )
