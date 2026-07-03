#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bus message types for inter-worker communication.

Defines the message hierarchy used by the `WorkerBus` for pub/sub messaging
between workers and the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.registry.types import WorkerRegistryEntry

if TYPE_CHECKING:
    from pipecat.pipeline.job_context import JobStatus
    from pipecat.workers.base_worker import BaseWorker

# ---------------------------------------------------------------------------
# Base types and mixins
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class BusMessage:
    """Base class for messages carried by the `WorkerBus`.

    Bus messages are independent of pipeline `Frame`s — if a worker needs
    to ship a frame between pipelines it wraps it in a `BusFrameMessage`.
    Subclasses choose delivery priority by extending :class:`BusDataMessage`
    (normal priority, FIFO) or :class:`BusSystemMessage` (high priority,
    delivered ahead of queued data messages).

    Parameters:
        source: Name of the worker or component that sent this message.
        target: Name of the intended recipient worker, or None for broadcast.
    """

    source: str
    target: str | None = None

    def __str__(self):
        return f"{type(self).__name__} (source={self.source}, target={self.target})"


class BusLocalMessage:
    """Mixin: message stays on the local bus, never forwarded to remote buses."""

    pass


@dataclass(kw_only=True)
class BusDataMessage(BusMessage):
    """Normal-priority bus message.

    Delivered in FIFO order on the subscriber's data queue.
    """

    pass


@dataclass(kw_only=True)
class BusSystemMessage(BusMessage):
    """High-priority bus message.

    Delivered ahead of any queued :class:`BusDataMessage` on the
    subscriber's priority queue.
    """

    pass


# ---------------------------------------------------------------------------
# Frame transport
# ---------------------------------------------------------------------------


@dataclass
class BusFrameMessage(BusDataMessage):
    """Wraps a Pipecat `Frame` for transport over the bus.

    Parameters:
        frame: The Pipecat frame to transport.
        direction: Direction the frame should travel in the recipient's pipeline.
        bridge: Optional bridge name for routing in multi-bridge setups.
    """

    frame: Frame
    direction: FrameDirection
    bridge: str | None = None


# ---------------------------------------------------------------------------
# Pipeline commands
# ---------------------------------------------------------------------------


@dataclass
class BusTTSSpeakMessage(BusDataMessage):
    """Asks a `PipelineWorker` to speak the given text via its TTS service.

    On receipt, the worker queues a `TTSSpeakFrame` into its pipeline.
    Pipelines without a TTS service let the frame flow through harmlessly.

    Parameters:
        text: The text to be spoken.
        append_to_context: Whether the spoken text should also be appended
            to the conversation context (forwarded to `TTSSpeakFrame`).
            Defaults to True, matching `TTSSpeakFrame.append_to_context`.
    """

    text: str
    append_to_context: bool = True


# ---------------------------------------------------------------------------
# Worker lifecycle
# ---------------------------------------------------------------------------


@dataclass
class BusActivateWorkerMessage(BusDataMessage):
    """Tells a targeted worker to become active and start processing.

    Parameters:
        args: Optional activation arguments forwarded to `on_activated`.
    """

    args: dict | None = None


@dataclass
class BusDeactivateWorkerMessage(BusDataMessage):
    """Tells a targeted worker to become inactive and stop processing."""

    pass


@dataclass
class BusEndMessage(BusDataMessage):
    """Request a graceful end of the session.

    Sent by a worker to the runner, which responds by sending
    `BusEndWorkerMessage` to each worker.

    Parameters:
        reason: Optional human-readable reason for ending.
    """

    reason: str | None = None


@dataclass
class BusEndWorkerMessage(BusDataMessage):
    """Tells a targeted worker to end its pipeline gracefully.

    Sent by the runner to individual workers during shutdown.

    Parameters:
        reason: Optional human-readable reason for ending.
    """

    reason: str | None = None


@dataclass
class BusCancelMessage(BusSystemMessage):
    """Request a hard cancel of the session.

    Sent by a worker to the runner, which responds by sending
    `BusCancelWorkerMessage` to each worker.

    Parameters:
        reason: Optional human-readable reason for the cancellation.
    """

    reason: str | None = None


@dataclass
class BusCancelWorkerMessage(BusSystemMessage):
    """Tells a targeted worker to cancel its pipeline.

    Sent by the runner to individual workers during cancellation.

    Parameters:
        reason: Optional human-readable reason for the cancellation.
    """

    reason: str | None = None


# ---------------------------------------------------------------------------
# Worker registry and errors
# ---------------------------------------------------------------------------


@dataclass
class BusAddWorkerMessage(BusSystemMessage, BusLocalMessage):
    """Request to add a worker to the local runner.

    Local-only: carries an in-memory worker reference that cannot be
    serialized over the network.

    Parameters:
        worker: The worker instance to add.
    """

    worker: BaseWorker


@dataclass
class BusWorkerRegistryMessage(BusSystemMessage):
    """Snapshot of workers managed by a runner.

    Sent by the runner on startup and when new runners connect,
    so that remote runners can discover each other's workers.

    Parameters:
        runner: Name of the runner that owns these workers.
        workers: List of worker entries with their state.
    """

    runner: str
    workers: list[WorkerRegistryEntry]


@dataclass
class BusWorkerReadyMessage(BusDataMessage):
    """Announces that a worker is ready.

    Sent when any worker (root or child) becomes ready. Carries the
    worker's parent name so observers can reconstruct the full hierarchy.

    Parameters:
        runner: Name of the runner managing this worker.
        parent: Name of the parent worker, or None for root workers.
        active: Whether the worker started active.
        bridged: Whether the worker is bridged (receives pipeline frames
            from the bus).
        started_at: Unix timestamp when the worker became ready.
    """

    runner: str
    parent: str | None = None
    active: bool = False
    bridged: bool = False
    started_at: float | None = None


@dataclass
class BusWorkerErrorMessage(BusSystemMessage):
    """Reports an error from a root worker.

    Sent over the network so remote workers can react. For child worker
    errors, see `BusWorkerLocalErrorMessage`.

    Parameters:
        error: Description of the error.
    """

    error: str


@dataclass
class BusWorkerLocalErrorMessage(BusSystemMessage, BusLocalMessage):
    """Reports an error from a child worker to its parent.

    Local-only: never crosses the network. The parent receives it
    via `on_worker_failed()`.

    Parameters:
        error: Description of the error.
    """

    error: str


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@dataclass
class BusJobRequestMessage(BusDataMessage):
    """Requests a worker worker to start work.

    Parameters:
        job_id: Unique identifier for this job.
        job_name: Optional job name for routing to named `@job` handlers.
        payload: Optional structured data describing the work.
    """

    job_id: str
    job_name: str | None = None
    payload: dict | None = None


@dataclass
class BusJobResponseMessage(BusDataMessage):
    """Response from a worker worker when its job completes.

    Parameters:
        job_id: The job identifier.
        status: Completion status.
        response: Optional result data.
    """

    job_id: str
    status: JobStatus
    response: dict | None = None


@dataclass
class BusJobResponseUrgentMessage(BusSystemMessage):
    """High-priority job response.

    Same semantics as `BusJobResponseMessage` but delivered with
    system priority, preempting queued data messages.

    Parameters:
        job_id: The job identifier.
        status: Completion status.
        response: Optional result data.
    """

    job_id: str
    status: JobStatus
    response: dict | None = None


@dataclass
class BusJobUpdateMessage(BusDataMessage):
    """Progress update from a worker worker.

    Parameters:
        job_id: The job identifier.
        update: Optional progress data.
    """

    job_id: str
    update: dict | None = None


@dataclass
class BusJobUpdateUrgentMessage(BusSystemMessage):
    """High-priority job progress update.

    Same semantics as `BusJobUpdateMessage` but delivered with
    system priority, preempting queued data messages.

    Parameters:
        job_id: The job identifier.
        update: Optional progress data.
    """

    job_id: str
    update: dict | None = None


@dataclass
class BusJobUpdateRequestMessage(BusDataMessage):
    """Request a progress update from a worker worker.

    Parameters:
        job_id: The job identifier.
    """

    job_id: str


@dataclass
class BusJobCancelMessage(BusSystemMessage):
    """Cancel a running job.

    Parameters:
        job_id: The job identifier.
        reason: Optional human-readable reason for cancellation.
    """

    job_id: str
    reason: str | None = None


# ---------------------------------------------------------------------------
# Job streaming
# ---------------------------------------------------------------------------


@dataclass
class BusJobStreamStartMessage(BusDataMessage):
    """Signals the start of a streaming job response.

    Parameters:
        job_id: The job identifier.
        data: Optional metadata (e.g. content type).
    """

    job_id: str
    data: dict | None = None


@dataclass
class BusJobStreamDataMessage(BusDataMessage):
    """A chunk of streaming job data.

    Parameters:
        job_id: The job identifier.
        data: The chunk payload.
    """

    job_id: str
    data: dict | None = None


@dataclass
class BusJobStreamEndMessage(BusDataMessage):
    """Signals the end of a streaming job response.

    Parameters:
        job_id: The job identifier.
        data: Optional final metadata.
    """

    job_id: str
    data: dict | None = None
