#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bus message types for inter-task communication.

Defines the message hierarchy used by the `TaskBus` for pub/sub messaging
between tasks, the session, and the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pipecat.frames.frames import DataFrame, Frame, SystemFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.registry.types import TaskRegistryEntry

if TYPE_CHECKING:
    from pipecat.pipeline.base_task import BaseTask
    from pipecat.pipeline.job_context import JobStatus

# ---------------------------------------------------------------------------
# Base types and mixins
# ---------------------------------------------------------------------------


class BusMessage:
    """Mixin carrying source/target metadata for bus messages.

    Not a frame itself. Combined with `DataFrame` or `SystemFrame`
    to create concrete message types with appropriate priority.
    """

    source: str
    target: str | None = None

    def __str__(self):
        return f"{type(self).__name__} (source={self.source}, target={self.target})"


class BusLocalMessage:
    """Mixin: message stays on the local bus, never forwarded to remote buses."""

    pass


@dataclass(kw_only=True)
class BusDataMessage(BusMessage, DataFrame):
    """Normal-priority bus message.

    Parameters:
        source: Name of the task or component that sent this message.
        target: Name of the intended recipient task, or None for broadcast.
    """

    source: str
    target: str | None = None


@dataclass(kw_only=True)
class BusSystemMessage(BusMessage, SystemFrame):
    """High-priority bus message that preempts normal messages in subscriber queues.

    Parameters:
        source: Name of the task or component that sent this message.
        target: Name of the intended recipient task, or None for broadcast.
    """

    source: str
    target: str | None = None


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
# Task lifecycle
# ---------------------------------------------------------------------------


@dataclass
class BusActivateTaskMessage(BusDataMessage):
    """Tells a targeted task to become active and start processing.

    Parameters:
        args: Optional activation arguments forwarded to `on_activated`.
    """

    args: dict | None = None


@dataclass
class BusDeactivateTaskMessage(BusDataMessage):
    """Tells a targeted task to become inactive and stop processing."""

    pass


@dataclass
class BusEndMessage(BusDataMessage):
    """Request a graceful end of the session.

    Sent by a task to the runner, which responds by sending
    `BusEndTaskMessage` to each task.

    Parameters:
        reason: Optional human-readable reason for ending.
    """

    reason: str | None = None


@dataclass
class BusEndTaskMessage(BusDataMessage):
    """Tells a targeted task to end its pipeline gracefully.

    Sent by the runner to individual tasks during shutdown.

    Parameters:
        reason: Optional human-readable reason for ending.
    """

    reason: str | None = None


@dataclass
class BusCancelMessage(BusSystemMessage):
    """Request a hard cancel of the session.

    Sent by a task to the runner, which responds by sending
    `BusCancelTaskMessage` to each task.

    Parameters:
        reason: Optional human-readable reason for the cancellation.
    """

    reason: str | None = None


@dataclass
class BusCancelTaskMessage(BusSystemMessage):
    """Tells a targeted task to cancel its pipeline.

    Sent by the runner to individual tasks during cancellation.

    Parameters:
        reason: Optional human-readable reason for the cancellation.
    """

    reason: str | None = None


# ---------------------------------------------------------------------------
# Task registry and errors
# ---------------------------------------------------------------------------


@dataclass
class BusAddTaskMessage(BusSystemMessage, BusLocalMessage):
    """Request to add a task to the local runner.

    Local-only: carries an in-memory task reference that cannot be
    serialized over the network.

    Parameters:
        task: The task instance to add.
    """

    task: BaseTask


@dataclass
class BusTaskRegistryMessage(BusSystemMessage):
    """Snapshot of tasks managed by a runner.

    Sent by the runner on startup and when new runners connect,
    so that remote runners can discover each other's tasks.

    Parameters:
        runner: Name of the runner that owns these tasks.
        tasks: List of task entries with their state.
    """

    runner: str
    tasks: list[TaskRegistryEntry]


@dataclass
class BusTaskReadyMessage(BusDataMessage):
    """Announces that a task is ready.

    Sent when any task (root or child) becomes ready. Carries the
    task's parent name so observers can reconstruct the full hierarchy.

    Parameters:
        runner: Name of the runner managing this task.
        parent: Name of the parent task, or None for root tasks.
        active: Whether the task started active.
        bridged: Whether the task is bridged (receives pipeline frames
            from the bus).
        started_at: Unix timestamp when the task became ready.
    """

    runner: str
    parent: str | None = None
    active: bool = False
    bridged: bool = False
    started_at: float | None = None


@dataclass
class BusTaskErrorMessage(BusSystemMessage):
    """Reports an error from a root task.

    Sent over the network so remote tasks can react. For child task
    errors, see `BusTaskLocalErrorMessage`.

    Parameters:
        error: Description of the error.
    """

    error: str


@dataclass
class BusTaskLocalErrorMessage(BusSystemMessage, BusLocalMessage):
    """Reports an error from a child task to its parent.

    Local-only: never crosses the network. The parent receives it
    via `on_task_failed()`.

    Parameters:
        error: Description of the error.
    """

    error: str


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@dataclass
class BusJobRequestMessage(BusDataMessage):
    """Requests a worker task to start work.

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
    """Response from a worker task when its job completes.

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
    """Progress update from a worker task.

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
    """Request a progress update from a worker task.

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
