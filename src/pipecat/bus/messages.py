#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bus message types for inter-agent communication.

Defines the message hierarchy used by the `TaskBus` for pub/sub messaging
between agents, the session, and the runner.
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

    Not a frame itself. Combined with ``DataFrame`` or ``SystemFrame``
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
        source: Name of the agent or component that sent this message.
        target: Name of the intended recipient agent, or None for broadcast.
    """

    source: str
    target: str | None = None


@dataclass(kw_only=True)
class BusSystemMessage(BusMessage, SystemFrame):
    """High-priority bus message that preempts normal messages in subscriber queues.

    Parameters:
        source: Name of the agent or component that sent this message.
        target: Name of the intended recipient agent, or None for broadcast.
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
# Agent lifecycle
# ---------------------------------------------------------------------------


@dataclass
class BusActivateTaskMessage(BusDataMessage):
    """Tells a targeted agent to become active and start processing.

    Parameters:
        args: Optional activation arguments forwarded to ``on_activated``.
    """

    args: dict | None = None


@dataclass
class BusDeactivateTaskMessage(BusDataMessage):
    """Tells a targeted agent to become inactive and stop processing."""

    pass


@dataclass
class BusEndMessage(BusDataMessage):
    """Request a graceful end of the session.

    Sent by an agent to the runner, which responds by sending
    `BusEndTaskMessage` to each agent.

    Parameters:
        reason: Optional human-readable reason for ending.
    """

    reason: str | None = None


@dataclass
class BusEndTaskMessage(BusDataMessage):
    """Tells a targeted agent to end its pipeline gracefully.

    Sent by the runner to individual agents during shutdown.

    Parameters:
        reason: Optional human-readable reason for ending.
    """

    reason: str | None = None


@dataclass
class BusCancelMessage(BusSystemMessage):
    """Request a hard cancel of the session.

    Sent by an agent to the runner, which responds by sending
    `BusCancelTaskMessage` to each agent.

    Parameters:
        reason: Optional human-readable reason for the cancellation.
    """

    reason: str | None = None


@dataclass
class BusCancelTaskMessage(BusSystemMessage):
    """Tells a targeted agent to cancel its pipeline task.

    Sent by the runner to individual agents during cancellation.

    Parameters:
        reason: Optional human-readable reason for the cancellation.
    """

    reason: str | None = None


# ---------------------------------------------------------------------------
# Agent registry and errors
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
    """Announces that an agent is ready.

    Sent when any agent (root or child) becomes ready. Carries the
    agent's parent name so observers can reconstruct the full hierarchy.

    Parameters:
        runner: Name of the runner managing this agent.
        parent: Name of the parent agent, or None for root agents.
        active: Whether the agent started active.
        bridged: Whether the agent is bridged (receives pipeline frames
            from the bus).
        started_at: Unix timestamp when the agent became ready.
    """

    runner: str
    parent: str | None = None
    active: bool = False
    bridged: bool = False
    started_at: float | None = None


@dataclass
class BusTaskErrorMessage(BusSystemMessage):
    """Reports an error from a root agent.

    Sent over the network so remote agents can react. For child agent
    errors, see ``BusTaskLocalErrorMessage``.

    Parameters:
        error: Description of the error.
    """

    error: str


@dataclass
class BusTaskLocalErrorMessage(BusSystemMessage, BusLocalMessage):
    """Reports an error from a child agent to its parent.

    Local-only: never crosses the network. The parent receives it
    via ``on_task_failed()``.

    Parameters:
        error: Description of the error.
    """

    error: str


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@dataclass
class BusJobRequestMessage(BusDataMessage):
    """Requests a task agent to start work.

    Parameters:
        job_id: Unique identifier for this task.
        job_name: Optional task name for routing to named handlers.
        payload: Optional structured data describing the work.
    """

    job_id: str
    job_name: str | None = None
    payload: dict | None = None


@dataclass
class BusJobResponseMessage(BusDataMessage):
    """Response from a task agent when it completes.

    Parameters:
        job_id: The task identifier.
        status: Completion status.
        response: Optional result data.
    """

    job_id: str
    status: JobStatus
    response: dict | None = None


@dataclass
class BusJobResponseUrgentMessage(BusSystemMessage):
    """High-priority response from a task agent.

    Same semantics as ``BusJobResponseMessage`` but delivered with
    system priority, preempting queued data messages.

    Parameters:
        job_id: The task identifier.
        status: Completion status.
        response: Optional result data.
    """

    job_id: str
    status: JobStatus
    response: dict | None = None


@dataclass
class BusJobUpdateMessage(BusDataMessage):
    """Progress update from a task agent.

    Parameters:
        job_id: The task identifier.
        update: Optional progress data.
    """

    job_id: str
    update: dict | None = None


@dataclass
class BusJobUpdateUrgentMessage(BusSystemMessage):
    """High-priority progress update from a task agent.

    Same semantics as ``BusJobUpdateMessage`` but delivered with
    system priority, preempting queued data messages.

    Parameters:
        job_id: The task identifier.
        update: Optional progress data.
    """

    job_id: str
    update: dict | None = None


@dataclass
class BusJobUpdateRequestMessage(BusDataMessage):
    """Request a progress update from a task agent.

    Parameters:
        job_id: The task identifier.
    """

    job_id: str


@dataclass
class BusJobCancelMessage(BusSystemMessage):
    """Cancel a running task.

    Parameters:
        job_id: The task identifier.
        reason: Optional human-readable reason for cancellation.
    """

    job_id: str
    reason: str | None = None


# ---------------------------------------------------------------------------
# Task streaming
# ---------------------------------------------------------------------------


@dataclass
class BusJobStreamStartMessage(BusDataMessage):
    """Signals the start of a streaming task response.

    Parameters:
        job_id: The task identifier.
        data: Optional metadata (e.g. content type).
    """

    job_id: str
    data: dict | None = None


@dataclass
class BusJobStreamDataMessage(BusDataMessage):
    """A chunk of streaming task data.

    Parameters:
        job_id: The task identifier.
        data: The chunk payload.
    """

    job_id: str
    data: dict | None = None


@dataclass
class BusJobStreamEndMessage(BusDataMessage):
    """Signals the end of a streaming task response.

    Parameters:
        job_id: The task identifier.
        data: Optional final metadata.
    """

    job_id: str
    data: dict | None = None
