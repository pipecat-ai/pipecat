#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agent bus package -- pub/sub messaging between agents and the runner.

Provides the pub/sub infrastructure that connects agents to each other and to
the runner. Key components:

- `TaskBus` -- abstract base class defining the send/receive interface.
- `AsyncQueueBus` -- in-process implementation backed by ``asyncio.Queue``.
- `BusBridgeProcessor` -- bidirectional mid-pipeline bridge for
  transport/session agents that exchanges frames with other agents
  through the bus.
- `BusMessage` and its subclasses -- the typed message hierarchy used for
  agent lifecycle events (activation, cancellation, shutdown), task
  coordination, and frame transport.
"""

from pipecat.bus.bridge_processor import BusBridgeProcessor
from pipecat.bus.bus import TaskBus
from pipecat.bus.local import AsyncQueueBus
from pipecat.bus.messages import (
    BusActivateTaskMessage,
    BusAddTaskMessage,
    BusCancelMessage,
    BusCancelTaskMessage,
    BusDataMessage,
    BusDeactivateTaskMessage,
    BusEndMessage,
    BusEndTaskMessage,
    BusFrameMessage,
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
    BusLocalMessage,
    BusMessage,
    BusSystemMessage,
    BusTaskErrorMessage,
    BusTaskLocalErrorMessage,
    BusTaskReadyMessage,
    BusTaskRegistryMessage,
)
from pipecat.bus.subscriber import BusSubscriber
from pipecat.registry.types import TaskRegistryEntry

__all__ = [
    "TaskBus",
    "AsyncQueueBus",
    "BusActivateTaskMessage",
    "BusAddTaskMessage",
    "BusTaskErrorMessage",
    "BusTaskLocalErrorMessage",
    "TaskRegistryEntry",
    "BusTaskReadyMessage",
    "BusTaskRegistryMessage",
    "BusBridgeProcessor",
    "BusCancelTaskMessage",
    "BusCancelMessage",
    "BusDeactivateTaskMessage",
    "BusEndTaskMessage",
    "BusEndMessage",
    "BusFrameMessage",
    "BusDataMessage",
    "BusLocalMessage",
    "BusMessage",
    "BusSubscriber",
    "BusSystemMessage",
    "BusJobCancelMessage",
    "BusJobRequestMessage",
    "BusJobResponseMessage",
    "BusJobResponseUrgentMessage",
    "BusJobStreamDataMessage",
    "BusJobStreamEndMessage",
    "BusJobStreamStartMessage",
    "BusJobUpdateMessage",
    "BusJobUpdateRequestMessage",
    "BusJobUpdateUrgentMessage",
]
