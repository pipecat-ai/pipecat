#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Worker bus package -- pub/sub messaging between workers and the runner.

Provides the pub/sub infrastructure that connects workers to each other and to
the runner. Key components:

- `WorkerBus` -- abstract base class defining the send/receive interface.
- `AsyncQueueBus` -- in-process implementation backed by ``asyncio.Queue``.
- `BusBridgeProcessor` -- bidirectional mid-pipeline bridge for
  transport/session workers that exchanges frames with other workers
  through the bus.
- `BusMessage` and its subclasses -- the typed message hierarchy used for
  worker lifecycle events (activation, cancellation, shutdown), job
  coordination, and frame transport.
"""

from pipecat.bus.bridge_processor import BusBridgeProcessor
from pipecat.bus.bus import WorkerBus
from pipecat.bus.local import AsyncQueueBus
from pipecat.bus.messages import (
    BusActivateWorkerMessage,
    BusAddWorkerMessage,
    BusCancelMessage,
    BusCancelWorkerMessage,
    BusDataMessage,
    BusDeactivateWorkerMessage,
    BusEndMessage,
    BusEndWorkerMessage,
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
    BusTTSSpeakMessage,
    BusWorkerErrorMessage,
    BusWorkerLocalErrorMessage,
    BusWorkerReadyMessage,
    BusWorkerRegistryMessage,
)
from pipecat.bus.subscriber import BusSubscriber
from pipecat.registry.types import WorkerRegistryEntry

__all__ = [
    "WorkerBus",
    "AsyncQueueBus",
    "BusActivateWorkerMessage",
    "BusAddWorkerMessage",
    "BusWorkerErrorMessage",
    "BusWorkerLocalErrorMessage",
    "WorkerRegistryEntry",
    "BusWorkerReadyMessage",
    "BusWorkerRegistryMessage",
    "BusBridgeProcessor",
    "BusCancelWorkerMessage",
    "BusCancelMessage",
    "BusDeactivateWorkerMessage",
    "BusEndWorkerMessage",
    "BusEndMessage",
    "BusFrameMessage",
    "BusDataMessage",
    "BusLocalMessage",
    "BusMessage",
    "BusSubscriber",
    "BusSystemMessage",
    "BusTTSSpeakMessage",
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
