#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bus bridge and edge processors for inter-worker frame routing.

Provides:

- `BusBridgeProcessor`: a mid-pipeline processor that exchanges frames
  with other workers through the bus, consuming local frames.
- `_BusEdgeProcessor`: a pipeline-edge processor used internally by
  `PipelineWorker` when ``bridged`` is set. Tees frames between the local
  pipeline and the bus (frames continue locally and are also forwarded
  to the bus).
"""

import warnings
from typing import TYPE_CHECKING

from pipecat.bus.bus import WorkerBus
from pipecat.bus.messages import BusFrameMessage, BusMessage
from pipecat.bus.subscriber import BusSubscriber
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    OutputTransportMessageUrgentFrame,
    PipelineFlushFrame,
    StartFrame,
    StopFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup

if TYPE_CHECKING:
    from pipecat.pipeline.worker import PipelineWorker

_LIFECYCLE_FRAMES = (StartFrame, EndFrame, CancelFrame, StopFrame)
_PASSTHROUGH_FRAMES = (OutputTransportMessageUrgentFrame,)


class BusBridgeProcessor(FrameProcessor, BusSubscriber):
    """Bidirectional mid-pipeline bridge between a Pipecat pipeline and the bus.

    Placed in a transport or session worker's pipeline to exchange frames
    with other workers via the `WorkerBus`. Lifecycle and excluded frames
    pass through locally without crossing the bus.
    """

    def __init__(
        self,
        *,
        bus: WorkerBus,
        worker_name: str,
        target_worker: str | None = None,
        bridge: str | None = None,
        exclude_frames: tuple[type[Frame], ...] | None = None,
        target_task: str | None = None,
        **kwargs,
    ):
        """Initialize the BusBridgeProcessor.

        Args:
            bus: The `WorkerBus` to exchange frames with.
            worker_name: Name of the owning worker, used as message source.
            target_worker: When set, only exchange frames with this worker.
            bridge: Optional bridge name for routing. When set, outgoing
                frames are tagged with this name and only incoming frames
                with the same bridge name are accepted.
            exclude_frames: Extra frame types that should never cross the bus
                (on top of lifecycle frames which are always excluded).
            target_task: When set, only exchange frames with this worker.

                .. deprecated:: 1.8.0
                    Use ``target_worker`` instead. Will be removed in 2.0.0.
            **kwargs: Additional arguments passed to `FrameProcessor`.
        """
        super().__init__(**kwargs)
        if target_task is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "`BusBridgeProcessor(target_task=...)` is deprecated since 1.8.0, "
                    "use `target_worker` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if target_worker is None:
                target_worker = target_task
        self._bus = bus
        self._worker_name = worker_name
        self._target_worker = target_worker
        self._bridge = bridge
        self._exclude_frames = exclude_frames or ()

    async def setup(self, setup: FrameProcessorSetup):
        """Subscribe to the bus during processor setup."""
        await super().setup(setup)
        await self._bus.subscribe(self)

    async def cleanup(self):
        """Unsubscribe from the bus on cleanup."""
        await super().cleanup()
        await self._bus.unsubscribe(self)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame: send to bus, or pass through locally if excluded.

        Args:
            frame: The frame to process.
            direction: The direction the frame is traveling.
        """
        await super().process_frame(frame, direction)

        # Lifecycle frames never cross the bus
        if isinstance(frame, _LIFECYCLE_FRAMES):
            await self.push_frame(frame, direction)
            return

        # Urgent transport frames belong to this pipeline rather than to a
        # peer: they have to reach the transport even with no peer active.
        if isinstance(frame, _PASSTHROUGH_FRAMES):
            await self.push_frame(frame, direction)
            return

        # Excluded frames never cross the bus
        if self._exclude_frames and isinstance(frame, self._exclude_frames):
            await self.push_frame(frame, direction)
            return

        # Send to bus
        msg = BusFrameMessage(
            source=self._worker_name,
            frame=frame,
            direction=direction,
            bridge=self._bridge,
        )
        await self._bus.send(msg)

    async def on_bus_message(self, message: BusMessage) -> None:
        """Handle an incoming bus message by pushing its frame into the pipeline.

        Args:
            message: The bus message to handle.
        """
        if not isinstance(message, BusFrameMessage):
            return

        # Skip own frames
        if message.source == self._worker_name:
            return

        # Filter by bridge name
        if self._bridge and message.bridge != self._bridge:
            return

        # If target_worker set, only accept from that worker
        if self._target_worker and message.source != self._target_worker:
            return

        # If message targeted at someone else, skip
        if message.target and message.target != self._worker_name:
            return

        # The frame has cleared every filter, so this pipeline is the one
        # taking it in.
        if isinstance(message.frame, PipelineFlushFrame):
            self.pipeline_worker.track_flush_probe(message.frame)

        await self.push_frame(message.frame, message.direction)


class _BusEdgeProcessor(FrameProcessor):
    """Pipeline-edge tap that publishes local frames onto the bus.

    Placed by `PipelineWorker` at the source and sink of a bridged
    pipeline. Frames always continue through the local pipeline; in
    addition, frames travelling in ``direction`` are published to the
    bus. The opposite path, a frame arriving from the bus, belongs to
    the worker, which queues it into the pipeline itself.
    """

    def __init__(
        self,
        *,
        worker: "PipelineWorker",
        direction: FrameDirection,
        exclude_frames: tuple[type[Frame], ...] | None = None,
        **kwargs,
    ):
        """Initialize the edge processor.

        Args:
            worker: The owning worker; the edge reads ``worker.bus`` lazily
                so the bus only needs to be set (via
                :meth:`PipelineWorker.attach`) by the time the processor is
                set up. ``worker.name`` is the message source.
            direction: Direction this edge captures and publishes to the
                bus.
            exclude_frames: Extra frame types that should never cross
                the bus (lifecycle frames are always excluded).
            **kwargs: Additional arguments passed to ``FrameProcessor``.
        """
        super().__init__(**kwargs)
        self._task = worker
        self._direction = direction
        self._exclude_frames = exclude_frames or ()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Pass the frame through locally and forward matching ones to the bus."""
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if direction != self._direction:
            return
        if isinstance(frame, _LIFECYCLE_FRAMES):
            return
        if self._exclude_frames and isinstance(frame, self._exclude_frames):
            return

        await self._task.bus.send(
            BusFrameMessage(source=self._task.name, frame=frame, direction=direction)
        )
