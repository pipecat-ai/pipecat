#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A pipeline processor for the OpenClaw Gateway."""

import asyncio
from contextlib import suppress

from loguru import logger

from pipecat.frames.frames import CancelFrame, EndFrame, Frame, StartFrame
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
    FrameProcessorSetup,
)
from pipecat.services.openclaw.client import OpenClawGatewayClient, OpenClawRun
from pipecat.services.openclaw.frames import (
    OpenClawAbortFrame,
    OpenClawRunCancelledFrame,
    OpenClawRunCompletedFrame,
    OpenClawRunFailedFrame,
    OpenClawRunStartedFrame,
    OpenClawSendFrame,
    OpenClawSteerFrame,
    OpenClawTextFrame,
)


class OpenClawGatewayProcessor(FrameProcessor):
    """Turns OpenClaw Gateway traffic into frames, and frames into Gateway calls.

    Downstream, :class:`~pipecat.services.openclaw.frames.OpenClawSendFrame`
    starts a run, :class:`~pipecat.services.openclaw.frames.OpenClawSteerFrame`
    redirects the one in flight, and
    :class:`~pipecat.services.openclaw.frames.OpenClawAbortFrame` stops it.
    What the Gateway streams back is pushed as
    :class:`~pipecat.services.openclaw.frames.OpenClawTextFrame` between a
    started frame and one terminal frame.

    An agent run is not a spoken turn: it can take minutes and it answers in
    prose. What that should sound like belongs to a service wrapping this
    processor, which is why nothing here reads or writes conversational frames.

    A session runs one turn at a time, so a send arriving while a run is live
    stops that run first. Every started frame is followed by exactly one
    terminal frame.

    Example::

        processor = OpenClawGatewayProcessor(
            OpenClawGatewayClient(token=os.getenv("OPENCLAW_TOKEN"))
        )
    """

    def __init__(self, client: OpenClawGatewayClient, **kwargs):
        """Initialize the processor.

        Args:
            client: The Gateway client to drive.
            **kwargs: Additional arguments passed to :class:`FrameProcessor`.
        """
        super().__init__(**kwargs)
        self._client = client
        self._run: OpenClawRun | None = None
        self._stream_task: asyncio.Task | None = None
        self._run_ended = True

    @property
    def client(self) -> OpenClawGatewayClient:
        """The Gateway client this processor drives."""
        return self._client

    async def setup(self, setup: FrameProcessorSetup):
        """Wire the client up with the pipeline's task manager.

        Args:
            setup: Configuration parameters for the frame processor.
        """
        await super().setup(setup)
        await self._client.setup(setup.task_manager)

        @self._client.event_handler("on_connection_error")
        async def _on_connection_error(client, message: str):
            await self.push_error(message)

    async def cleanup(self):
        """Release the run in flight and the connection."""
        await super().cleanup()
        await self._stop_run(notify=False)
        await self._client.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame, driving the Gateway from the ones addressed to it.

        Args:
            frame: The frame to process.
            direction: The direction the frame is travelling.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self._connect()
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop_run(notify=False)
            await self._client.disconnect()
            await self.push_frame(frame, direction)
        elif isinstance(frame, OpenClawSendFrame):
            await self._send(frame)
        elif isinstance(frame, OpenClawSteerFrame):
            await self._steer(frame)
        elif isinstance(frame, OpenClawAbortFrame):
            # An abort is a system frame, so it must not hold the system frame
            # path open for a Gateway round trip.
            self.create_task(self._abort(frame), name=f"{self}::abort")
        else:
            await self.push_frame(frame, direction)

    async def _connect(self):
        """Open the connection, reporting a Gateway that cannot be reached."""
        try:
            await self._client.connect()
        except Exception as e:
            await self.push_error(f"{self} could not reach the OpenClaw Gateway: {e}", exception=e)

    async def _send(self, frame: OpenClawSendFrame):
        """Start a run and stream it downstream."""
        await self._stop_run(notify=True)
        try:
            run = await self._client.start(frame.message, session_key=frame.session_key)
        except Exception as e:
            await self.push_error(f"{self} could not start an OpenClaw run: {e}", exception=e)
            return

        self._run = run
        self._run_ended = False
        await self.push_frame(OpenClawRunStartedFrame(run_id=run.run_id))
        self._stream_task = self.create_task(self._stream(run), name=f"{self}::stream")

    async def _steer(self, frame: OpenClawSteerFrame):
        """Redirect the run in flight."""
        if not self._run:
            logger.warning(f"{self} ignoring a steer with no run in flight")
            return
        try:
            await self._client.steer(self._run, frame.message)
        except Exception as e:
            await self.push_error(f"{self} could not steer the OpenClaw run: {e}", exception=e)

    async def _abort(self, frame: OpenClawAbortFrame):
        """Stop the run in flight.

        The Gateway reports the stop on the run's event stream, so the terminal
        frame comes from :meth:`_stream` rather than from here.
        """
        if not self._run:
            return
        try:
            await self._client.abort(self._run, frame.reason)
        except Exception as e:
            await self.push_error(f"{self} could not abort the OpenClaw run: {e}", exception=e)

    async def _stream(self, run: OpenClawRun):
        """Push a run's events downstream until it ends."""
        async for event in self._client.events(run):
            if event.kind == "text_delta":
                await self.push_frame(OpenClawTextFrame(text=event.text, run_id=run.run_id))
            elif event.kind == "completed":
                await self._end(OpenClawRunCompletedFrame(run_id=run.run_id, text=event.text))
            elif event.kind == "cancelled":
                await self._end(OpenClawRunCancelledFrame(run_id=run.run_id, text=event.text))
            elif event.kind == "failed":
                await self._end(OpenClawRunFailedFrame(run_id=run.run_id, error=event.text))

    async def _end(self, frame: Frame):
        """Push a run's terminal frame and let go of the run.

        The processor stops holding a finished run so that a steer or an abort
        arriving afterwards has nothing to act on. Steering a run the Gateway
        has already finished would start a replacement nobody is streaming.
        """
        self._run = None
        self._stream_task = None
        self._run_ended = True
        await self.push_frame(frame)

    async def _stop_run(self, *, notify: bool):
        """Stop whatever is in flight and stop streaming it.

        Args:
            notify: Whether to report a run that never reached a terminal frame
                as cancelled. A pipeline that is ending has nowhere to report it.
        """
        run, self._run = self._run, None
        if run and not run.done:
            with suppress(Exception):
                await self._client.abort(run, "the processor stopped the run")
        if self._stream_task:
            await self.cancel_task(self._stream_task)
            self._stream_task = None
        if run and not self._run_ended:
            self._run_ended = True
            if notify:
                await self.push_frame(OpenClawRunCancelledFrame(run_id=run.run_id))
