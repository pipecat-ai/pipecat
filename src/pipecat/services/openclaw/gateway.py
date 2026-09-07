#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A pipeline service for the OpenClaw Gateway."""

import asyncio
from contextlib import suppress

from loguru import logger

from pipecat.frames.frames import CancelFrame, EndFrame, Frame
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessorSetup,
)
from pipecat.services.ai_service import AIService
from pipecat.services.openclaw.client import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_GATEWAY_URL,
    DEFAULT_MAX_MESSAGE_SIZE,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_ROLE,
    DEFAULT_RUN_TIMEOUT,
    DEFAULT_SESSION_KEY,
    OpenClawGatewayClient,
    OpenClawRun,
)
from pipecat.services.openclaw.frames import (
    OpenClawAbortFrame,
    OpenClawEndFrame,
    OpenClawSendFrame,
    OpenClawStartedFrame,
    OpenClawSteerFrame,
    OpenClawTextFrame,
)
from pipecat.services.settings import ServiceSettings


class OpenClawGatewayService(AIService):
    """Turns OpenClaw Gateway traffic into frames, and frames into Gateway calls.

    Downstream, :class:`~pipecat.services.openclaw.frames.OpenClawSendFrame`
    starts a run, :class:`~pipecat.services.openclaw.frames.OpenClawSteerFrame`
    redirects the one in flight, and
    :class:`~pipecat.services.openclaw.frames.OpenClawAbortFrame` stops it.
    What the Gateway streams back is pushed as
    :class:`~pipecat.services.openclaw.frames.OpenClawTextFrame` between an
    :class:`~pipecat.services.openclaw.frames.OpenClawStartedFrame` and one
    :class:`~pipecat.services.openclaw.frames.OpenClawEndFrame`.

    An agent run is not a spoken turn: it can take minutes and it answers in
    prose. What that should sound like belongs to whatever wraps this, which
    is why nothing here reads or writes conversational frames.

    A session runs one turn at a time, so a send arriving while a run is live
    stops that run first. Every started frame is followed by exactly one end
    frame.

    Example::

        service = OpenClawGatewayService(token=os.getenv("OPENCLAW_TOKEN"))
    """

    def __init__(
        self,
        *,
        url: str = DEFAULT_GATEWAY_URL,
        token: str | None = None,
        password: str | None = None,
        session_key: str = DEFAULT_SESSION_KEY,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        run_timeout: float = DEFAULT_RUN_TIMEOUT,
        scopes: list[str] | None = None,
        role: str = DEFAULT_ROLE,
        max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
        reconnect_on_error: bool = True,
        **kwargs,
    ):
        """Initialize the service and the client it drives.

        Args:
            url: The Gateway websocket, or the port a NemoClaw sandbox
                republishes it on.
            token: The Gateway's shared token. Required even on loopback.
            password: Gateway password, if the deployment uses one instead.
            session_key: Which OpenClaw session to run in.
            connect_timeout: Seconds to wait for the handshake.
            request_timeout: Seconds to wait for a Gateway method to answer.
            run_timeout: Seconds the agent is given to finish a run.
            scopes: Handshake scopes.
            role: Handshake role.
            max_message_size: Largest websocket frame to accept.
            reconnect_on_error: Whether to reconnect after the socket fails.
            **kwargs: Additional arguments passed to :class:`AIService`.

        See :class:`~pipecat.services.openclaw.client.OpenClawGatewayClient`
        for what each of these means to the Gateway.
        """
        super().__init__(settings=ServiceSettings(model=None), **kwargs)
        self._client = OpenClawGatewayClient(
            url=url,
            token=token,
            password=password,
            session_key=session_key,
            connect_timeout=connect_timeout,
            request_timeout=request_timeout,
            run_timeout=run_timeout,
            scopes=scopes,
            role=role,
            max_message_size=max_message_size,
            reconnect_on_error=reconnect_on_error,
        )
        self._run: OpenClawRun | None = None
        self._stream_task: asyncio.Task | None = None
        self._run_ended = True

    @property
    def client(self) -> OpenClawGatewayClient:
        """The Gateway client this service drives."""
        return self._client

    async def setup(self, setup: FrameProcessorSetup):
        """Wire the client up with the pipeline's task manager, and connect.

        Connecting here rather than on the ``StartFrame`` overlaps the Gateway
        handshake with the rest of the pipeline setting up.

        Args:
            setup: Configuration parameters for the frame processor.
        """
        await super().setup(setup)
        await self._client.setup(setup.task_manager)

        @self._client.event_handler("on_connection_error")
        async def _on_connection_error(client, message: str, force_treat_as_permanent: bool):
            await self.push_error(message, force_treat_as_permanent=force_treat_as_permanent)

        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the run in flight and close the connection.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._shutdown()

    async def cancel(self, frame: CancelFrame):
        """Stop the run in flight and close the connection, at once.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._shutdown()

    async def cleanup(self):
        """Release the run in flight and the connection."""
        await super().cleanup()
        await self._shutdown()
        await self._client.cleanup()

    async def _shutdown(self):
        """Stop whatever is in flight and let go of the connection."""
        await self._stop_run(notify=False)
        await self._client.disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame, driving the Gateway from the ones addressed to it.

        Args:
            frame: The frame to process.
            direction: The direction the frame is travelling.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, OpenClawSendFrame):
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
        await self.push_frame(OpenClawStartedFrame(run_id=run.run_id))
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
            elif event.kind in ("completed", "cancelled", "failed"):
                await self._end(
                    OpenClawEndFrame(run_id=run.run_id, status=event.kind, text=event.text)
                )
                if event.kind == "failed":
                    # A run the Gateway gave up on is the pipeline's business
                    # too, not just the caller's.
                    await self.push_error(f"{self} OpenClaw run failed: {event.text}")

    async def _end(self, frame: Frame):
        """Push a run's terminal frame and let go of the run.

        The service stops holding a finished run so that a steer or an abort
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
                await self._client.abort(run, "the service stopped the run")
        if self._stream_task:
            await self.cancel_task(self._stream_task)
            self._stream_task = None
        if run and not self._run_ended:
            self._run_ended = True
            if notify:
                await self.push_frame(OpenClawEndFrame(run_id=run.run_id, status="cancelled"))
