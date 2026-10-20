#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The eval transport over a WebSocket the bot accepted itself.

On Pipecat Cloud a session's WebSocket arrives already accepted, and the hops
in between pass on only the session body, never the URL's query. The harness
therefore sends its connection's flags as its first message, ``eval-connect``,
which ``create_transport`` reads before building this transport with them.
The flags are those :class:`~pipecat.evals.transport.EvalTransport` reads from
its URL, applied the same way: ``skip_tts`` before ``on_client_connected``, so
a greeting made there is silent too.

The connection is the session, so ``on_client_disconnected`` fires whenever
the harness disconnects, letting the bot end with it.
"""

from fastapi import WebSocket
from loguru import logger

from pipecat.evals.serializer import EvalConnectionFlags, EvalSerializer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import (
    Frame,
    LLMConfigureOutputFrame,
    OutputImageRawFrame,
    UserImageRequestFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketInputTransport,
    FastAPIWebsocketOutputTransport,
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)


class EvalFastAPIWebsocketInputTransport(FastAPIWebsocketInputTransport):
    """Input transport that serves the harness's image.

    A vision bot asks for the user's camera image; under eval there is no
    camera, so the image the harness registered for the turn is served instead.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Serve image requests; otherwise behave like the base input transport."""
        await super().process_frame(frame, direction)
        if isinstance(frame, UserImageRequestFrame):
            await self._serve_user_image(frame)

    async def _serve_user_image(self, request: UserImageRequestFrame) -> None:
        serializer = self._params.serializer
        image = None
        if isinstance(serializer, EvalSerializer):
            image = await serializer.user_image_frame(request)
        if image is None:
            logger.warning(f"{self}: UserImageRequestFrame but no eval image registered")
            return
        await self.push_frame(image)


class EvalFastAPIWebsocketOutputTransport(FastAPIWebsocketOutputTransport):
    """Output transport that reports the bot's images to the harness.

    An image the bot outputs is handed to the serializer as it arrives, whether
    or not the bot enabled video output; the serializer sends it only when the
    harness asked.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Report output images; otherwise behave like the base output transport."""
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputImageRawFrame):
            await self._write_frame(frame)


class EvalFastAPIWebsocketTransport(FastAPIWebsocketTransport):
    """The eval transport over an accepted FastAPI WebSocket (see the module docstring).

    Event handlers available:

    - on_client_connected(transport, websocket): The harness connected
    - on_client_disconnected(transport, websocket): The harness disconnected
    - on_session_timeout(transport, websocket): Session timed out
    """

    def __init__(
        self,
        websocket: WebSocket,
        params: EvalTransportParams,
        flags: EvalConnectionFlags,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        """Initialize the transport.

        Args:
            websocket: The accepted WebSocket, its ``eval-connect`` message
                already read.
            params: The bot's eval transport parameters.
            flags: The connection's flags, from its ``eval-connect`` message.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
        """
        super().__init__(
            websocket,
            FastAPIWebsocketParams(**params.model_dump()),
            input_name=input_name,
            output_name=output_name,
        )
        self._flags = flags

        serializer = self._params.serializer
        if isinstance(serializer, EvalSerializer):
            serializer.set_capture_audio(flags.capture_bot_audio)
            serializer.set_capture_images(flags.capture_bot_images)

        self._input = EvalFastAPIWebsocketInputTransport(
            self, self._client, self._params, name=self._input_name
        )
        self._output = EvalFastAPIWebsocketOutputTransport(
            self, self._client, self._params, name=self._output_name
        )

    async def _on_client_connected(self, websocket):
        """Configure the bot's output for the connection, then proceed (config before any greeting)."""
        skip_tts = self._flags.skip_tts
        logger.debug(f"{self}: configuring eval LLM output with {skip_tts=}")
        await self._input.push_frame(LLMConfigureOutputFrame(skip_tts=skip_tts))
        await super()._on_client_connected(websocket)
