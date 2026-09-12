#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The bot's eval transport: a WebSocket server speaking RTVI, plus the eval's own behavior.

The harness sets per-connection query flags. ``skip_tts`` silences the
bot's speech for the session (text mode), applied before
``on_client_connected`` so a greeting made there is silent too.
``capture_bot_audio`` forwards the bot's synthesized audio to the harness,
for transcription and the recording. ``trigger_disconnect`` fires the bot's
``on_client_disconnected`` when the connection ends; it is off by default,
since bots often cancel their pipeline there and the server serves several
scenarios in a row.

The input transport also serves the harness's image to a vision bot, which
has no camera under eval. The user's audio arrives as a continuous stream,
so nothing else is special on the way in.
"""

import asyncio
import io
from urllib.parse import parse_qs, urlsplit

from loguru import logger
from PIL import Image

from pipecat.frames.frames import (
    Frame,
    LLMConfigureOutputFrame,
    UserImageRawFrame,
    UserImageRequestFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.websocket.server import (
    SingleClientWebsocketServerInputTransport,
    SingleClientWebsocketServerOutputTransport,
    SingleClientWebsocketServerParams,
    SingleClientWebsocketServerTransport,
)

SKIP_TTS_QUERY_PARAM = "skip_tts"
CAPTURE_AUDIO_QUERY_PARAM = "capture_bot_audio"
TRIGGER_DISCONNECT_QUERY_PARAM = "trigger_disconnect"


def _query_string(websocket) -> str:
    """The connection URL's query string (handles both websockets API versions)."""
    # websockets exposes the request target as ``.path`` (legacy) or
    # ``.request.path`` (newer); both include the query string.
    path = getattr(websocket, "path", None)
    if path is None:
        request = getattr(websocket, "request", None)
        path = getattr(request, "path", "") if request is not None else ""
    return urlsplit(path or "").query


def _query_flag(websocket, name: str) -> bool:
    """Whether the client's connection URL set the boolean query param ``name``."""
    values = parse_qs(_query_string(websocket)).get(name, [])
    return bool(values) and values[0].strip().lower() in ("1", "true", "yes")


class EvalTransportParams(SingleClientWebsocketServerParams):
    """Parameters of the eval transport, so a bot's ``transport_params`` names it as such."""

    pass


class EvalInputTransport(SingleClientWebsocketServerInputTransport):
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
        serializer = getattr(self._params, "serializer", None)
        image = None
        if serializer is not None and hasattr(serializer, "get_user_image"):
            image = serializer.get_user_image()
        if image is None:
            logger.warning(f"{self}: UserImageRequestFrame but no eval image registered")
            return
        data, _fmt = image
        # The harness sends the image encoded over the wire, but a real camera
        # transport pushes raw frames -- so decode it to raw RGB here and serve a
        # genuine ``UserImageRawFrame``. The LLM context re-encodes raw frames to
        # JPEG anyway, and consumers that decode directly (e.g. a local vision
        # model doing ``Image.frombytes``) need the raw pixels and real size.
        decoded = await asyncio.to_thread(lambda: Image.open(io.BytesIO(data)).convert("RGB"))
        await self.push_frame(
            UserImageRawFrame(
                image=decoded.tobytes(),
                size=decoded.size,
                format="RGB",
                user_id=request.user_id,
                text=request.text,
                append_to_context=request.append_to_context,
                request=request,
            )
        )


class EvalOutputTransport(SingleClientWebsocketServerOutputTransport):
    """Output transport of the eval transport; adds nothing to the WebSocket server output yet, and exists for symmetry with :class:`EvalInputTransport`."""

    pass


class EvalTransport(SingleClientWebsocketServerTransport):
    """WebSocket server transport used by the eval harness (see the module docstring)."""

    def input(self) -> SingleClientWebsocketServerInputTransport:
        """Return an input transport that can serve harness-provided images."""
        if not self._input:
            self._input = EvalInputTransport(
                self, self._host, self._port, self._params, self._callbacks, name=self._input_name
            )
        return self._input

    def output(self) -> SingleClientWebsocketServerOutputTransport:
        """Return the eval output transport."""
        if not self._output:
            self._output = EvalOutputTransport(self, self._params, name=self._output_name)
        return self._output

    async def _on_client_connected(self, websocket):
        """Apply per-connection eval flags, then proceed (config before any greeting)."""
        serializer = getattr(self._params, "serializer", None)
        if serializer is not None and hasattr(serializer, "set_capture_audio"):
            serializer.set_capture_audio(_query_flag(websocket, CAPTURE_AUDIO_QUERY_PARAM))

        if self._input is not None:
            skip_tts = _query_flag(websocket, SKIP_TTS_QUERY_PARAM)
            logger.debug(f"{self}: configuring eval LLM output with {skip_tts=}")
            await self._input.push_frame(LLMConfigureOutputFrame(skip_tts=skip_tts))

        await super()._on_client_connected(websocket)

    async def _emit_client_disconnected(self, websocket):
        """Fire ``on_client_disconnected`` only when the harness asked for it.

        Bots often cancel their pipeline there, which would end it between scenarios.
        """
        if _query_flag(websocket, TRIGGER_DISCONNECT_QUERY_PARAM):
            await super()._emit_client_disconnected(websocket)
