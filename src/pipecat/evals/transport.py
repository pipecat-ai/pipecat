#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WebSocket server transport for the eval harness.

A subclass of :class:`~pipecat.transports.websocket.server.SingleClientWebsocketServerTransport`
that adds eval-only behavior driven by per-connection query flags the harness
sets:

- ``?skip_tts=true`` silences the bot's output for the session (text mode),
  including any on-connect greeting. This is pushed as an
  :class:`~pipecat.frames.frames.LLMConfigureOutputFrame` *before*
  ``on_client_connected`` fires: pipecat processes frames in order, and a bot
  that greets in ``on_client_connected`` queues its greeting there, so a config
  sent afterwards (as a client message) would arrive too late.
- ``?capture_bot_audio=true`` makes the serializer forward the bot's synthesized
  audio to the harness, for transcription (``response`` / ``tts_response``) and so
  the harness can record the bot's side. Recording lives in the harness pipeline
  now, not here: the harness already sees both sides (the bot's audio via this
  flag, the user's as its own TTS output), so the bot needs no recorder.

The input side needs no special handling: the harness streams the user audio over
the wire as a continuous real-time stream (paced, with silence when idle — see
:class:`~pipecat.evals.client_transport.EvalHarnessOutputTransport`), so the bot's
stock input transport consumes it directly. This input transport only adds image
serving (a function-calling-video bot has no camera under eval).

Client disconnects behave as on any transport: the bot's
``on_client_disconnected`` handler fires normally, and whether the pipeline
survives the disconnect is the application's choice. The server itself keeps
running either way, so a bot that opts not to cancel can serve several
sequential eval connections.
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
    """Transport parameters for the eval harness.

    A thin subclass of :class:`~pipecat.transports.websocket.server.SingleClientWebsocketServerParams`
    that gives the eval transport its own parameter type. Bots configure the
    ``"eval"`` entry of ``transport_params`` with this class so the eval setup
    reads as eval-specific rather than leaking the underlying WebSocket server
    transport.
    """

    pass


class EvalInputTransport(SingleClientWebsocketServerInputTransport):
    """Input transport that serves the harness's images.

    A function-calling-video bot pushes a ``UserImageRequestFrame`` upstream when
    it needs the user's camera image. There is no camera under eval, so we serve
    the image the harness registered for the turn (an ``eval-image`` message,
    stored on the serializer) as a ``UserImageRawFrame`` — mirroring
    ``daily/transport.py`` but sourcing the image from the serializer instead of a
    live video frame.

    The harness streams the user audio over the wire as a continuous real-time
    stream (see :class:`~pipecat.evals.client_transport.EvalHarnessOutputTransport`),
    so this side needs no special handling: the bot's stock input handles the
    incoming audio.
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
    """Output transport used by the eval harness.

    The eval harness sends the bot's output over the same WebSocket connection
    as any client, so this currently adds no behavior beyond
    :class:`~pipecat.transports.websocket.server.SingleClientWebsocketServerOutputTransport`.
    It exists for naming symmetry with :class:`EvalInputTransport` and as a hook
    for any future eval-specific output behavior.
    """

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

        if self._input is not None and _query_flag(websocket, SKIP_TTS_QUERY_PARAM):
            logger.debug(f"{self}: eval client requested skip_tts; configuring LLM output")
            await self._input.push_frame(LLMConfigureOutputFrame(skip_tts=True))

        await super()._on_client_connected(websocket)

    async def _emit_client_disconnected(self, websocket):
        """Fire ``on_client_disconnected`` only when the harness asks for it.

        Bots often cancel their pipeline in ``on_client_disconnected``, so the
        event is suppressed by default to avoid that between eval scenarios. The
        harness sets ``?trigger_disconnect=true`` (via a scenario's
        ``trigger_disconnect`` field or ``pipecat eval run --trigger-disconnect``)
        to exercise the bot's disconnect path. Independent of ``--stop-bot``,
        which tears the bot down reliably via ``eval-cancel``.
        """
        if _query_flag(websocket, TRIGGER_DISCONNECT_QUERY_PARAM):
            await super()._emit_client_disconnected(websocket)
