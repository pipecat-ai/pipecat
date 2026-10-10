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
for transcription and the recording. ``capture_bot_images`` reports each
image the bot outputs, by size and format. ``trigger_disconnect`` fires the bot's
``on_client_disconnected`` when the connection ends; it is off by default,
since bots often cancel their pipeline there and the server serves several
scenarios in a row.

The input transport also serves the harness's image to a vision bot, which
has no camera under eval. The user's audio arrives as a continuous stream, at
the rate the scenario synthesized it, which need not be the bot's
``audio_in_sample_rate``; audio at another rate is resampled on the way in.
"""

import asyncio
import io
from urllib.parse import parse_qs, urlsplit

from loguru import logger
from PIL import Image

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMConfigureOutputFrame,
    OutputImageRawFrame,
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
CAPTURE_IMAGES_QUERY_PARAM = "capture_bot_images"
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
    """Input transport that serves the harness's image and matches its audio rate.

    A vision bot asks for the user's camera image; under eval there is no
    camera, so the image the harness registered for the turn is served instead.

    The user's speech is synthesized at the rate its scenario declared, which
    need not be the bot's ``audio_in_sample_rate``. Audio arriving at another
    rate is resampled to the transport's, so the pipeline's VAD, turn detection
    and STT see the signal at the rate they were configured for instead of a
    time-stretched, pitch-shifted one.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the transport.

        Args:
            *args: Forwarded to
                :class:`~pipecat.transports.websocket.server.SingleClientWebsocketServerInputTransport`.
            **kwargs: Forwarded to the parent transport.
        """
        super().__init__(*args, **kwargs)
        # Created on the first frame that needs resampling, and replaced when the
        # rate changes: a stream resampler is bound to one rate pair, and a
        # kept-alive server serves scenarios that speak at different rates.
        self._resampler: BaseAudioResampler | None = None
        self._resampler_rates: tuple[int, int] | None = None

    async def push_audio_frame(self, frame: InputAudioRawFrame):
        """Push the frame to the audio path, resampled if it is at another rate.

        Args:
            frame: The input audio frame.
        """
        if self._needs_resampling(frame):
            frame = await self._resampled(frame)
        await super().push_audio_frame(frame)

    def _needs_resampling(self, frame: InputAudioRawFrame) -> bool:
        """Whether the frame's audio is at a known rate other than the transport's.

        A rate of 0 means unknown: the transport's is unset until ``setup()``, and
        a frame that declares none carries no rate to convert from.
        """
        return bool(
            frame.audio
            and self.sample_rate
            and frame.sample_rate
            and frame.sample_rate != self.sample_rate
        )

    async def _resampled(self, frame: InputAudioRawFrame) -> InputAudioRawFrame:
        """The frame with its audio resampled to the transport's input rate."""
        if frame.num_channels != 1:
            # The resampler is mono-only, and mistaking interleaved channels for
            # mono ones would pitch-shift the audio rather than convert it.
            logger.warning(
                f"{self}: cannot resample {frame.num_channels}-channel user audio from "
                f"{frame.sample_rate} Hz to {self.sample_rate} Hz; passing it on as is"
            )
            return frame

        rates = (frame.sample_rate, self.sample_rate)
        if self._resampler is None or self._resampler_rates != rates:
            self._resampler = create_stream_resampler()
            self._resampler_rates = rates
            logger.debug(f"{self}: resampling user audio {rates[0]} -> {rates[1]} Hz")
        audio = await self._resampler.resample(frame.audio, rates[0], rates[1])
        return InputAudioRawFrame(
            audio=audio,
            sample_rate=self.sample_rate,
            num_channels=frame.num_channels,
        )

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
    """Output transport that reports the bot's images to the harness.

    The WebSocket server output has no video, so an image the bot outputs is
    handed to the serializer as it arrives, whether or not the bot enabled
    video output; the serializer sends it only when the harness asked.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Report output images; otherwise behave like the base output transport."""
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputImageRawFrame):
            await self._write_frame(frame)


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
        if serializer is not None and hasattr(serializer, "set_capture_images"):
            serializer.set_capture_images(_query_flag(websocket, CAPTURE_IMAGES_QUERY_PARAM))

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
