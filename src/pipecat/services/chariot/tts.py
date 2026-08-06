#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Chariot text-to-speech service implementation.

This module provides a TTS service using Chariot's streaming WebSocket API.
Chariot serves lifelike voices for Indian languages and English from
infrastructure in India, which keeps time-to-first-audio low for
India-region agents.

See https://docs.chariot.in/api-reference/introduction for full API details.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from urllib.parse import urlencode

from loguru import logger
from websockets.exceptions import ConnectionClosedError
from websockets.protocol import State

from pipecat import version as pipecat_version
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import InterruptibleTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

# Chariot always synthesizes 16-bit mono PCM at 44.1 kHz. The server also
# declares the rate per generation segment in its audio.start event, which
# this service trusts over the constant.
CHARIOT_SAMPLE_RATE = 44100

# Matt, a published Chariot stock voice (English). Used when no voice is
# given so the service works out of the box with just an API key.
CHARIOT_DEFAULT_VOICE = "bac7d666-094d-4698-91fa-741d60fce662"

# Close code the server uses when the workspace's credit balance is exhausted.
_CLOSE_CODE_NO_CREDITS = 4402

# The server caps the un-flushed text buffer at this many characters. Longer
# utterances are sent as several buffer-sized segments, each followed by a
# flush, which is exactly how multi-sentence turns already behave: one
# generation segment per flush, contiguous audio.
_MAX_INPUT_TEXT_CHARS = 500

# The server closes a session that has been idle for this many seconds (300 is
# the maximum it accepts). A keepalive prevents that during quiet stretches;
# if the connection is lost anyway, the service reconnects transparently on
# the next utterance.
_IDLE_TIMEOUT_SECS = 300

# How often to nudge the session during silence. The protocol has no ping
# message, but an input.flush with nothing buffered is a no-op the server
# ignores, and any application message resets the idle timer.
_KEEPALIVE_SECS = 240


def _split_text(text: str, limit: int) -> list[str]:
    """Split text into pieces of at most ``limit`` characters.

    Cuts at the last space inside the limit when there is one, so words stay
    whole; a single over-long token is hard-split. Concatenating the pieces
    reproduces the input exactly, which is what the server's buffer sees.
    """
    pieces = []
    while len(text) > limit:
        cut = text.rfind(" ", 1, limit + 1)
        if cut <= 0:
            cut = limit
        pieces.append(text[:cut])
        text = text[cut:]
    if text:
        pieces.append(text)
    return pieces


@dataclass
class ChariotTTSSettings(TTSSettings):
    """Settings for ChariotTTSService.

    Parameters:
        voice: Chariot voice UUID, from ``GET https://api.chariot.in/v1/voices``.
            Any voice speaks any supported language, so the language of the
            audio follows the text rather than the voice.
    """

    pass


class ChariotTTSService(InterruptibleTTSService):
    """Chariot real-time text-to-speech service using WebSocket streaming.

    Provides real-time text-to-speech synthesis using Chariot's WebSocket API.
    Audio streams back as raw 16-bit mono PCM at 44.1 kHz while the server is
    still generating. The protocol has no per-context cancel message, so
    interruptions are handled by reconnecting the WebSocket.

    Example::

        tts = ChariotTTSService(
            api_key="your-api-key",
            settings=ChariotTTSService.Settings(
                voice="your-voice-uuid",
            ),
        )
    """

    Settings = ChariotTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str | None = None,
        base_url: str = "wss://api.chariot.in",
        sample_rate: int | None = CHARIOT_SAMPLE_RATE,
        optimize_streaming_latency: int = 0,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Chariot WebSocket TTS service.

        Args:
            api_key: Chariot API key for authentication.
            voice_id: Chariot voice UUID from ``GET /v1/voices``.
                ``settings.voice`` takes precedence. Defaults to Matt, a
                Chariot stock voice, when neither is given.
            base_url: Base WebSocket URL for the Chariot API.
            sample_rate: Audio sample rate in Hz. Chariot outputs 44100; the
                server declares the rate per generation segment and that
                declaration is applied to every audio frame.
            optimize_streaming_latency: Latency-optimization level, 0-4.
                0 (default) applies all quality passes; higher values trade
                quality passes for lower latency.
            settings: Runtime-updatable settings; values here win over the
                direct arguments.
            **kwargs: Additional arguments passed to InterruptibleTTSService.
        """
        if sample_rate != CHARIOT_SAMPLE_RATE:
            logger.warning(
                f"Chariot TTS streams at {CHARIOT_SAMPLE_RATE} Hz; "
                f"configured sample_rate={sample_rate}"
            )

        default_settings = self.Settings(voice=voice_id)
        if settings is not None:
            default_settings.apply_update(settings)
        if not default_settings.voice:
            default_settings.voice = CHARIOT_DEFAULT_VOICE

        super().__init__(
            push_text_frames=True,
            pause_frame_processing=True,
            push_stop_frames=True,
            push_start_frame=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._optimize_streaming_latency = optimize_streaming_latency
        # The rate stamped on outgoing audio frames. Updated from every
        # audio.start event, so a server-side change cannot mis-rate audio.
        self._output_sample_rate = sample_rate or CHARIOT_SAMPLE_RATE

        self._receive_task = None
        self._keepalive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Chariot service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Chariot's language format.

        The API takes no language parameter: any voice speaks any supported
        language and the audio follows the text, so there is nothing to map.

        Args:
            language: The language to convert.

        Returns:
            None, always.
        """
        return None

    def _websocket_url(self) -> str:
        query = urlencode(
            {
                "voice_id": self._settings.voice,
                "response_format": "pcm",
                "idle_timeout": _IDLE_TIMEOUT_SECS,
                "optimize_streaming_latency": self._optimize_streaming_latency,
            }
        )
        return f"{self._base_url}/v1/tts/ws?{query}"

    async def start(self, frame: StartFrame):
        """Start the Chariot TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def flush_audio(self, context_id: str | None = None):
        """Force synthesis of any text the server is still buffering."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                await self._websocket.send(json.dumps({"type": "input.flush"}))
        except Exception as e:
            await self.push_error(error_msg=f"Error sending flush to Chariot: {e}", exception=e)

    async def _update_settings(self, delta: TTSSettings) -> dict:
        """Apply a settings delta; a voice change requires a reconnect.

        The voice travels in the connection URL rather than in a message, so
        switching voices mid-session means a new session.
        """
        changed = await super()._update_settings(delta)
        if "voice" in changed and self._websocket and self._websocket.state is State.OPEN:
            await self._disconnect()
            await self._connect()
        return changed

    async def _connect(self):
        """Connect to the Chariot WebSocket and start the receive task."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_task_handler())

    async def _disconnect(self):
        """Disconnect from the Chariot WebSocket and stop the receive task."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish the WebSocket connection to the Chariot API."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            self._websocket = await self._websocket_connect(
                self._websocket_url(),
                additional_headers={
                    "chariotai-api-key": self._api_key,
                    "X-Source": "pipecat",
                    "X-Pipecat-Version": pipecat_version(),
                },
                max_size=None,
            )
            logger.debug("Connected to Chariot TTS WebSocket")

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(
                error_msg=f"Error connecting to Chariot TTS WebSocket: {e}", exception=e
            )
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close the WebSocket connection.

        The session-terminating ``input.done`` message is deliberately not
        sent: on interruption the point is to stop the server generating, and
        closing the socket does that immediately.
        """
        try:
            if self._websocket:
                await self._websocket.close()
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Receive audio and control events from the Chariot WebSocket.

        Binary frames are raw PCM for the active context. JSON control events
        mark segment boundaries: ``audio.start`` opens a generation segment
        and declares the sample rate, ``audio.done`` closes it.
        """
        try:
            async for message in self._get_websocket():
                context_id = self.get_active_audio_context_id()
                if isinstance(message, bytes):
                    await self.stop_ttfb_metrics()
                    frame = TTSAudioRawFrame(
                        message, self._output_sample_rate, 1, context_id=context_id
                    )
                    await self.append_to_audio_context(context_id, frame)
                    continue

                event = json.loads(message)
                event_type = event.get("type")
                if event_type == "audio.start":
                    self._output_sample_rate = event.get("sample_rate") or self._output_sample_rate
                elif event_type == "audio.done":
                    # Synthesis for the active context is complete. Emit the
                    # TTSStoppedFrame immediately so BotStoppedSpeakingFrame
                    # tracks the end of audio, instead of waiting on
                    # stop_frame_timeout_s.
                    if context_id and self.audio_context_available(context_id):
                        await self.append_to_audio_context(
                            context_id, TTSStoppedFrame(context_id=context_id)
                        )
                        await self.remove_audio_context(context_id)
                elif event_type == "error":
                    error_msg = event.get("message", "unknown error")
                    await self.push_error(error_msg=f"Chariot TTS error: {error_msg}")
                    if context_id and self.audio_context_available(context_id):
                        await self.append_to_audio_context(
                            context_id, ErrorFrame(error=f"Chariot TTS error: {error_msg}")
                        )
        except ConnectionClosedError as e:
            if e.rcvd is not None and e.rcvd.code == _CLOSE_CODE_NO_CREDITS:
                await self.push_error(
                    error_msg="Chariot TTS: credit balance exhausted; top up at platform.chariot.in"
                )
            else:
                raise

    async def _keepalive_task_handler(self):
        """Keep the session alive through quiet stretches.

        An empty flush is ignored by the server but resets its idle timer,
        which would otherwise close the session after _IDLE_TIMEOUT_SECS.
        """
        while True:
            await asyncio.sleep(_KEEPALIVE_SECS)
            await self.flush_audio()

    async def _send_text(self, text: str):
        """Send one utterance for synthesis.

        The session is flush-driven: ``input.text`` appends to the server's
        buffer and ``input.flush`` synthesizes everything buffered. The buffer
        is capped, so an over-long utterance is sent as several segments, each
        flushed on its own; every flush produces one generation segment, the
        same shape a multi-sentence turn already has.
        """
        if self._websocket and self._websocket.state is State.OPEN:
            for piece in _split_text(text, _MAX_INPUT_TEXT_CHARS):
                await self._websocket.send(json.dumps({"type": "input.text", "text": piece}))
                await self._websocket.send(json.dumps({"type": "input.flush"}))
        else:
            logger.warning("WebSocket not ready, cannot send text")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech audio frames from input text using Chariot TTS.

        Sends text over the WebSocket for synthesis; audio arrives on the
        receive task and is routed to the audio context.

        Args:
            text: The text input to synthesize.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame objects, or None when audio is delivered via the context.
        """
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                await self._send_text(text)
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
