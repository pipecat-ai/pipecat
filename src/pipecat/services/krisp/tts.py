#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Krisp streaming text-to-speech service implementation.

Krisp's TTS WebSocket synthesizes ``text`` messages as they arrive and streams
untagged 24 kHz PCM. The server answers every ``text`` message with one
``end_of_sentence`` once its audio has been sent, which is how the service knows
a turn's audio is complete.
"""

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.krisp.common import (
    KRISP_REST_BASE_URL,
    KrispServiceError,
    connect_session,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import InterruptibleTTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven

KRISP_TTS_URL = "wss://streaming.krisp.ai/tts"

# The API streams 24 kHz mono 16-bit PCM.
KRISP_TTS_SAMPLE_RATE = 24000


def language_to_krisp_tts_language(language: Language) -> str:
    """Convert a Language enum to a Krisp TTS language code.

    Args:
        language: The language to convert.

    Returns:
        The Krisp language code, e.g. ``en-US``.
    """
    LANGUAGE_MAP = {
        Language.DE: "de-DE",
        Language.DE_DE: "de-DE",
        Language.EN: "en-US",
        Language.EN_US: "en-US",
        Language.ES: "es-ES",
        Language.ES_ES: "es-ES",
        Language.ES_US: "es-US",
        Language.FR: "fr-FR",
        Language.FR_FR: "fr-FR",
        Language.RU: "ru-RU",
        Language.RU_RU: "ru-RU",
    }
    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


@dataclass
class KrispTTSSettings(TTSSettings):
    """Settings for KrispTTSService.

    Parameters:
        speed: Playback speed: 0.9, 0.95, 1.0, 1.05 or 1.1. None uses the
            voice's configured speed.
    """

    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class KrispTTSService(InterruptibleTTSService):
    """Krisp streaming text-to-speech service.

    Each ``run_tts`` call sends its text for synthesis; audio is appended to the
    active audio context, and the context ends once the turn's text has been
    flushed and the server has answered every ``text`` message. Krisp's audio
    carries no context ID, so the connection is reopened on interruption while
    the bot is speaking (see :class:`InterruptibleTTSService`).
    """

    Settings = KrispTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = KRISP_TTS_URL,
        rest_base_url: str = KRISP_REST_BASE_URL,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Krisp TTS service.

        Args:
            api_key: Krisp master API key. It is only used to create a session
                token before each connection and never sent on the WebSocket.
            url: Krisp TTS WebSocket endpoint.
            rest_base_url: Krisp REST API base URL used to create session tokens.
            settings: Runtime-updatable settings (``voice``, ``language``,
                ``speed``, ``model``). Changing any of them reconnects, since
                they are fixed by the session handshake.
            **kwargs: Additional arguments passed to :class:`InterruptibleTTSService`.
        """
        default_settings = self.Settings(
            model="krisp-tts-v1", voice="Amy", language="en-US", speed=None
        )
        if settings is not None:
            default_settings.apply_update(settings)

        # pause_frame_processing: audio carries no context ID, so the next
        # reply's text waits until the previous reply has finished playing.
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            pause_frame_processing=True,
            sample_rate=KRISP_TTS_SAMPLE_RATE,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._rest_base_url = rest_base_url
        self._receive_task = None

        # ``text`` messages not yet answered with ``end_of_sentence``, and the
        # context whose text is complete (set by ``flush_audio``).
        self._pending_texts = 0
        self._flushed_context_id: str | None = None
        # Incremented on every new connection; texts sent on an earlier one
        # will never be answered.
        self._connection_id = 0
        # Set when a settings change arrives while the bot is speaking; the
        # reconnect that applies it waits until the bot stops.
        self._reconnect_after_turn = False

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Krisp TTS supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to a Krisp TTS language code.

        Args:
            language: The language to convert.

        Returns:
            The Krisp language code.
        """
        return language_to_krisp_tts_language(language)

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect with a new handshake.

        Settings are fixed per connection. While the bot is speaking, the
        reconnect is deferred until it stops, so the current utterance isn't
        cut off.

        Args:
            delta: A :class:`TTSSettings` (or ``KrispTTSService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if changed and self._websocket:
            if self._bot_speaking or self._pending_texts:
                self._reconnect_after_turn = True
            else:
                await self._disconnect()
                await self._connect()
        return changed

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, applying a deferred settings reconnect once the bot stops.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if (
            isinstance(frame, BotStoppedSpeakingFrame)
            and self._reconnect_after_turn
            and not self._pending_texts
        ):
            await self._disconnect()
            await self._connect()

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def _connect(self):
        await super()._connect()

        await self._connect_websocket()

        # A receive loop that has ended (e.g. after a permanent error closed the
        # socket) can't serve the new connection.
        if self._receive_task and self._receive_task.done():
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()
        # The reply is torn down here, not on every reconnect: a reconnect
        # inside ``send_with_retry`` keeps the reply going.
        await self.remove_active_audio_context()
        self._flushed_context_id = None

    def _build_handshake(self) -> dict[str, Any]:
        handshake: dict[str, Any] = {
            "model": self._settings.model,
            "voice": self._settings.voice,
        }
        if self._settings.language:
            handshake["language"] = self._settings.language
        if self._settings.speed is not None:
            handshake["speed"] = self._settings.speed
        return handshake

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Krisp TTS")
            websocket, ready = await connect_session(
                self._websocket_connect,
                self._url,
                api_key=self._api_key,
                usage_type="tts",
                rest_base_url=self._rest_base_url,
                handshake=self._build_handshake(),
            )
            self._websocket = websocket
            # Texts sent on the previous connection will never be answered,
            # and the new connection already uses the current settings.
            self._connection_id += 1
            self._pending_texts = 0
            self._reconnect_after_turn = False
            await self._maybe_finish_context()
            if ready.get("warning"):
                logger.warning(f"{self}: {ready['warning']}")
            logger.debug(
                f"Connected to Krisp TTS (request_id={ready.get('request_id')}, "
                f"voice={ready.get('voice')}, language={ready.get('language')})"
            )
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Krisp TTS: {e}", exception=e)
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug("Disconnecting from Krisp TTS")
                await self._websocket.send(json.dumps({"type": "end"}))
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def flush_audio(self, context_id: str | None = None):
        """Mark the turn's text as complete.

        The context ends once the server has answered every ``text`` message
        sent so far.

        Args:
            context_id: The context to flush. If None, uses the active context.
        """
        flush_id = context_id or self.get_active_audio_context_id()
        if not flush_id:
            return
        self._flushed_context_id = flush_id
        await self._maybe_finish_context()

    async def on_audio_context_interrupted(self, context_id: str):
        """Stop metrics when the bot is interrupted.

        Args:
            context_id: The interrupted context.
        """
        await self.stop_all_metrics()
        await super().on_audio_context_interrupted(context_id)

    async def _maybe_finish_context(self):
        context_id = self._flushed_context_id
        if not context_id or self._pending_texts:
            return
        self._flushed_context_id = None
        if self.audio_context_available(context_id):
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

    async def _receive_messages(self):
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                context_id = self.get_active_audio_context_id()
                if context_id and self.audio_context_available(context_id):
                    frame = TTSAudioRawFrame(
                        audio=message,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )
                    await self.append_to_audio_context(context_id, frame)
                continue

            msg = json.loads(message)
            msg_type = msg.get("type")
            if msg_type == "end_of_sentence":
                self._pending_texts = max(0, self._pending_texts - 1)
                await self._maybe_finish_context()
            elif msg_type == "error":
                error = KrispServiceError.from_message(msg)
                await self.stop_all_metrics()
                await self.push_error(error_msg=f"Krisp TTS error: {error}", exception=error)
                if error.is_permanent:
                    # The error marks the service unusable, so the receive loop
                    # won't reconnect; close the session client-side.
                    await self._get_websocket().close()
                    return
            # "all_processed", "info" and "Done" carry nothing the service needs.

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send text to Krisp for synthesis.

        Args:
            text: The text to synthesize.
            context_id: The audio context the resulting audio belongs to.

        Yields:
            None on success (audio arrives on the receive task), or an
            ``ErrorFrame`` and ``TTSStoppedFrame`` on failure.
        """
        if not self._websocket or self._websocket.state is State.CLOSED:
            await self._connect()

        # Counted before sending: the receive task can run while the send yields.
        connection_id = self._connection_id
        self._pending_texts += 1
        try:
            await self.send_with_retry(
                json.dumps({"type": "text", "text": text}), self._report_error
            )
            if not self._websocket:
                raise ConnectionError("unable to reconnect")
            if self._connection_id != connection_id:
                # Reconnected during the send: only the resent text is in flight.
                self._pending_texts = 1
            await self.start_tts_usage_metrics(text)
        except Exception as e:
            if self._connection_id == connection_id:
                self._pending_texts = max(0, self._pending_texts - 1)
            yield ErrorFrame(error=f"Krisp TTS send failed: {e}")
            yield TTSStoppedFrame(context_id=context_id)
            return
        yield None
