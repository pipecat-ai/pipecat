# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License

"""Gradium Text-to-Speech service implementation."""

import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pydantic import BaseModel
from websockets import ConnectionClosedOK
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, WebsocketTTSService
from pipecat.utils.deprecation import deprecated
from pipecat.utils.tracing.service_decorators import traced_tts

SAMPLE_RATE = 48000


@dataclass
class GradiumTTSSettings(TTSSettings):
    """Settings for GradiumTTSService."""

    pass


class GradiumTTSService(WebsocketTTSService):
    """Text-to-Speech service using Gradium's websocket API.

    By default, LLM tokens are streamed to Gradium as they arrive
    (``TextAggregationMode.TOKEN``). Gradium treats the end of each ``text``
    message as a word boundary, so the service only sends whole words: the
    trailing partial word of each token is held back until the following text
    completes it or the turn ends.
    """

    Settings = GradiumTTSSettings
    _settings: Settings

    @deprecated(
        "`GradiumTTSService.InputParams` is deprecated since 0.0.105 and will be removed in "
        "2.0.0. Use `GradiumTTSService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Configuration parameters for Gradium TTS service.

        .. deprecated:: 0.0.105
            Use ``GradiumTTSService.Settings`` directly via the ``settings`` parameter instead.
            Will be removed in 2.0.0.

        Parameters:
            temp: Temperature to be used for generation, defaults to 0.6.
        """

        temp: float | None = 0.6

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str | None = None,
        url: str = "wss://api.gradium.ai/api/speech/tts",
        model: str | None = None,
        json_config: str | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        text_aggregation_mode: TextAggregationMode | None = None,
        aggregate_sentences: bool | None = None,
        **kwargs,
    ):
        """Initialize the Gradium TTS service.

        Args:
            api_key: Gradium API key for authentication.
            voice_id: the voice identifier.

                .. deprecated:: 0.0.105
                    Use ``settings=GradiumTTSService.Settings(voice=...)`` instead.
                    Will be removed in 2.0.0.

            url: Gradium websocket API endpoint.
            model: Model ID to use for synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=GradiumTTSService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            json_config: Optional JSON configuration string for additional model settings.
            params: Additional configuration parameters.

                .. deprecated:: 0.0.105
                    Use ``settings=GradiumTTSService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            text_aggregation_mode: How to aggregate incoming text before synthesis.
                Defaults to ``TextAggregationMode.TOKEN``, streaming LLM tokens to
                Gradium as whole words for the lowest latency.
            aggregate_sentences: Whether to aggregate sentences before synthesis.

                .. deprecated:: 0.0.104
                    Use ``text_aggregation_mode`` instead. Will be removed in 2.0.0.

            **kwargs: Additional arguments passed to parent class.
        """
        if text_aggregation_mode is None and aggregate_sentences is None:
            text_aggregation_mode = TextAggregationMode.TOKEN

        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="default",
            voice="_6Aslh2DxfmnRLmP",
            language=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            # Note: params.temp has no corresponding settings field

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            text_aggregation_mode=text_aggregation_mode,
            aggregate_sentences=aggregate_sentences,
            push_stop_frames=True,
            push_start_frame=True,
            push_text_frames=False,
            pause_frame_processing=True,
            sample_rate=SAMPLE_RATE,
            settings=default_settings,
            **kwargs,
        )

        # Store service configuration
        self._api_key = api_key
        self._url = url
        self._json_config = json_config

        # State tracking
        self._receive_task = None
        self._setup_context_ids: set[str] = set()
        # Per context, the trailing partial word not yet sent when streaming tokens.
        self._pending_words: dict[str, str] = {}

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Gradium service supports metrics generation.
        """
        return True

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if voice changed.

        Args:
            delta: A :class:`TTSSettings` (or ``GradiumTTSService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if "voice" in changed:
            await self._disconnect()
            await self._connect()
        else:
            self._warn_unhandled_updated_settings(changed)
        return changed

    def _build_setup_msg(self, context_id: str) -> dict:
        """Build setup message for Gradium API.

        Args:
            context_id: Context ID to use as ``client_req_id``.
        """
        setup_msg: dict[str, Any] = {
            "type": "setup",
            "output_format": "pcm",
            "voice_id": self._settings.voice,
            "close_ws_on_eos": False,
            "client_req_id": context_id,
        }
        if self._json_config is not None:
            setup_msg["json_config"] = self._json_config
        return setup_msg

    def _build_text_msg(self, text: str = "", context_id: str = "") -> dict:
        """Build text message for Gradium API."""
        msg = {"text": text, "type": "text", "client_req_id": context_id}
        return msg

    def _take_whole_words(self, text: str, context_id: str) -> str:
        """Return the text that can be sent now without splitting a word.

        When streaming tokens, the text after the last whitespace may be the
        start of a word the next token continues, so it is kept for the next
        call or for :meth:`flush_audio`.
        """
        if not self._is_streaming_tokens:
            return text
        text = self._pending_words.pop(context_id, "") + text
        end = max((i + 1 for i, c in enumerate(text) if c.isspace()), default=0)
        if end < len(text):
            self._pending_words[context_id] = text[end:]
        return text[:end]

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close connection.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel current operation and clean up.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        """Establish websocket connection and start receive task."""
        await super()._connect()

        logger.debug(f"{self}: connecting")

        # If the server disconnected, cancel the receive-task so that it can be reset below.
        if self._websocket is None or self._websocket.state is not State.OPEN:
            if self._receive_task:
                await self.cancel_task(self._receive_task)
                self._receive_task = None

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            logger.debug(f"{self}: setting receive task")
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Close websocket connection and clean up tasks."""
        await super()._disconnect()

        logger.debug(f"{self}: disconnecting")
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Connect to Gradium websocket API with configured settings."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            headers = {"x-api-key": self._api_key, "x-api-source": "pipecat"}
            self._websocket = await self._websocket_connect(self._url, additional_headers=headers)

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close websocket connection and reset state."""
        try:
            await self.stop_all_metrics()
            if self._websocket:
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._websocket = None
            self._setup_context_ids.clear()
            self._pending_words.clear()
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get active websocket connection or raise exception."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def flush_audio(self, context_id: str | None = None):
        """Flush any pending audio synthesis."""
        flush_id = context_id or self.get_active_audio_context_id()
        if not flush_id or not self._websocket:
            return
        try:
            if rest := self._pending_words.pop(flush_id, ""):
                await self._websocket.send(json.dumps(self._build_text_msg(rest, flush_id)))
            msg = {"type": "end_of_stream", "client_req_id": flush_id}
            await self._websocket.send(json.dumps(msg))
        except ConnectionClosedOK:
            logger.debug(f"{self}: connection closed normally during flush")
        except Exception as e:
            logger.error(f"{self} exception: {e}")

    async def on_audio_context_interrupted(self, context_id: str):
        """Called when an audio context is cancelled due to an interruption.

        No WebSocket message is needed — audio from the interrupted
        ``client_req_id`` will be silently dropped by the base class once the
        audio context no longer exists.
        """
        self._pending_words.pop(context_id, None)
        await self.stop_all_metrics()
        await super().on_audio_context_interrupted(context_id)

    async def on_audio_context_completed(self, context_id: str):
        """Called after an audio context has finished playing all of its audio.

        No close message is needed: Gradium signals completion with an
        ``end_of_stream`` message (handled in ``_receive_messages``), after
        which the server-side context is already closed.
        """
        await super().on_audio_context_completed(context_id)

    async def _receive_messages(self):
        """Process incoming websocket messages, demultiplexing by client_req_id."""
        # TODO(laurent): This should not be necessary as it should happen when
        # receiving the messages but this does not seem to always be the case
        # and that may lead to a busy polling loop.
        if self._websocket and self._websocket.state is State.CLOSED:
            raise ConnectionClosedOK(None, None)
        async for message in self._get_websocket():
            msg = json.loads(message)
            ctx_id = msg.get("client_req_id")

            if msg["type"] == "audio":
                if not ctx_id or not self.audio_context_available(ctx_id):
                    continue
                frame = TTSAudioRawFrame(
                    audio=base64.b64decode(msg["audio"]),
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=ctx_id,
                )
                await self.append_to_audio_context(ctx_id, frame)

            elif msg["type"] == "text":
                if ctx_id and self.audio_context_available(ctx_id):
                    await self.add_word_timestamps([(msg["text"], msg["start_s"])], ctx_id)

            elif msg["type"] == "ready":
                pass

            elif msg["type"] == "end_of_stream":
                if ctx_id and self.audio_context_available(ctx_id):
                    await self.append_to_audio_context(ctx_id, TTSStoppedFrame(context_id=ctx_id))
                    await self.remove_audio_context(ctx_id)
                if ctx_id:
                    self._setup_context_ids.discard(ctx_id)
                await self.stop_all_metrics()

            elif msg["type"] == "error":
                await self.push_frame(TTSStoppedFrame(context_id=ctx_id))
                await self.stop_all_metrics()
                await self.push_error(error_msg=f"Error: {msg.get('message', msg)}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Gradium's streaming API.

        Args:
            text: The text to convert to speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                self._websocket = None
                await self._connect()

            try:
                ws = self._get_websocket()
                if context_id not in self._setup_context_ids:
                    await ws.send(json.dumps(self._build_setup_msg(context_id)))
                    self._setup_context_ids.add(context_id)
                if words := self._take_whole_words(text, context_id):
                    msg = self._build_text_msg(text=words, context_id=context_id)
                    await ws.send(json.dumps(msg))
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
