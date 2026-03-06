# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License

"""Gradium Text-to-Speech service implementation."""

import base64
import json
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings, _warn_deprecated_param
from pipecat.services.tts_service import AudioContextTTSService
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets import ConnectionClosedOK
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Gradium, you need to `pip install pipecat-ai[gradium]`.")
    raise Exception(f"Missing module: {e}")

SAMPLE_RATE = 48000


@dataclass
class GradiumTTSSettings(TTSSettings):
    """Settings for the Gradium TTS service."""

    pass


class GradiumTTSService(AudioContextTTSService):
    """Text-to-Speech service using Gradium's websocket API."""

    _settings: GradiumTTSSettings

    class InputParams(BaseModel):
        """Configuration parameters for Gradium TTS service.

        .. deprecated:: 0.0.105
            Use ``GradiumTTSSettings`` directly via the ``settings`` parameter instead.

        Parameters:
            temp: Temperature to be used for generation, defaults to 0.6.
        """

        temp: Optional[float] = 0.6

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: Optional[str] = None,
        url: str = "wss://eu.api.gradium.ai/api/speech/tts",
        model: Optional[str] = None,
        json_config: Optional[str] = None,
        params: Optional[InputParams] = None,
        settings: Optional[GradiumTTSSettings] = None,
        **kwargs,
    ):
        """Initialize the Gradium TTS service.

        Args:
            api_key: Gradium API key for authentication.
            voice_id: the voice identifier.

                .. deprecated:: 0.0.105
                    Use ``settings=GradiumTTSSettings(voice=...)`` instead.

            url: Gradium websocket API endpoint.
            model: Model ID to use for synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=GradiumTTSSettings(model=...)`` instead.

            json_config: Optional JSON configuration string for additional model settings.
            params: Additional configuration parameters.

                .. deprecated:: 0.0.105
                    Use ``settings=GradiumTTSSettings(...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to parent class.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = GradiumTTSSettings(
            model="default",
            voice="YTpq7expH9539ERJ",
            language=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            _warn_deprecated_param("model", GradiumTTSSettings, "model")
            default_settings.model = model
        if voice_id is not None:
            _warn_deprecated_param("voice_id", GradiumTTSSettings, "voice")
            default_settings.voice = voice_id

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            _warn_deprecated_param("params", GradiumTTSSettings)
            # Note: params.temp has no corresponding settings field

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_stop_frames=True,
            push_text_frames=False,
            pause_frame_processing=True,
            supports_word_timestamps=True,
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

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Gradium service supports metrics generation.
        """
        return True

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if voice changed.

        Args:
            delta: A :class:`TTSSettings` (or ``GradiumTTSSettings``) delta.

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

    def _build_msg(self, text: str = "") -> dict:
        """Build JSON message for Gradium API."""
        msg = {"text": text, "type": "text"}
        context_id = self.get_active_audio_context_id()
        if context_id:
            msg["client_req_id"] = context_id
        return msg

    async def start(self, frame: StartFrame):
        """Start the service and establish websocket connection.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
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
            self._websocket = await websocket_connect(self._url, additional_headers=headers)

            setup_msg = {
                "type": "setup",
                "output_format": "pcm",
                "voice_id": self._settings.voice,
                "close_ws_on_eos": False,
            }
            if self._json_config is not None:
                setup_msg["json_config"] = self._json_config
            await self._websocket.send(json.dumps(setup_msg))
            ready_msg = await self._websocket.recv()
            ready_msg = json.loads(ready_msg)
            if ready_msg["type"] == "error":
                raise Exception(f"received error {ready_msg['message']}")
            if ready_msg["type"] != "ready":
                raise Exception(f"unexpected first message type {ready_msg['type']}")

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
            await self.remove_active_audio_context()
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get active websocket connection or raise exception."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def flush_audio(self):
        """Flush any pending audio synthesis."""
        context_id = self.get_active_audio_context_id()
        if not context_id or not self._websocket:
            return
        try:
            msg = {"type": "end_of_stream", "client_req_id": context_id}
            await self._websocket.send(json.dumps(msg))
            self.reset_active_audio_context()
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
        await self.stop_all_metrics()

    async def on_audio_context_completed(self, context_id: str):
        """Called after an audio context has finished playing all of its audio.

        No close message is needed: Gradium signals completion with an
        ``end_of_stream`` message (handled in ``_receive_messages``), after
        which the server-side context is already closed.
        """
        pass

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
                await self.stop_ttfb_metrics()
                await self.start_word_timestamps()
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

            elif msg["type"] == "end_of_stream":
                if ctx_id and self.audio_context_available(ctx_id):
                    await self.add_word_timestamps([("TTSStoppedFrame", 0), ("Reset", 0)], ctx_id)
                    await self.remove_audio_context(ctx_id)
                await self.stop_all_metrics()

            elif msg["type"] == "error":
                await self.push_frame(TTSStoppedFrame(context_id=ctx_id))
                await self.stop_all_metrics()
                await self.push_error(error_msg=f"Error: {msg.get('message', msg)}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Gradium's streaming API.

        Args:
            text: The text to convert to speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                self._websocket = None
                await self._connect()

            try:
                if not self.has_active_audio_context():
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame(context_id=context_id)
                    await self.create_audio_context(context_id)

                msg = self._build_msg(text=text)
                await self._get_websocket().send(json.dumps(msg))
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
