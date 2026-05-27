#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""SLNG text-to-speech service."""

import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven, is_given
from pipecat.services.tts_service import TTSService, WebsocketTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use SLNG TTS, you need to `pip install pipecat-ai[slng]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class SlngTTSSettings(TTSSettings):
    """Settings for SlngTTSService.

    Parameters:
        voice: Voice identifier for speech synthesis.
        language: Language for speech synthesis.
        speed: Speech speed multiplier. When not given, the server default is used.
    """

    speed: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class SlngTTSService(WebsocketTTSService):
    """Text-to-speech service using the SLNG bridge WebSocket API.

    Provides real-time speech synthesis through a persistent WebSocket
    connection to the SLNG TTS bridge. Audio is streamed back as base64-encoded
    ``audio_chunk`` messages and a ``flushed`` message signals end of speech.

    Protocol overview:

    1. Connect with ``Authorization: Bearer <api_key>`` header.
    2. Send ``init`` with voice and audio config immediately after connecting.
    3. Send ``text`` messages for each text chunk to synthesise.
    4. Send ``flush`` to signal end of the current utterance.
    5. Receive ``audio_chunk`` messages (base64 PCM) and a ``flushed`` message.
    """

    Settings = SlngTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "slng/deepgram/aura:2-en",
        voice: str | None = None,
        base_url: str = "api.slng.ai",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize SlngTTSService.

        Args:
            api_key: Authentication key for the SLNG API.
            model: The TTS model to use. Defaults to "slng/deepgram/aura:2-en".
            voice: Voice identifier for synthesis (e.g. "aura-2-thalia-en").
            base_url: The API host (without scheme) or a full WebSocket URL
                (e.g. "ws://localhost:8080" for testing). Defaults to "api.slng.ai".
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline sample rate.
            settings: Runtime-updatable settings override.
            **kwargs: Additional arguments passed to parent WebsocketTTSService.
        """
        default_settings = self.Settings(
            model=model,
            voice=voice,
            language=Language.EN,
            speed=NOT_GIVEN,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_stop_frames=False,
            push_start_frame=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url
        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics.

        Returns:
            True, indicating metrics are supported.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the TTS service and establish the WebSocket connection.

        Args:
            frame: Frame indicating service should start.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the TTS service and close the WebSocket connection.

        Args:
            frame: Frame indicating service should stop.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the TTS service and close the WebSocket connection.

        Args:
            frame: Frame indicating service should be cancelled.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish the WebSocket connection and send the init message."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            model = self._settings.model or "slng/deepgram/aura:2-en"
            logger.debug(f"Connecting to SLNG TTS ({model})")

            if "://" in self._base_url:
                ws_url = f"{self._base_url}/v1/bridges/unmute/tts/{model}"
            else:
                ws_url = f"wss://{self._base_url}/v1/bridges/unmute/tts/{model}"

            headers = {"Authorization": f"Bearer {self._api_key}"}
            self._websocket = await websocket_connect(ws_url, additional_headers=headers)

            # Build init config
            config: dict[str, Any] = {
                "sample_rate": self.sample_rate,
                "encoding": "linear16",
            }

            if is_given(self._settings.language) and self._settings.language is not None:
                config["language"] = str(self._settings.language)

            if is_given(self._settings.speed):
                config["speed"] = self._settings.speed

            init_msg: dict[str, Any] = {
                "type": "init",
                "config": config,
            }
            if self._settings.voice:
                init_msg["voice"] = self._settings.voice

            await self._websocket.send(json.dumps(init_msg))
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to SLNG TTS: {e}", exception=e)

    async def _disconnect_websocket(self):
        """Send a close message and shut down the WebSocket."""
        ws = self._websocket
        try:
            if ws and ws.state is State.OPEN:
                logger.debug("Disconnecting from SLNG TTS")
                await ws.send(json.dumps({"type": "close"}))
                await ws.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing SLNG TTS websocket: {e}", exception=e)
        finally:
            await self.stop_all_metrics()
            await self.remove_active_audio_context()
            if self._websocket is ws:
                self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("SLNG TTS websocket not connected")

    async def on_audio_context_interrupted(self, context_id: str):
        """Send a clear message to the server when the bot is interrupted.

        Args:
            context_id: The ID of the interrupted audio context.
        """
        await self.stop_all_metrics()
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(json.dumps({"type": "clear"}))
            except Exception as e:
                logger.warning(f"{self}: failed to send clear on interruption: {e}")
        await super().on_audio_context_interrupted(context_id)

    async def flush_audio(self, context_id: str | None = None):
        """Flush pending audio for the current utterance.

        Sends a ``flush`` message to the server, which will respond with a
        ``flushed`` message when all audio has been sent.

        Args:
            context_id: The specific context to flush. If None, falls back to
                the currently active context.
        """
        if not self._websocket or self._websocket.state is not State.OPEN:
            return
        logger.trace(f"{self}: flushing audio")
        try:
            await self._websocket.send(json.dumps({"type": "flush"}))
        except Exception as e:
            logger.warning(f"{self}: failed to send flush: {e}")

    async def _receive_messages(self):
        """Receive and dispatch incoming WebSocket messages."""
        async for message in self._get_websocket():
            try:
                data = json.loads(message)
                await self._process_message(data)
            except json.JSONDecodeError:
                logger.warning(f"{self}: received non-JSON message: {message!r}")
            except Exception as e:
                logger.error(f"{self}: error processing message: {e}")

    async def _process_message(self, data: dict[str, Any]):
        """Dispatch a decoded server message.

        Args:
            data: Decoded JSON payload from the server.
        """
        msg_type = data.get("type")

        if msg_type == "ready":
            session_id = data.get("session_id", "")
            logger.debug(f"{self}: SLNG TTS session ready (id={session_id})")

        elif msg_type == "audio_chunk":
            raw_audio = base64.b64decode(data["data"])
            ctx_id = self.get_active_audio_context_id()
            frame = TTSAudioRawFrame(
                audio=raw_audio,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=ctx_id,
            )
            await self.stop_ttfb_metrics()
            await self.append_to_audio_context(ctx_id, frame)

        elif msg_type == "flushed":
            ctx_id = self.get_active_audio_context_id()
            if ctx_id:
                await self.append_to_audio_context(ctx_id, TTSStoppedFrame(context_id=ctx_id))
                await self.remove_audio_context(ctx_id)

        elif msg_type in ("cleared", "segment_start", "segment_end", "audio_end"):
            pass  # No-op

        elif msg_type == "error":
            error_msg = data.get("message", "Unknown SLNG TTS error")
            logger.error(f"{self}: SLNG TTS error: {error_msg}")
            await self.push_error(error_msg=error_msg)
            await self.stop_all_metrics()

        else:
            logger.debug(f"{self}: unknown message type: {msg_type!r}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using the SLNG TTS API.

        Sends a ``text`` message over the WebSocket. Audio arrives asynchronously
        via the receive task as ``audio_chunk`` messages.

        Args:
            text: The text to synthesise into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            None — audio frames are delivered via the receive task.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is not State.OPEN:
                await self._connect()

            if not self._websocket:
                yield ErrorFrame(error="SLNG TTS websocket not connected")
                return

            try:
                await self._websocket.send(json.dumps({"type": "text", "text": text}))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"SLNG TTS send error: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return

            yield None

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")


# ---------------------------------------------------------------------------
# HTTP streaming TTS (voiceai-sdk)
# ---------------------------------------------------------------------------

try:
    from voiceai_sdk import AsyncSlng
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use SLNG, you need to `pip install pipecat-ai[slng]`.")
    raise Exception(f"Missing module: {e}")


class SlngHttpTTSService(TTSService):
    """SLNG HTTP streaming text-to-speech service.

    Sends text to SLNG's HTTP TTS endpoint via the voiceai-sdk and streams
    the binary audio response as ``TTSAudioRawFrame`` chunks. Simpler than
    the WebSocket variant — no persistent connection or flush/clear protocol.
    """

    Settings = SlngTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "slng/deepgram/aura:2-en",
        voice: str | None = None,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the SLNG HTTP TTS service.

        Args:
            api_key: SLNG API key for authentication.
            model: SLNG model variant (e.g. ``"slng/deepgram/aura:2-en"``).
            voice: Voice identifier (e.g. ``"aura-2-thalia-en"``).
            sample_rate: Audio sample rate in Hz. If None, uses service default.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        default_settings = self.Settings(
            model=model,
            voice=voice,
            language=Language.EN,
            speed=NOT_GIVEN,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._client = AsyncSlng(api_key=api_key)

    def can_generate_metrics(self) -> bool:
        """Return True — SLNG HTTP TTS supports metrics generation."""
        return True

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Stream audio from SLNG HTTP TTS endpoint.

        Args:
            text: Text to synthesize.
            context_id: Audio context ID for tracking frames.

        Yields:
            TTSAudioRawFrame: Audio chunks from the streaming response.
            ErrorFrame: On HTTP or network errors.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            await self.start_ttfb_metrics()
            kwargs = {}
            if self._settings.voice:
                kwargs["voice"] = self._settings.voice

            async with self._client.text_to_speech.with_streaming_response.create(
                self._settings.model,
                text=text,
                **kwargs,
            ) as response:
                await self.start_tts_usage_metrics(text)
                async for chunk in response.iter_bytes(chunk_size=self.chunk_size):
                    if chunk:
                        yield TTSAudioRawFrame(
                            audio=chunk,
                            sample_rate=self.sample_rate,
                            num_channels=1,
                            context_id=context_id,
                        )
        except Exception as e:
            yield ErrorFrame(error=f"SLNG HTTP TTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()
