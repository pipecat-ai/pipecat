#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Together AI text-to-speech service implementation."""

import asyncio
import base64
import json
from typing import AsyncGenerator, List, Optional, Union

from loguru import logger
from pydantic import BaseModel
try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Together, you need to `pip install pipecat-ai[together]`.")
    raise Exception(f"Missing module: {e}")

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts


class TogetherTTSService(TTSService):
    """Together AI TTS service with WebSocket streaming.

    Provides text-to-speech using Together AI's realtime WebSocket API.
    Supports streaming synthesis with configurable voice and model options.
    """

    class InputParams(BaseModel):
        """Input parameters for Together TTS configuration.

        Parameters:
            language: Language to use for synthesis.
            voice: Voice to use for synthesis.
            max_partial_length: Maximum partial text length before forcing TTS.
        """

        language: Optional[Language] = Language.EN
        voice: Optional[str] = "tara"
        max_partial_length: Optional[int] = None

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "canopylabs/orpheus-3b-0.1-ft",
        url: str = "wss://api.together.ai/v1/audio/speech/websocket",
        sample_rate: Optional[int] = 24000,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Together AI TTS service.

        Args:
            api_key: Together AI API key for authentication.
            model: TTS model to use.
            url: WebSocket URL for Together AI TTS API.
            sample_rate: Audio sample rate (default: 24000).
            params: Additional input parameters for voice customization.
            **kwargs: Additional arguments passed to the parent service.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or TogetherTTSService.InputParams()

        self._api_key = api_key
        self._model = model
        self._url = url
        self._voice = params.voice or "tara"
        self._language = params.language.value if params.language else "en"
        self._max_partial_length = params.max_partial_length
        self._websocket = None
        self._session_id = None
        self._receive_task = None

        self.set_model_name(model)
        super().set_voice(self._voice)

        

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Together TTS service supports metrics generation.
        """
        return True

    async def set_model(self, model: str):
        """Set the TTS model.

        Args:
            model: The model name to use for synthesis.
        """
        self._model = model
        await super().set_model(model)
        logger.info(f"Switching TTS model to: [{model}]")

    async def set_voice(self, voice: str):
        """Set the voice.

        Args:
            voice: The voice name to use for synthesis.
        """
        self._voice = voice
        await super().set_voice(voice)
        logger.info(f"Switching TTS voice to: [{voice}]")
        
        # Send voice update to WebSocket if connected
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                voice_update_msg = {
                    "type": "tts_session.updated",
                    "session": {"voice": voice}
                }
                await self._websocket.send(json.dumps(voice_update_msg))
                logger.info(f"Sent voice update to WebSocket: {voice}")
            except Exception as e:
                logger.error(f"Error sending voice update: {e}")

    def _build_websocket_url(self) -> str:
        """Build the WebSocket URL with query parameters."""
        url = f"{self._url}?model={self._model}&voice={self._voice}"
        if self._max_partial_length is not None:
            url += f"&max_partial_length={self._max_partial_length}"
        return url

    async def start(self, frame: StartFrame):
        """Start the Together AI TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Together AI TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Together AI TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        """Connect to Together AI WebSocket."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            ws_url = self._build_websocket_url()
            logger.debug(f"Connecting to Together AI TTS: {ws_url}")

            headers = {"Authorization": f"Bearer {self._api_key}"}
            self._websocket = await websocket_connect(
                ws_url, additional_headers=headers
            )
            await self._call_event_handler("on_connected")

            # Start receiving messages
            if not self._receive_task or self._receive_task.done():
                self._receive_task = asyncio.create_task(self._receive_messages())

            # Ensure voice is set on the server side
            try:
                voice_update_msg = {
                    "type": "tts_session.updated",
                    "session": {"voice": self._voice}
                }
                await self._websocket.send(json.dumps(voice_update_msg))
                logger.debug(f"Sent initial voice setting to WebSocket: {self._voice}")
            except Exception as e:
                logger.error(f"Error sending initial voice setting: {e}")

            logger.debug("Connected to Together AI TTS")

        except Exception as e:
            logger.error(f"{self} connection error: {e}")
            await self._call_event_handler("on_connection_error", str(e))
            await self.push_error(ErrorFrame(f"Connection error: {e}"))
            self._websocket = None

    async def _disconnect(self):
        """Disconnect from Together AI WebSocket."""
        try:
            if self._receive_task and not self._receive_task.done():
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass
            self._receive_task = None

            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug("Disconnecting from Together AI TTS")
                await self._websocket.close()
                await self._call_event_handler("on_disconnected")
        except Exception as e:
            logger.error(f"{self} error during disconnect: {e}")
        finally:
            self._websocket = None
            self._session_id = None

    async def _receive_messages(self):
        """Receive messages from Together AI WebSocket."""
        try:
            async for message in self._websocket:
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        await self._handle_message(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"{self} JSON decode error: {e}")
                        continue
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"{self} receive error: {e}")
            await self._call_event_handler("on_connection_error", str(e))

    async def _handle_message(self, data: dict):
        """Handle messages from Together AI WebSocket.

        Args:
            data: The parsed JSON message data.
        """
        msg_type = data.get("type", "")

        if msg_type == "session.created":
            session = data.get("session", {})
            self._session_id = session.get("id")
            logger.debug(f"{self} session created: {self._session_id}")
            
        elif msg_type == "session.updated":
            session = data.get("session", {})
            if "voice" in session:
                updated_voice = session.get("voice")
                logger.debug(f"{self} voice updated to: {updated_voice}")

        elif msg_type == "conversation.item.input_text.received":
            text = data.get("text", "")
            logger.debug(
                f"{self} text received: {text[:50]}{'...' if len(text) > 50 else ''}"
            )

        elif msg_type == "conversation.item.audio_output.delta":
            item_id = data.get("item_id")
            delta = data.get("delta")
            if delta:
                try:
                    # Decode base64 audio chunk
                    audio_chunk = base64.b64decode(delta)
                    frame = TTSAudioRawFrame(
                        audio=audio_chunk,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                    )
                    await self.push_frame(frame)
                except Exception as e:
                    logger.error(f"{self} error processing audio delta: {e}")

        elif msg_type == "conversation.item.audio_output.done":
            item_id = data.get("item_id")
            logger.debug(f"{self} audio generation complete for: {item_id}")
            await self.stop_ttfb_metrics()
            await self.push_frame(TTSStoppedFrame())

        elif msg_type == "conversation.item.tts.failed":
            error = data.get("error", {})
            logger.error(f"{self} TTS error: {error}")
            await self.push_error(ErrorFrame(f"TTS error: {error}"))
            await self.push_frame(TTSStoppedFrame())

        elif msg_type == "error":
            error = data.get("error", {})
            logger.error(f"{self} API error: {error}")
            await self.push_error(ErrorFrame(f"API error: {error}"))

        else:
            logger.debug(f"{self} unhandled message type: {msg_type}")

    async def _handle_interruption(
        self, frame: InterruptionFrame, direction: FrameDirection
    ):
        """Handle interruption by canceling current generation.

        Args:
            frame: The interruption frame.
            direction: Frame processing direction.
        """
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()

        # For Together AI, we can send a simple clear buffer message
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                # Send clear buffer message
                clear_msg = {"type": "input_text_buffer.clear"}
                await self._websocket.send(json.dumps(clear_msg))
            except Exception as e:
                logger.error(f"{self} error sending clear buffer: {e}")

    async def flush_audio(self):
        """Flush any pending audio and finalize the current context."""
        if not self._websocket or self._websocket.state is not State.OPEN:
            return

        logger.trace(f"{self}: flushing audio")
        try:
            commit_msg = {"type": "input_text_buffer.commit"}
            await self._websocket.send(json.dumps(commit_msg))
        except Exception as e:
            logger.error(f"{self} error flushing audio: {e}")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Together AI's streaming API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is not State.OPEN:
                await self._connect()
                if not self._websocket or self._websocket.state is not State.OPEN:
                    logger.error(f"{self} failed to connect to WebSocket")
                    yield TTSStoppedFrame()
                    return

            # Start TTS metrics
            await self.start_ttfb_metrics()
            yield TTSStartedFrame()

            # Send text to be synthesized
            text_msg = {"type": "input_text_buffer.append", "text": text}

            try:
                await self._websocket.send(json.dumps(text_msg))

                # Commit the text to trigger TTS processing
                commit_msg = {"type": "input_text_buffer.commit"}
                await self._websocket.send(json.dumps(commit_msg))

                await self.start_tts_usage_metrics(text)
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame()
                await self._disconnect()
                await self._connect()
                return

            yield None

        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error(ErrorFrame(f"Error generating TTS: {e}"))
            yield TTSStoppedFrame()
