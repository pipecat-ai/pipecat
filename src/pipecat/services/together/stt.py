#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Together AI speech-to-text service implementation."""

import asyncio
import base64
import json
from typing import AsyncGenerator, Optional

from loguru import logger
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
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt


class TogetherSTTService(STTService):
    """Together AI speech-to-text service.

    Provides real-time speech recognition using Together AI's WebSocket API
    with OpenAI-compatible speech-to-text endpoints.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "openai/whisper-large-v3",
        language: Language = Language.EN,
        sample_rate: int = 16000,
        base_url: str = "wss://api.together.xyz/v1",
        **kwargs,
    ):
        """Initialize the Together AI STT service.

        Args:
            api_key: Together AI API key for authentication.
            model: The model to use for transcription.
            language: Language for transcription (default: English).
            sample_rate: Audio sample rate (default: 16000). Together AI requires 16kHz input.
            url: The URL of the Together AI WebSocket API.
            **kwargs: Additional arguments passed to the parent STTService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._language = language.value if isinstance(language, Language) else language
        self._base_url = base_url
        self._connection = None
        self._receive_task = None
        self._language = language

        self.set_model_name(model)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Together STT service supports metrics generation.
        """
        return True

    async def set_model(self, model: str):
        """Set the Together AI model and reconnect.

        Args:
            model: The Together AI model name to use.
        """
        await super().set_model(model)
        logger.info(f"Switching STT model to: [{model}]")
        self._model = model
        await self._disconnect()
        await self._connect()

    async def set_language(self, language: Language):
        """Set the recognition language.

        Args:
            language: The language to use for speech recognition.
        """
        logger.info(f"Switching STT language to: [{language}]")
        self._language = language.value if isinstance(language, Language) else language
        
        if self._connection and self._connection.state is State.OPEN:
            try:
                update_language_msg = {"type": "transcription_session.updated", "session": {"language": self._language}}
                await self._connection.send(json.dumps(update_language_msg))
            except Exception as e:
                logger.error(f"{self} error sending language update: {e}")
                await self.push_error(ErrorFrame(f"Error sending language update: {e}"))
        yield None

    async def start(self, frame: StartFrame):
        """Start the Together AI STT service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Together AI STT service.

        Args:
            frame: The end frame.
        """
        await self._disconnect()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the Together AI STT service.

        Args:
            frame: The cancel frame.
        """
        await self._disconnect()
        await super().cancel(frame)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio data to Together AI for transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: None (transcription results come via WebSocket callbacks).
        """
        if self._connection and self._connection.state is State.OPEN:
            try:
                audio_b64 = base64.b64encode(audio).decode("utf-8")
                audio_msg = {"type": "input_audio_buffer.append", "audio": audio_b64}
                await self._connection.send(json.dumps(audio_msg))
            except Exception as e:
                logger.error(f"{self} error sending audio: {e}")
                await self.push_error(ErrorFrame(f"Error sending audio: {e}"))
        yield None

    async def _connect(self):
        """Connect to Together AI WebSocket."""
        logger.debug("Connecting to Together AI")

        try:
            url = f"{self._base_url}/realtime?intent=transcription&model={self._model}&input_audio_format=pcm_s16le_16000&language={self._language}"
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "OpenAI-Beta": "realtime=v1",
            }

            self._connection = await websocket_connect(url, additional_headers=headers)
            await self._call_event_handler("on_connected")

            # Start receiving messages
            if not self._receive_task or self._receive_task.done():
                self._receive_task = asyncio.create_task(self._receive_messages())

        except Exception as e:
            logger.error(f"{self} connection error: {e}")
            await self._call_event_handler("on_connection_error", str(e))
            await self.push_error(ErrorFrame(f"Connection error: {e}"))
            self._connection = None

    async def _disconnect(self):
        """Disconnect from Together AI WebSocket."""
        if self._connection:

            # Cancel and cleanup receive task
            if self._receive_task and not self._receive_task.done():
                self._receive_task.cancel()
                try:
                    await asyncio.wait_for(self._receive_task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                except Exception as e:
                    logger.debug(f"{self} error waiting for receive task: {e}")
            self._receive_task = None

            # Close connection
            try:
                if self._connection.state is State.OPEN:
                    await asyncio.wait_for(self._connection.close(), timeout=1.0)
                await self._call_event_handler("on_disconnected")
            except Exception as e:
                logger.debug(f"{self} error closing connection: {e}")

            self._connection = None

    async def start_metrics(self):
        """Start TTFB and processing metrics collection."""
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

    async def _receive_messages(self):
        """Receive messages from Together AI WebSocket."""
        try:
            async for message in self._connection:
                logger.debug(f"{self} received message: {message}")
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
            await self.push_error(ErrorFrame(f"Receive error: {e}"))

    async def _handle_message(self, data: dict):
        """Handle messages from Together AI WebSocket."""
        msg_type = data.get("type", "")

        if msg_type == "session.created":
            logger.debug(f"{self} session created")

        elif msg_type == "conversation.item.input_audio_transcription.delta":
            # Interim transcription
            delta = data.get("transcript", "")
            if delta.strip():
                await self.stop_ttfb_metrics()
                await self.push_frame(
                    InterimTranscriptionFrame(
                        delta,
                        self._user_id,
                        time_now_iso8601(),
                        Language(self._language),
                    )
                )

        elif msg_type == "conversation.item.input_audio_transcription.completed":
            # Final transcription
            transcript = data.get("transcript", "").strip()
            if transcript:
                await self.stop_ttfb_metrics()
                await self.push_frame(
                    TranscriptionFrame(
                        transcript,
                        self._user_id,
                        time_now_iso8601(),
                        Language(self._language),
                    )
                )
                await self._handle_transcription(
                    transcript, True, Language(self._language)
                )
                await self.stop_processing_metrics()

        elif msg_type == "conversation.item.input_audio_transcription.failed":
            error_info = data.get("error", {})
            logger.error(f"{self} transcription failed: {error_info}")
            await self.push_error(ErrorFrame(f"Transcription failed: {error_info}"))

        elif msg_type == "error":
            error_info = data.get("error", {})
            logger.error(f"{self} API error: {error_info}")
            await self.push_error(ErrorFrame(f"API error: {error_info}"))

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with Together AI-specific handling."""
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            # Start metrics when user starts speaking
            await self.start_metrics()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Signal end of speech to Together AI
            if self._connection and self._connection.state is State.OPEN:
                try:
                    commit_msg = {"type": "input_audio_buffer.commit"}
                    await self._connection.send(json.dumps(commit_msg))
                except Exception as e:
                    logger.error(f"{self} error sending commit: {e}")