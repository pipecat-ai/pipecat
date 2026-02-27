#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Smallest AI speech-to-text service implementations.

This module provides two STT services using Smallest AI's Waves API:

- ``SmallestSTTService``: HTTP-based segmented STT. Buffers audio during speech,
  sends as a single request once the user stops speaking (VAD-triggered).
- ``SmallestRealtimeSTTService``: WebSocket-based real-time STT. Streams audio
  continuously and receives interim/final transcripts with low latency.
"""

import asyncio
import io
import json
from enum import Enum
from typing import AsyncGenerator, Optional
from urllib.parse import urlencode

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_latency import SMALLEST_TTFS_P99
from pipecat.services.stt_service import SegmentedSTTService, WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    import httpx
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Smallest, you need to `pip install pipecat-ai[smallest]`.")
    raise Exception(f"Missing module: {e}")

try:
    import numpy as np
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Smallest, you need to `pip install pipecat-ai[smallest]`.")
    raise Exception(f"Missing module: {e}")

try:
    import soundfile as sf
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Smallest, you need to `pip install pipecat-ai[smallest]`.")
    raise Exception(f"Missing module: {e}")

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Smallest, you need to `pip install pipecat-ai[smallest]`.")
    raise Exception(f"Missing module: {e}")


def language_to_smallest_language(language: Language) -> Optional[str]:
    """Convert a Language enum to Smallest's language code format.

    Smallest AI currently supports English and Hindi. Falls back to extracting
    the base language code if the exact Language enum isn't mapped.

    Args:
        language: The Language enum value to convert.

    Returns:
        The Smallest language code string, or None if unsupported.
    """
    BASE_LANGUAGES = {
        Language.EN: "en",
        Language.HI: "hi",
    }

    result = BASE_LANGUAGES.get(language)

    if not result:
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result


class SmallestSTTModel(str, Enum):
    """Available Smallest AI STT models."""

    PULSE = "pulse"


class SmallestSTTService(SegmentedSTTService):
    """Smallest AI speech-to-text service using the Waves HTTP API.

    This is a segmented STT service that buffers audio while the user speaks
    (using VAD) and sends the complete audio segment to Smallest AI's HTTP
    endpoint for transcription once the user stops speaking.

    Requires VAD to be enabled in the pipeline.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Smallest STT service.

        Parameters:
            language: Language code for transcription. Defaults to "en".
            age_detection: Enable age detection. Defaults to False.
            emotion_detection: Enable emotion detection. Defaults to False.
            gender_detection: Enable gender detection. Defaults to False.
        """

        language: str = "en"
        age_detection: bool = False
        emotion_detection: bool = False
        gender_detection: bool = False

    def __init__(
        self,
        *,
        api_key: str,
        model: str = SmallestSTTModel.PULSE,
        url: str = "https://api.smallest.ai/waves/v1/pulse/get_text",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        ttfs_p99_latency: Optional[float] = SMALLEST_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Smallest AI STT service.

        Args:
            api_key: Smallest AI API key for authentication.
            model: Model to use for transcription. Defaults to "pulse".
            url: API endpoint URL. Defaults to the Smallest Waves API endpoint.
            sample_rate: Audio sample rate. If None, will be determined from the
                start frame.
            params: Configuration parameters for the STT service.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment.
            **kwargs: Additional arguments passed to the parent SegmentedSTTService.
        """
        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )

        params = params or SmallestSTTService.InputParams()

        self._api_key = api_key
        self._url = url
        self._language = params.language

        model_str = model.value if isinstance(model, Enum) else model
        self.set_model_name(model_str)

        self._client = httpx.AsyncClient()
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        self._payload = {
            "model": model_str,
            "age_detection": "true" if params.age_detection else "false",
            "gender_detection": "true" if params.gender_detection else "false",
            "emotion_detection": "true" if params.emotion_detection else "false",
            "language": params.language,
        }

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Smallest STT supports metrics generation.
        """
        return True

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing.

        This method is decorated with @traced_stt for observability.
        The actual work (pushing frames) is done in run_stt; this method
        exists solely as a tracing hook.
        """
        pass

    def _audio_bytes_to_wav_buffer(self, audio: bytes) -> io.BytesIO:
        """Convert raw PCM16 audio bytes to a WAV-formatted buffer.

        The Smallest API expects WAV-formatted audio. This converts raw signed
        16-bit PCM audio bytes into a WAV buffer with proper headers.

        Args:
            audio: Raw PCM16 audio bytes.

        Returns:
            A BytesIO buffer containing WAV-formatted audio data.
        """
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_float, self.sample_rate, format="WAV", subtype="PCM_16")
        wav_buffer.seek(0)
        return wav_buffer

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe audio using the Smallest AI HTTP API.

        Called by the base SegmentedSTTService when the user stops speaking.
        The audio parameter contains the complete WAV-encoded speech segment.

        Args:
            audio: WAV-encoded audio bytes from the speech segment.

        Yields:
            TranscriptionFrame on success, ErrorFrame on failure.
        """
        wav_buffer = self._audio_bytes_to_wav_buffer(audio)

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        try:
            response = await self._client.post(
                self._url,
                headers=self._headers,
                content=wav_buffer.getvalue(),
                params=self._payload,
            )
            response.raise_for_status()
            result = response.json()
            text: str = result.get("transcription", "").strip()
        except httpx.HTTPStatusError as e:
            logger.error(f"{self} API error: {e.response.status_code} - {e.response.text}")
            yield ErrorFrame(error=f"Smallest API error: {e.response.status_code}", exception=e)
            return
        except Exception as e:
            logger.exception(f"{self} transcription error: {type(e).__name__}: {e}")
            yield ErrorFrame(error=f"Smallest transcription error: {type(e).__name__}: {e}")
            return

        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

        if text:
            logger.debug(f"Transcription: [{text}]")
            await self._handle_transcription(text, True, self._language)
            yield TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
            )

    async def cleanup(self):
        """Clean up resources used by the Smallest STT service."""
        await super().cleanup()
        await self._client.aclose()


class SmallestRealtimeSTTService(WebsocketSTTService):
    """Smallest AI real-time speech-to-text service using the Pulse WebSocket API.

    Streams audio continuously over a WebSocket connection and receives
    interim and final transcription results with low latency. Best suited
    for real-time voice applications where immediate feedback is needed.

    Uses Pipecat's VAD to detect when the user stops speaking and sends
    a finalize message to flush the final transcript.

    Example::

        stt = SmallestRealtimeSTTService(
            api_key="your-api-key",
            params=SmallestRealtimeSTTService.InputParams(
                language="en",
                word_timestamps=True,
            ),
        )
    """

    class InputParams(BaseModel):
        """Configuration parameters for Smallest Realtime STT service.

        Parameters:
            language: Language code for transcription. Use "multi" for auto-detection.
                Defaults to "en".
            encoding: Audio encoding format. Defaults to "linear16".
            word_timestamps: Include word-level timestamps. Defaults to False.
            full_transcript: Include cumulative transcript. Defaults to False.
            sentence_timestamps: Include sentence-level timestamps. Defaults to False.
            redact_pii: Redact personally identifiable information. Defaults to False.
            redact_pci: Redact payment card information. Defaults to False.
            numerals: Convert spoken numerals to digits. Defaults to "auto".
            diarize: Enable speaker diarization. Defaults to False.
        """

        language: str = "en"
        encoding: str = "linear16"
        word_timestamps: bool = False
        full_transcript: bool = False
        sentence_timestamps: bool = False
        redact_pii: bool = False
        redact_pci: bool = False
        numerals: str = "auto"
        diarize: bool = False

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://api.smallest.ai",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        ttfs_p99_latency: Optional[float] = SMALLEST_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Smallest AI Realtime STT service.

        Args:
            api_key: Smallest AI API key for authentication.
            base_url: Base WebSocket URL for the Smallest API.
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline's rate.
            params: Configuration parameters for the STT service.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
            **kwargs: Additional arguments passed to WebsocketSTTService.
        """
        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            keepalive_timeout=10,
            keepalive_interval=5,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._params = params or SmallestRealtimeSTTService.InputParams()
        self._receive_task = None
        self._connected_event = asyncio.Event()
        self._connected_event.set()

        self.set_model_name("pulse")

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Start the service and connect to the WebSocket."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and disconnect from the WebSocket."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and disconnect from the WebSocket."""
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, handling VAD events for finalization."""
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self.start_processing_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._websocket and self._websocket.state is State.OPEN:
                try:
                    await self._websocket.send(json.dumps({"type": "finalize"}))
                except Exception as e:
                    logger.warning(f"{self} failed to send finalize: {e}")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio to the Smallest Pulse WebSocket for transcription.

        Args:
            audio: Raw PCM audio bytes.

        Yields:
            None -- transcription results arrive via WebSocket messages.
        """
        await self._connected_event.wait()

        if not self._websocket or self._websocket.state is State.CLOSED:
            await self._connect()

        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(audio)
            except Exception as e:
                yield ErrorFrame(error=f"Smallest Realtime STT error: {e}")

        yield None

    async def _connect(self):
        self._connected_event.clear()
        try:
            await self._connect_websocket()
            await super()._connect()

            if self._websocket and not self._receive_task:
                self._receive_task = self.create_task(
                    self._receive_task_handler(self._report_error)
                )
        finally:
            self._connected_event.set()

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish WebSocket connection to the Smallest Pulse STT API."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Smallest Realtime STT")

            query_params = {
                "language": self._params.language,
                "encoding": self._params.encoding,
                "sample_rate": str(self.sample_rate),
                "word_timestamps": str(self._params.word_timestamps).lower(),
                "full_transcript": str(self._params.full_transcript).lower(),
                "sentence_timestamps": str(self._params.sentence_timestamps).lower(),
                "redact_pii": str(self._params.redact_pii).lower(),
                "redact_pci": str(self._params.redact_pci).lower(),
                "numerals": self._params.numerals,
                "diarize": str(self._params.diarize).lower(),
            }

            ws_url = f"{self._base_url}/waves/v1/pulse/get_text?{urlencode(query_params)}"

            self._websocket = await websocket_connect(
                ws_url,
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
            )
            await self._call_event_handler("on_connected")
            logger.debug("Connected to Smallest Realtime STT")
        except Exception as e:
            await self.push_error(
                error_msg=f"Smallest Realtime STT connection error: {e}", exception=e
            )
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close the WebSocket connection."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug("Disconnecting from Smallest Realtime STT")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Receive and process messages from the Smallest Pulse WebSocket."""
        async for message in self._get_websocket():
            try:
                data = json.loads(message)
                await self._process_response(data)
            except json.JSONDecodeError:
                logger.warning(f"{self} received non-JSON message: {message}")
            except Exception as e:
                logger.error(f"{self} error processing message: {e}")

    async def _process_response(self, data: dict):
        """Process a transcription response from the Pulse API.

        Args:
            data: Parsed JSON response containing transcript data.
        """
        is_final = data.get("is_final", False)
        text = data.get("transcript", "").strip()

        if not text:
            return

        if is_final:
            await self.stop_processing_metrics()
            logger.debug(f"Smallest final transcript: [{text}]")
            await self._handle_transcription(text, True, data.get("language"))
            await self.push_frame(
                TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    data.get("language"),
                    result=data,
                )
            )
        else:
            logger.trace(f"Smallest interim transcript: [{text}]")
            await self.push_frame(
                InterimTranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    data.get("language"),
                    result=data,
                )
            )

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        """Handle a transcription result with tracing."""
        pass
