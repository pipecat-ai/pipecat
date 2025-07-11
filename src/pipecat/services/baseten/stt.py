import asyncio
import json
import urllib.parse
from typing import AsyncGenerator, Optional

import websockets
from websockets.protocol import State
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
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


class BasetenLiveOptions:
    """Configuration options for Baseten Live STT service.

    Manages transcription parameters including model selection, language,
    audio encoding format, and sample rate settings.
    """

    def __init__(
        self,
        *,
        model: str = "whisper-v3",
        language: str = "en",
        encoding: str = "pcm_s16le",
        sample_rate: int = 16000,
        **kwargs,
    ):
        """Initialize BasetenLiveOptions with default or provided parameters.

        Args:
            model: The transcription model to use. Defaults to "ink-whisper".
            language: Target language for transcription. Defaults to English.
            encoding: Audio encoding format. Defaults to "pcm_s16le".
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            **kwargs: Additional parameters for the transcription service.
        """
        self.model = model
        self.language = language
        self.encoding = encoding
        self.sample_rate = sample_rate
        self.additional_params = kwargs

    def to_dict(self):
        """Convert options to dictionary format.

        Returns:
            Dictionary containing all configuration parameters.
        """
        params = {
            "model": self.model,
            "language": self.language if isinstance(self.language, str) else self.language.value,
            "encoding": self.encoding,
            "sample_rate": str(self.sample_rate),
        }

        return params

    def items(self):
        """Get configuration items as key-value pairs.

        Returns:
            Iterator of (key, value) tuples for all configuration parameters.
        """
        return self.to_dict().items()

    def get(self, key, default=None):
        """Get a configuration value by key.

        Args:
            key: The configuration parameter name to retrieve.
            default: Default value if key is not found.

        Returns:
            The configuration value or default if not found.
        """
        if hasattr(self, key):
            return getattr(self, key)
        return self.additional_params.get(key, default)

    @classmethod
    def from_json(cls, json_str: str) -> "BasetenLiveOptions":
        """Create options from JSON string.

        Args:
            json_str: JSON string containing configuration parameters.

        Returns:
            New BasetenLiveOptions instance with parsed parameters.
        """
        return cls(**json.loads(json_str))


class BasetenSTTService(STTService):
    """Speech-to-text service using Baseten Live API.

    Provides real-time speech transcription through WebSocket connection
    to Baseten's Live transcription service.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "",
        encoding: str = "pcm_s16le",
        sample_rate: int = 16000,
        language: str = "en",
        options: dict = {},
        live_options: Optional[BasetenLiveOptions] = None,
        **kwargs,
    ):
        """Initialize BasetenSTTService with API key and options.

        Args:
            api_key: Authentication key for Baseten API.
            base_url: Custom API endpoint URL.
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            live_options: Configuration options for transcription service.
            **kwargs: Additional arguments passed to parent STTService.
        """
        sample_rate = sample_rate or (live_options.sample_rate if live_options else None)
        super().__init__(sample_rate=sample_rate, **kwargs)

        default_options = BasetenLiveOptions(
            model="whisper-v3",
            language="en",
            encoding="pcm_s16le",
            sample_rate=sample_rate,
        )

        merged_options = default_options
        if live_options:
            merged_options_dict = default_options.to_dict()
            merged_options_dict.update(live_options.to_dict())
            merged_options = BasetenLiveOptions(
                **{
                    k: v
                    for k, v in merged_options_dict.items()
                    if not isinstance(v, str) or v != "None"
                }
            )

        self._settings = merged_options
        self.set_model_name(merged_options.model)
        self._api_key = api_key
        self._base_url = base_url
        self._options = options
        self._encoding = encoding
        self._sample_rate = sample_rate
        self._language = language
        self._audio_buffer = bytearray()
        self._chunk_size_bytes = 1024
        self._connection = None
        self._receiver_task = None

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics.

        Returns:
            True, indicating metrics are supported.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the STT service and establish connection.

        Args:
            frame: Frame indicating service should start.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the STT service and close connection.

        Args:
            frame: Frame indicating service should stop.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service and close connection.

        Args:
            frame: Frame indicating service should be cancelled.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio data for speech-to-text transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            None - transcription results are handled via WebSocket responses.
        """
        # If the connection is closed, due to timeout, we need to reconnect when the user starts speaking again
        if not self._connection or self._connection.state is State.CLOSED:
            await self._connect()


        self._audio_buffer.extend(audio)

        while len(self._audio_buffer) >= self._chunk_size_bytes:
            chunk = bytes(self._audio_buffer[ :self._chunk_size_bytes])
            self._audio_buffer = self._audio_buffer[self._chunk_size_bytes: ]
            await self._connection.send(chunk)

        yield None


    async def _connect(self):
        params = self._settings.to_dict()
        ws_url = self._base_url
        logger.debug(f"Connecting to Baseten: {ws_url}")
        headers = {"Authorization": f"Api-Key {self._api_key}"}

        # Build and send the metadata payload as the first message
        metadata = {
            "streaming_vad_config": {
                "threshold": self._options["threshold"],
                "min_silence_duration_ms": self._options["min_silence_duration_ms"],
                "speech_pad_ms": self._options["speech_pad_ms"],
            },
            "streaming_params": {
                "encoding": self._encoding,
                "sample_rate": self._sample_rate,
                "enable_partial_transcripts": False
            },
            "whisper_params": {"audio_language": self._language},
        }

        try:
            self._connection = await websockets.connect(ws_url, extra_headers=headers)
            await self._connection.send(json.dumps(metadata))
            # Setup the receiver task to handle the incoming messages from the Baseten server
            if self._receiver_task is None or self._receiver_task.done():
                self._receiver_task = asyncio.create_task(self._receive_messages())
            logger.debug(f"Connected to Baseten")
        except Exception as e:
            logger.error(f"{self}: unable to connect to Baseten: {e}")

    async def _receive_messages(self):
        try:
            while True:
                if not self._connection or self._connection.state is State.CLOSED:
                    break

                message = await self._connection.recv()
                print(message)
                try:
                    data = json.loads(message)
                    await self._process_response(data)
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON message: {message}")
                print(data)
        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosed as e:
            logger.debug(f"WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"Error in message receiver: {e}")

    async def _process_response(self, data):
        if "transcript" in data:
            await self._on_transcript(data)
        else:
            logger.error("NO TRANSCRIPT KEY")

        # elif data["type"] == "error":
        #     logger.error(f"Baseten error: {data.get('message', 'Unknown error')}")

    @traced_stt
    async def _handle_transcription(
        self, message: str, is_final: bool, language: Optional[Language] = None
    ):
        pass

    async def _on_transcript(self, data):
        if "transcript" not in data:
            logger.error("NO TRANSCRIPT")
            return

        transcript = data.get("transcript", "")
        is_final = data.get("is_final", False)
        language = None
        # logger.debug(f"Transcript: {transcript}")

        if "language" in data:
            try:
                language = Language(data["language"])
            except (ValueError, KeyError):
                pass

        if len(transcript) > 0:
            await self.stop_ttfb_metrics()

            if is_final:
                frame = TranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    language,
                )
                await self.push_frame(
                    frame
                )
                await self._handle_transcription(transcript, is_final, language)
                await self.stop_processing_metrics()
            else:
                frame = InterimTranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    language,
                )
                await self.push_frame(
                    frame
                )

    async def _disconnect(self):
        if self._receiver_task:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.exception(f"Unexpected exception while cancelling task: {e}")
            self._receiver_task = None

        if self._connection and self._connection.state is State.OPEN:
            logger.debug("Disconnecting from Baseten")

            await self._connection.close()
            self._connection = None

    async def start_metrics(self):
        """Start performance metrics collection for transcription processing."""
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle speech events.

        Args:
            frame: The frame to process.
            direction: Direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            await self.start_metrics()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Send finalize command to flush the transcription session
            if self._connection and self._connection.state is State.OPEN:
                await self._connection.send("finalize")