#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from dataclasses import asdict, dataclass
from typing import Any, AsyncGenerator, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from loguru import logger
from speechmatics.rt import (
    AsyncClient,
    AudioEncoding,
    AudioFormat,
    ConnectionConfig,
    OperatingPoint,
    ServerMessageType,
    TranscriptionConfig,
    TranscriptResult,
    __version__,
)
from speechmatics.rt._models import ConversationConfig, SpeakerDiarizationConfig

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language


class SpeechmaticsSTTService(STTService):
    """Speechmatics STT service implementation.

    This service provides real-time speech-to-text transcription using the Speechmatics API.
    It supports partial and final transcriptions, multiple languages, various audio formats,
    and speaker diarization.

    Args:
        api_key: Speechmatics API key for authentication.
        language: Language code for transcription.
            Defaults to `en`.
        base_url: Base URL for Speechmatics API.
            Defaults to `eu2.rt.speechmatics.com`.
        enable_partials: Enable partial transcription results.
            Defaults to `True`.
        max_delay: Maximum delay for transcription in seconds.
            Defaults to `2.0`.
        sample_rate: Audio sample rate in Hz.
            Defaults to `16000`.
        chunk_size: Audio chunk size for streaming.
            Defaults to `256`.
        audio_encoding: Audio encoding format.
            Defaults to `pcm_s16le`.
        end_of_utterance_silence_trigger: Silence duration in seconds to trigger end of utterance detection.
            Defaults to `None`.
        operating_point: Operating point for transcription accuracy vs. latency tradeoff.
            Defaults to `enhanced`.
        enable_speaker_diarization: Enable speaker diarization to identify different speakers.
            Defaults to `False`.
        max_speakers: Maximum number of speakers to detect.
            Defaults to `None` (auto-detect).
        transcription_config: Custom transcription configuration (other set parameters are merged).
            Defaults to `None`.
        **kwargs: Additional arguments passed to STTService.
    """

    def __init__(
        self,
        *,
        api_key: str,
        language: Language = Language.EN,
        base_url: str = "eu2.rt.speechmatics.com",
        enable_partials: bool = True,
        max_delay: float = 2.0,
        sample_rate: Optional[int] = 16000,
        chunk_size: int = 256,
        audio_encoding: AudioEncoding = AudioEncoding.PCM_S16LE,
        end_of_utterance_silence_trigger: Optional[float] = None,
        operating_point: OperatingPoint = OperatingPoint.ENHANCED,
        enable_speaker_diarization: bool = False,
        max_speakers: Optional[int] = None,
        transcription_config: Optional[TranscriptionConfig] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        # Client configuration
        self._api_key: str = api_key
        self._language: Language = language
        self._base_url: str = base_url
        self._enable_partials: bool = enable_partials
        self._max_delay: float = max_delay
        self._sample_rate: int = sample_rate
        self._chunk_size: int = chunk_size
        self._audio_encoding: AudioEncoding = audio_encoding
        self._end_of_utterance_silence_trigger: Optional[float] = end_of_utterance_silence_trigger
        self._operating_point: OperatingPoint = operating_point
        self._enable_speaker_diarization: bool = enable_speaker_diarization
        self._max_speakers: Optional[int] = max_speakers

        # Complete configuration objects
        self._transcription_config: TranscriptionConfig = None
        self._process_config(transcription_config)

        # STT client
        self._client: AsyncClient = None
        self._audio_buffer: AudioBuffer = AudioBuffer(maxsize=10)

    async def start(self, frame: StartFrame):
        """Called when the new session starts."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Called when the session ends."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Called when the session is cancelled."""
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Adds audio to the audio buffer and yields None."""
        self._audio_buffer.write_audio(audio)
        yield None

    async def _run_client(self):
        """Runs the Speechmatics client in a thread."""
        await self._client.transcribe(
            self._audio_buffer,
            transcription_config=self._transcription_config,
            audio_format=AudioFormat(
                encoding=self._audio_encoding,
                sample_rate=self.sample_rate,
                chunk_size=self._chunk_size,
            ),
        )

    async def _connect(self):
        """Connect to the STT service."""
        # Create new STT RT client
        self._client = AsyncClient(
            api_key=self._api_key,
            url=_get_endpoint_url(self._base_url),
        )

        # Recognition started event
        @self._client.on(ServerMessageType.RECOGNITION_STARTED)
        def on_recognition_started(message: dict[str, Any]):
            logger.debug(f"Recognition started (session: {message.get('id')})")

        # Partial transcript event
        @self._client.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT)
        def on_partial_transcript(message: dict[str, Any]):
            asyncio.create_task(self._handle_transcript(message, is_final=False))

        # Final transcript event
        @self._client.on(ServerMessageType.ADD_TRANSCRIPT)
        def on_final_transcript(message: dict[str, Any]):
            asyncio.create_task(self._handle_transcript(message, is_final=True))

        # End of Utterance
        @self._client.on(ServerMessageType.END_OF_UTTERANCE)
        def on_end_of_utterance(message: dict[str, Any]):
            logger.debug("End of utterance received from STT")

        # Start the client in a thread
        asyncio.create_task(self._run_client())

    async def _disconnect(self):
        """Disconnect from the STT service."""
        # Stop the audio buffer
        self._audio_buffer.stop()

        # Disconnect the client
        if self._client:
            await self._client.close()

    def _process_config(self, transcription_config: Optional[TranscriptionConfig] = None) -> None:
        """Create a formatted STT transcription config.

        This takes an optional TranscriptionConfig object and populates it with the
        values from the STT service. Individual parameters take priority over those
        within the config object.

        Args:
            transcription_config: Optional transcription config to use.
        """
        # Transcription config
        if not transcription_config:
            transcription_config = TranscriptionConfig(
                language=self._language,
                operating_point=self._operating_point,
                diarization="speaker" if self._enable_speaker_diarization else None,
                enable_partials=self._enable_partials,
                max_delay=self._max_delay or 2.0,
            )
        else:
            if self._language:
                transcription_config.language = self._language
            if self._operating_point:
                transcription_config.operating_point = self._operating_point
            if self._enable_speaker_diarization:
                transcription_config.diarization = "speaker"
            if self._enable_partials:
                transcription_config.enable_partials = self._enable_partials
            if self._max_delay:
                transcription_config.max_delay = self._max_delay

        # Diarization
        if self._enable_speaker_diarization and self._max_speakers:
            transcription_config.speaker_diarization_config = SpeakerDiarizationConfig(
                max_speakers=self._max_speakers,
            )

        # End of Utterance
        if self._end_of_utterance_silence_trigger:
            transcription_config.conversation_config = ConversationConfig(
                end_of_utterance_silence_trigger=self._end_of_utterance_silence_trigger,
            )

        # Set config
        self._transcription_config = transcription_config

    async def _handle_transcript(self, message: dict[str, Any], is_final: bool = False):
        """Handle the partial and final transcript events."""
        logger.debug(f"transcript: {message}")


class AudioBuffer:
    """Audio buffer for STT clients."""

    def __init__(self, maxsize: int = 0):
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._current_chunk = b""
        self._position = 0
        self._closed = False

    def write_audio(self, data: bytes) -> None:
        """Write audio data to the buffer (thread-safe)."""
        if data:
            try:
                self._queue.put_nowait(data)
            except asyncio.QueueFull:
                pass

    async def read(self, size: int) -> bytes:
        """Read exactly `size` bytes from the buffer."""
        result = b""
        bytes_needed = size

        while bytes_needed > 0 and not self._closed:
            # Use data from current chunk if available
            if self._position < len(self._current_chunk):
                available = len(self._current_chunk) - self._position
                take = min(bytes_needed, available)
                result += self._current_chunk[self._position : self._position + take]
                self._position += take
                bytes_needed -= take
                continue

            # Get next chunk
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                if chunk is None:
                    continue
                self._current_chunk = chunk
                self._position = 0
            except asyncio.TimeoutError:
                await asyncio.sleep(0)
                continue

        return result

    def stop(self):
        """Close the audio buffer."""
        self._closed = True


@dataclass
class SpeechFragment:
    """Fragment of an utterance.

    Attributes:
        start_time: Start time of the fragment in seconds (from session start).
        end_time: End time of the fragment in seconds (from session start).
        language: Language of the fragment.
            Defaults to `en`.
        is_eos: Whether the fragment is the end of a sentence.
            Defaults to `False`.
        is_final: Whether the fragment is the final fragment.
            Defaults to `False`.
        attaches_to: Whether the fragment attaches to the previous or next fragment (punctuation).
            Defaults to empty string.
        content: Content of the fragment.
            Defaults to empty string.
        speaker: Speaker of the fragment (if diarization is enabled).
            Defaults to `None`.
        confidence: Confidence of the fragment (0.0 to 1.0).
            Defaults to `1.0`.
    """

    start_time: float
    end_time: float
    language: str = "en"
    is_eos: bool = False
    is_final: bool = False
    attaches_to: str = ""
    content: str = ""
    speaker: Optional[str] = None
    confidence: float = 1.0


def _get_endpoint_url(url: str) -> str:
    """Format the endpoint URL with the SDK and app versions."""
    url_path = f"wss://{url}/v2"

    query_params = dict()
    query_params["sm-app"] = f"pipecat/{__version__}"
    query = urlencode(query_params)

    return f"{url_path}?{query}"
