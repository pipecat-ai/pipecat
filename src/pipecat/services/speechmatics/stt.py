#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Optional

from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language

from .types import (
    AudioSettings,
    RTConversationConfig,
    RTSpeakerDiarizationConfig,
    TranscriptionConfig,
)


class SpeechmaticsSTTService(STTService):
    """Speechmatics STT service implementation.

    This service provides real-time speech-to-text transcription using the Speechmatics API.
    It supports partial and final transcriptions, multiple languages, various audio formats,
    and speaker diarization.

    Args:
        api_key: Speechmatics API key for authentication.
        language: Language code for transcription (default: Language.EN).
        base_url: Base URL for Speechmatics API (default: eu2.rt.speechmatics.com).
        enable_partials: Enable partial transcription results (default: True).
        max_delay: Maximum delay for transcription in seconds (default: 5).
        sample_rate: Audio sample rate in Hz (default: None, inferred from pipeline).
        chunk_size: Audio chunk size for streaming (default: 256).
        audio_encoding: Audio encoding format (default: "pcm_f32le").
        end_of_utterance_silence_trigger: Silence duration in seconds to trigger end of utterance detection (default: None, disabled).
        operating_point: Operating point for transcription accuracy vs. latency tradeoff (default: "enhanced").
        enable_speaker_diarization: Enable speaker diarization to identify different speakers (default: False).
        max_speakers: Maximum number of speakers to detect (default: None, auto-detect).
        transcription_config: Custom transcription configuration.
        **kwargs: Additional arguments passed to STTService.
    """

    def __init__(
        self,
        *,
        api_key: str,
        language: Language = Language.EN,
        base_url: str = "preview.rt.speechmatics.com",
        enable_partials: bool = True,
        max_delay: float = 1.5,
        sample_rate: Optional[int] = None,
        chunk_size: int = 256,
        audio_encoding: str = "pcm_s16le",
        end_of_utterance_silence_trigger: Optional[float] = None,
        operating_point: str = "enhanced",
        enable_speaker_diarization: bool = False,
        max_speakers: Optional[int] = None,
        transcription_config: Optional[TranscriptionConfig] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._language = language
        self._base_url = base_url
        self._enable_partials = enable_partials
        self._max_delay = max_delay
        self._chunk_size = chunk_size
        self._audio_encoding = audio_encoding
        self._end_of_utterance_silence_trigger = end_of_utterance_silence_trigger
        self._operating_point = operating_point
        self._enable_speaker_diarization = enable_speaker_diarization
        self._max_speakers = max_speakers

        self._transcription_config: TranscriptionConfig = None
        self._audio_settings: AudioSettings = None

        self._process_config(transcription_config)
        
        print(self._transcription_config)
        

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Returns transcript as a string."""
        pass

    def _process_config(self, transcription_config: Optional[TranscriptionConfig] = None) -> None:
        """Create a formatted STT transcription config."""
        # Transcription config
        transcription_config = TranscriptionConfig(
            language=self._language,
            operating_point=self._operating_point,
            diarization="speaker" if self._enable_speaker_diarization else None,
            enable_partials=self._enable_partials,
            max_delay=self._max_delay or 2.0,
            **transcription_config,
        )

        # Diarization
        if self._enable_speaker_diarization and self._max_speakers:
            transcription_config.speaker_diarization_config = RTSpeakerDiarizationConfig(
                max_speakers=self._max_speakers,
            )

        # End of Utterance
        if self._end_of_utterance_silence_trigger:
            transcription_config.conversation_config = RTConversationConfig(
                end_of_utterance_silence_trigger=self._end_of_utterance_silence_trigger,
            )

        # Audio settings
        audio_settings = AudioSettings(
            encoding=self._audio_encoding, sample_rate=self.sample_rate, chunk_size=self._chunk_size
        )
