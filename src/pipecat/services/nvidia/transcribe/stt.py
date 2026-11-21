#
# Copyright (c) 2026, Daily
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""HTTP-only STT for NVIDIA Nemotron 3.5 Transcribe NIM.

Transcribe whole VAD segments into final transcripts without partial results.
Use pure VAD turn ends with ``SpeechTimeoutUserTurnStopStrategy`` and official
NIM transcription prompt templates.

NIM guide: https://docs.nvidia.com/nim/speech/latest/asr/deploy-asr-models/index.html
Model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B
"""

import base64
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from typing import cast

from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionContentPartParam

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.whisper.base_stt import language_to_whisper_language
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

# Provisional until measured against a same-host NIM.
NVIDIA_TRANSCRIBE_TTFS_P99: float = 2.5


@dataclass
class NvidiaTranscribeSTTSettings(STTSettings):
    """Runtime-updatable settings for NvidiaTranscribeSTTService.

    Parameters:
        prompt: Optional transcription instruction sent next to the audio.
            None sends audio only. Use the official NIM prompt templates.
    """

    prompt: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class NvidiaTranscribeSTTService(SegmentedSTTService):
    """Transcribe whole VAD segments with NVIDIA Nemotron 3.5 Transcribe NIM.

    This HTTP-only service returns final transcripts without partial results.
    Configure pure VAD turn ends with ``SpeechTimeoutUserTurnStopStrategy``.
    """

    Settings = NvidiaTranscribeSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 10.0,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = NVIDIA_TRANSCRIBE_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Transcribe NIM service.

        Args:
            base_url: Self-hosted NIM URL including ``/v1``.
            api_key: Optional authentication key for the deployment.
            timeout: HTTP request timeout in seconds.
            settings: Runtime-updatable model, language and prompt settings.
            ttfs_p99_latency: P99 speech-end-to-final-transcript latency in seconds.
            **kwargs: Additional arguments passed to SegmentedSTTService.
        """
        default_settings = self.Settings(
            model="nvidia-nemotron-3p5-transcribe",
            language=None,
            prompt=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            settings=default_settings,
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )
        self._client = AsyncOpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url,
            max_retries=2,
            timeout=timeout,
        )

    def can_generate_metrics(self) -> bool:
        """Whether this service supports processing metrics.

        Returns:
            True, as this service supports processing metrics.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Pipecat language to its base ISO code.

        Args:
            language: Pipecat language value.

        Returns:
            The base ISO language code.
        """
        return language_to_whisper_language(language)

    async def cleanup(self):
        """Release the service and its HTTP client."""
        try:
            await super().cleanup()
        finally:
            await self._client.close()

    @staticmethod
    def contextual_prompt(terms: Sequence[str], domain: str = "general") -> str:
        """Return the official contextual template for English audio only.

        Args:
            terms: Expected words or phrases. Keep lists to about 20 terms.
            domain: Subject area of the audio.

        Returns:
            The official contextual transcription prompt.
        """
        return (
            f"This audio belongs to the {domain} field and may contain the following words or phrases: "
            f"{', '.join(terms)}. Transcribe the English audio into text, "
            "ensuring all punctuation marks are included."
        )

    @traced_stt
    async def _handle_transcription(
        self,
        transcript: str,
        is_final: bool,
        language: Language | None = None,
    ):
        """Handle a transcription result with tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe one WAV segment, yielding a final transcript or an error.

        Args:
            audio: Audio segment in WAV format.

        Yields:
            A TranscriptionFrame, or an ErrorFrame if the request fails.
        """
        try:
            language = cast("Language | None", assert_given(self._settings.language))
            content: list[ChatCompletionContentPartParam] = [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(audio).decode("ascii"),
                        "format": "wav",
                    },
                }
            ]
            if self._settings.prompt:
                content.append({"type": "text", "text": assert_given(self._settings.prompt)})

            await self.start_processing_metrics()
            try:
                response = await self._client.chat.completions.create(
                    model=cast(str, self._settings.model),
                    messages=[{"role": "user", "content": content}],
                    stream=False,
                    extra_body={"language": language} if language else None,
                )
            finally:
                await self.stop_processing_metrics()

            text = (response.choices[0].message.content or "").strip()
            if not text:
                logger.debug(f"{self}: Received empty transcription from NIM")
                return

            await self._handle_transcription(text, True, language)
            yield TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=response,
            )
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"{self} error: {e}")
