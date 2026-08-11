#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Inworld AI speech-to-text service implementation."""

import base64
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from loguru import logger
from pydantic import ValidationError

from pipecat import version as pipecat_version
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.inworld.frames import InworldVoiceProfile, InworldVoiceProfileFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

USER_AGENT = f"pipecat/{pipecat_version()}"


def language_to_inworld_stt_language(language: Language) -> str:
    """Convert a language enum to an Inworld STT language code.

    Args:
        language: The language to convert.

    Returns:
        The corresponding ISO 639 language code. Regional variants fall back
        to their base language code.
    """
    language_map = {
        Language.AR: "ar",
        Language.CS: "cs",
        Language.DA: "da",
        Language.DE: "de",
        Language.EL: "el",
        Language.EN: "en",
        Language.ES: "es",
        Language.FA: "fa",
        Language.FI: "fi",
        Language.FIL: "fil",
        Language.FR: "fr",
        Language.HI: "hi",
        Language.HU: "hu",
        Language.ID: "id",
        Language.IT: "it",
        Language.JA: "ja",
        Language.KO: "ko",
        Language.MK: "mk",
        Language.MS: "ms",
        Language.NL: "nl",
        Language.PL: "pl",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SV: "sv",
        Language.TH: "th",
        Language.TL: "fil",
        Language.TR: "tr",
        Language.VI: "vi",
        Language.YUE: "yue",
        Language.ZH: "zh",
    }
    return resolve_language(language, language_map, use_base_code=True)


@dataclass
class InworldSTTSettings(STTSettings):
    """Settings for :class:`InworldSTTService`.

    Parameters:
        prompts: Terms that bias recognition toward names, jargon, and acronyms.
        enable_voice_profile: Whether to analyze speaker age, emotion, pitch,
            vocal style, and accent. See https://docs.inworld.ai/stt/voice-profiles
        voice_profile_top_n: Maximum labels returned for each Voice Profile category.
    """

    prompts: list[str] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    enable_voice_profile: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    voice_profile_top_n: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class InworldSTTService(SegmentedSTTService):
    """Speech-to-text service using Inworld's synchronous transcription API.

    The service buffers utterances according to Pipecat VAD events, sends each
    utterance to Inworld as a WAV file, and emits one final transcription frame.
    When Voice Profile analysis is enabled, it also emits an
    :class:`InworldVoiceProfileFrame` before the transcription.
    """

    Settings = InworldSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "https://api.inworld.ai",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = None,
        **kwargs,
    ):
        """Initialize the Inworld STT service.

        Args:
            api_key: Inworld API key containing Base64 credentials.
            aiohttp_session: aiohttp client session for HTTP requests.
            base_url: Base URL for the Inworld API.
            sample_rate: Audio sample rate in Hz. If not provided, uses the pipeline's rate.
            settings: Runtime-updatable model, language, and recognition prompts.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to :class:`SegmentedSTTService`.
        """
        default_settings = self.Settings(
            model="inworld/inworld-stt-1",
            language=None,
            prompts=[],
            enable_voice_profile=False,
            voice_profile_top_n=10,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            settings=default_settings,
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        """Check whether the service can generate processing metrics.

        Returns:
            True, as Inworld STT supports processing metrics.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a language enum to Inworld's STT language format.

        Args:
            language: The language to convert.

        Returns:
            The Inworld ISO 639 language code.
        """
        return language_to_inworld_stt_language(language)

    def _request_payload(self, audio: bytes) -> dict[str, Any]:
        """Build a transcription request payload.

        Args:
            audio: WAV-encoded audio bytes.

        Returns:
            The Inworld transcription request payload.

        Raises:
            ValueError: If no model is configured.
        """
        model = assert_given(self._settings.model)
        if not model:
            raise ValueError("Inworld STT model must be specified")

        config: dict[str, Any] = {
            "modelId": model,
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": self.sample_rate,
            "numberOfChannels": 1,
        }

        language_setting = assert_given(self._settings.language)
        if language_setting:
            config["language"] = str(language_setting)

        prompts = assert_given(self._settings.prompts)
        if prompts:
            config["prompts"] = prompts

        enable_voice_profile = assert_given(self._settings.enable_voice_profile)
        if enable_voice_profile:
            voice_profile_config: dict[str, Any] = {"enableVoiceProfile": True}
            top_n = assert_given(self._settings.voice_profile_top_n)
            if top_n is not None:
                if top_n < 1:
                    raise ValueError("Inworld Voice Profile top_n must be at least 1")
                voice_profile_config["topN"] = top_n
            config["voiceProfileConfig"] = voice_profile_config

        return {
            "transcribeConfig": config,
            "audioData": {"content": base64.b64encode(audio).decode("ascii")},
        }

    async def _transcribe(self, audio: bytes) -> dict[str, Any]:
        """Send one WAV utterance to Inworld.

        Args:
            audio: WAV-encoded audio bytes.

        Returns:
            The decoded Inworld response.

        Raises:
            RuntimeError: If Inworld returns an unsuccessful status.
        """
        headers = {
            "Authorization": f"Basic {self._api_key}",
            "Content-Type": "application/json",
            "X-Request-Id": str(uuid.uuid4()),
            "X-User-Agent": USER_AGENT,
        }
        async with self._session.post(
            f"{self._base_url}/stt/v1/transcribe",
            json=self._request_payload(audio),
            headers=headers,
        ) as response:
            if not 200 <= response.status < 300:
                error_text = await response.text()
                raise RuntimeError(f"Inworld API error ({response.status}): {error_text}")
            return await response.json()

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: str | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe a WAV utterance with Inworld.

        Args:
            audio: WAV-encoded audio bytes produced by :class:`SegmentedSTTService`.

        Yields:
            An optional Voice Profile frame followed by a transcription frame for
            non-empty text, or an error frame on failure.
        """
        await self.start_processing_metrics()
        try:
            result = await self._transcribe(audio)
            timestamp = time_now_iso8601()

            voice_profile_data = result.get("voiceProfile", result.get("voice_profile"))
            if voice_profile_data is not None:
                try:
                    voice_profile = InworldVoiceProfile.model_validate(voice_profile_data)
                    yield InworldVoiceProfileFrame(
                        user_id=self._user_id,
                        timestamp=timestamp,
                        voice_profile=voice_profile,
                    )
                except ValidationError as e:
                    yield ErrorFrame(error=f"Inworld Voice Profile error: {e}", exception=e)

            transcript = result.get("transcription", {}).get("transcript", "").strip()
            if not transcript:
                logger.debug("Inworld returned an empty transcription")
                return

            language_setting = assert_given(self._settings.language)
            language = str(language_setting) if language_setting else None
            await self._handle_transcription(transcript, True, language)
            logger.debug(f"Transcription: [{transcript}]")

            try:
                frame_language = Language(language) if language else None
            except ValueError:
                frame_language = None

            yield TranscriptionFrame(
                transcript,
                self._user_id,
                timestamp,
                frame_language,
                result=result,
            )
        except Exception as e:
            yield ErrorFrame(error=f"Inworld STT error: {e}", exception=e)
        finally:
            await self.stop_processing_metrics()
