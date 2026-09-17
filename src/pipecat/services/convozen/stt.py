#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ConvoZen Akshara speech-to-text service implementation."""

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import ClassVar

import aiohttp
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import CONVOZEN_TTFS_P99
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven

CONVOZEN_BASE_URL = "https://voice.convozen.ai/sdk-models/developer-api/api"

# Language hints Akshara accepts. The server rejects anything outside this set,
# so a language not listed here is sent without a hint rather than as one.
LANG_TAGS: set[str] = {"bn", "en", "gu", "hi", "kn", "ml", "mr", "ta", "te"}

_MAX_ERROR_BODY = 512


def language_to_convozen_language(language: Language) -> str | None:
    """Convert a Language enum to a ConvoZen language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The two-letter code Akshara accepts, or None if unsupported.
    """
    base = str(language.value).split("-")[0].lower()
    return base if base in LANG_TAGS else None


@dataclass
class ConvozenSTTSettings(STTSettings):
    """Settings for ConvozenSTTService.

    Parameters:
        lang_tags: Language hints, e.g. ``["hi", "en"]`` for code-mixed speech.
            Overrides the hint derived from ``language``.
        keywords: Domain words to bias recognition towards.
        blank_penalty: Penalizes blank/silence tokens; increase it to reduce empty
            gaps in the transcript. Unbounded — a raw penalty weight, not a
            normalized value. ``None`` lets the server derive one from the
            language hints.
        word_timestamps: Whether to request per-word timings in the response.
    """

    lang_tags: list[str] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    keywords: list[str] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    blank_penalty: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    word_timestamps: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class ConvozenSTTService(SegmentedSTTService):
    """ConvoZen Akshara speech-to-text service.

    Transcribes complete utterances through Akshara's REST API, covering nine
    Indian languages including code-mixed speech. Akshara returns one transcript
    per request and emits no interim results, so VAD must be enabled in the
    pipeline to segment speech into utterances.

    Example::

        stt = ConvozenSTTService(
            api_key=os.getenv("CONVOZEN_API_KEY"),
            aiohttp_session=session,
            settings=ConvozenSTTService.Settings(
                language=Language.HI,
                lang_tags=["hi", "en"],
            ),
        )
    """

    Settings: ClassVar[type[ConvozenSTTSettings]] = ConvozenSTTSettings
    _settings: ConvozenSTTSettings

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        model: str = "akshara-pro",
        base_url: str = CONVOZEN_BASE_URL,
        ttfs_p99_latency: float = CONVOZEN_TTFS_P99,
        settings: ConvozenSTTSettings | None = None,
        **kwargs,
    ):
        """Initialize the ConvoZen STT service.

        Args:
            api_key: ConvoZen API key for authentication.
            aiohttp_session: Shared aiohttp session for HTTP requests.
            model: Akshara model, ``"akshara-pro"`` (default) or ``"akshara"``.
            base_url: API base URL, for self-hosted deployments.
            ttfs_p99_latency: P99 latency from speech end to final transcript in
                seconds. Override for your deployment.
            settings: Runtime-updatable settings (language, hints, keywords).
            **kwargs: Additional arguments passed to SegmentedSTTService.
        """
        default_settings = self.Settings(
            model=model,
            language=Language.EN,
            lang_tags=None,
            keywords=None,
            blank_penalty=None,
            word_timestamps=False,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, ttfs_p99_latency=ttfs_p99_latency, **kwargs)

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        """Return whether this service can generate processing metrics.

        Returns:
            True — processing metrics are reported for each transcription.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to a ConvoZen language code.

        Args:
            language: The Language enum value to convert.

        Returns:
            The ConvoZen language code, or None if unsupported.
        """
        return language_to_convozen_language(language)

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Record a transcription result for tracing.

        Args:
            transcript: The recognized text.
            is_final: Whether this is a final transcript.
            language: The language of the transcript, if known.
        """
        pass

    def _resolve_lang_tags(self) -> list[str] | None:
        """Return the language hints to send, or None to send no hint."""
        tags = self._settings.lang_tags
        if tags and not isinstance(tags, NotGiven):
            return list(tags)

        language = self._settings.language
        if language and not isinstance(language, NotGiven):
            code = str(language).split("-")[0].lower()
            if code in LANG_TAGS:
                return [code]
        return None

    def _build_form(self, audio: bytes) -> aiohttp.FormData:
        """Build the multipart body for the transcribe endpoint."""
        form = aiohttp.FormData()
        form.add_field("file", audio, filename="audio.wav", content_type="audio/wav")
        form.add_field("model", str(self._settings.model))
        # A pipeline delivers one mixed track, so recognition is always mono.
        # Diarization is not exposed: it returns a different response shape.
        form.add_field("audio_channels", "mono")

        if tags := self._resolve_lang_tags():
            form.add_field("lang_tags", json.dumps(tags))

        keywords = self._settings.keywords
        if keywords and not isinstance(keywords, NotGiven):
            form.add_field("keywords", json.dumps(list(keywords)))

        penalty = self._settings.blank_penalty
        if penalty is not None and not isinstance(penalty, NotGiven):
            form.add_field("blank_penalty", str(penalty))

        if self._settings.word_timestamps is True:
            form.add_field("word_timestamps", "true")

        return form

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe a complete audio segment through the Akshara REST API.

        Args:
            audio: WAV audio bytes for one utterance, supplied by the base class.

        Yields:
            TranscriptionFrame on success, ErrorFrame on API or network failure.
        """
        try:
            await self.start_processing_metrics()

            async with self._session.post(
                f"{self._base_url}/v2/akshara/transcribe",
                data=self._build_form(audio),
                headers={"x-api-key": self._api_key},
            ) as response:
                if response.status != 200:
                    body = (await response.text())[:_MAX_ERROR_BODY]
                    msg = f"ConvoZen Akshara error ({response.status}): {body}"
                    await self.push_error(error_msg=msg)
                    yield ErrorFrame(error=msg)
                    return

                result = await response.json()

            await self.stop_processing_metrics()

            text = (result.get("text") or "").strip()
            if not text:
                return

            language = self._settings.language
            language = None if isinstance(language, NotGiven) else language

            await self._handle_transcription(text, True, language)
            logger.debug(f"Transcription: [{text}]")

            yield TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=result,
            )
        except Exception as e:
            # The exception is not interpolated into the message: an aiohttp error
            # carries RequestInfo, including the API key header, in its detail.
            msg = f"ConvoZen Akshara request failed: {type(e).__name__}"
            await self.push_error(error_msg=msg, exception=e)
            yield ErrorFrame(error=msg)
