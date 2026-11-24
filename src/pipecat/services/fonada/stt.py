"""Segmented FonadaLabs Kapture V3 speech-to-text service."""

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import aiohttp

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt


@dataclass
class FonadaSTTSettings(STTSettings):
    """Settings for Fonada ASR."""


class FonadaSTTService(SegmentedSTTService):
    """Transcribe each VAD-delimited utterance with Fonada Kapture V3."""

    Settings = FonadaSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        language: str | Language = Language.EN,
        base_url: str = "https://api.fonada.ai/v3/asr/transcribe",
        aiohttp_session: aiohttp.ClientSession | None = None,
        request_timeout: float = 90.0,
        **kwargs,
    ):
        """Initialize Fonada ASR settings and the HTTP session."""
        settings = self.Settings(model="kapture-v3", language=language)
        super().__init__(settings=settings, **kwargs)
        self._api_key = api_key or os.getenv("FONADALABS_API_KEY", "")
        if not self._api_key:
            raise ValueError("FONADALABS_API_KEY must be provided")
        self._base_url = base_url
        self._timeout = aiohttp.ClientTimeout(total=request_timeout)
        self._session = aiohttp_session
        self._owns_session = aiohttp_session is None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def cleanup(self):
        """Close the internally owned HTTP session."""
        await super().cleanup()
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    def language_to_service_language(self, language: Language) -> str:
        """Convert Pipecat's language enum to Fonada's base language code."""
        value = getattr(language, "value", str(language))
        return value.split("-", 1)[0].lower()

    @traced_stt
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Upload one VAD-delimited WAV segment and emit its transcript."""
        language = self.language_to_service_language(self._settings.language or Language.EN)
        form = aiohttp.FormData()
        form.add_field("file", audio, filename="utterance.wav", content_type="audio/wav")
        form.add_field("language_id", language)
        headers = {"Authorization": f"Bearer {self._api_key}"}
        session = await self._get_session()
        try:
            async with session.post(self._base_url, data=form, headers=headers) as response:
                body = await response.text()
                if response.status != 200:
                    yield ErrorFrame(error=f"Fonada ASR error ({response.status}): {body[:500]}")
                    return
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    yield ErrorFrame(error="Fonada ASR returned invalid JSON")
                    return
            text = str(payload.get("text", "")).strip() if isinstance(payload, dict) else ""
            if text:
                yield TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    self._settings.language,
                    result=payload,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield ErrorFrame(error=f"Fonada ASR request failed: {exc}")
