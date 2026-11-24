"""Streaming FonadaLabs Klone v2 Flash text-to-speech service."""

import asyncio
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import aiohttp

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import InterruptibleTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts


@dataclass
class FonadaTTSSettings(TTSSettings):
    """Settings for Fonada Klone v2 Flash."""

    share_id: str | None = None
    speed: float | None = None
    instruct: str | None = None


class FonadaTTSService(InterruptibleTTSService):
    """Stream Fonada Klone v2 Flash PCM audio into a Pipecat pipeline."""

    Settings = FonadaTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        share_id: str | None = None,
        language: str | Language = Language.EN,
        aiohttp_session: aiohttp.ClientSession | None = None,
        base_url: str = "https://api.fonada.ai/v1/voice-clone/stream",
        request_timeout: float = 120.0,
        **kwargs,
    ):
        """Initialize Fonada streaming TTS settings and the HTTP session."""
        settings = self.Settings(model="klone_v2_flash", language=language, share_id=share_id)
        super().__init__(push_start_frame=True, push_stop_frames=True, settings=settings, **kwargs)
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

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Stream Fonada PCM audio for one assistant text turn."""
        share_id = (self._settings.share_id or os.getenv("FONADA_VOICE_SHARE_ID", "")).strip()
        if not share_id:
            yield ErrorFrame(error="Fonada TTS requires FONADA_VOICE_SHARE_ID or settings.share_id")
            yield TTSStoppedFrame(context_id=context_id)
            return
        language = getattr(self._settings.language, "value", str(self._settings.language or "en"))
        data = {
            "text": text,
            "language": language.split("-", 1)[0].lower(),
            "output_audio_codec": "pcm",
            "model": "klone_v2_flash",
            "share_id": share_id,
        }
        if self._settings.speed is not None:
            data["speed"] = str(self._settings.speed)
        if self._settings.instruct:
            data["instruct"] = self._settings.instruct
        session = await self._get_session()
        try:
            async with session.post(
                self._base_url,
                data=data,
                headers={"Authorization": f"Bearer {self._api_key}"},
            ) as response:
                if response.status != 200:
                    body = (await response.text())[:500]
                    yield ErrorFrame(error=f"Fonada TTS error ({response.status}): {body}")
                    yield TTSStoppedFrame(context_id=context_id)
                    return
                await self.start_tts_usage_metrics(text)
                async for chunk in response.content.iter_chunked(self.chunk_size):
                    if chunk:
                        yield TTSAudioRawFrame(chunk, 24000, 1, context_id=context_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield ErrorFrame(error=f"Fonada TTS request failed: {exc}")
        finally:
            await self.stop_ttfb_metrics()
