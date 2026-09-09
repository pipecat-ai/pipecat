#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The user's voice: a caching TTS in the harness pipeline.

In audio mode the harness synthesizes each user turn and streams it to the
bot. :class:`CachingTTSService` wraps a real TTS service and caches its
audio on disk, keyed by service, voice, model, language, and text, so a
scripted utterance is synthesized once and reused across runs and bots.

Only local (Kokoro) and HTTP (Cartesia) services fit, since the wrapper
reads the inner service's ``run_tts`` generator directly.
"""

import hashlib
import os
import wave
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import cast

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.services.websocket_service import WebsocketService


def _default_cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME")
    base = Path(root) if root else Path.home() / ".cache"
    return base / "pipecat" / "evals" / "tts"


DEFAULT_CACHE_DIR = _default_cache_dir()

# Default sample rate for generated audio (matches pipecat input default).
DEFAULT_SAMPLE_RATE = 16000


def tts_sample_rate(voice_cfg: dict) -> int:
    """The sample rate a ``user_audio`` block asks for (default 16 kHz)."""
    return int(voice_cfg.get("sample_rate", DEFAULT_SAMPLE_RATE))


def tts_cache_key(voice_cfg: dict) -> str:
    """A stable identity for a ``user_audio`` config: service, voice, model, and language, not the sample rate."""
    service = str(voice_cfg.get("service", "")).lower()
    voice = str(voice_cfg.get("voice", ""))
    model = str(voice_cfg.get("model", ""))
    language = str(voice_cfg.get("language") or "")
    return "\x00".join((service, voice, model, language))


def _resolve_cache_dir(override: str | Path | None) -> Path:
    """Resolve the cache directory: explicit override > env var > default."""
    if override is not None:
        base = Path(override)
    elif os.environ.get("PIPECAT_EVALS_CACHE_DIR"):
        base = Path(os.environ["PIPECAT_EVALS_CACHE_DIR"])
    else:
        base = DEFAULT_CACHE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base


def _read_wav(path: Path) -> tuple[bytes, int]:
    """Read a mono 16-bit PCM WAV file. Returns ``(pcm_bytes, sample_rate)``."""
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono, got {wf.getnchannels()} channels")
        if wf.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit, got {wf.getsampwidth() * 8}-bit")
        return wf.readframes(wf.getnframes()), wf.getframerate()


def _write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    """Write mono 16-bit PCM to a WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


class CachingTTSService(TTSService):
    """A pipeline TTS that wraps a real service and caches its audio on disk.

    On a ``TTSSpeakFrame`` it emits the cached audio when present, otherwise
    it runs the wrapped service, caches the audio, and forwards it. The
    inner service's lifecycle is forwarded so it works inside the pipeline.
    Only local and HTTP inner services are supported.
    """

    def __init__(
        self,
        inner: TTSService,
        *,
        cache_key: str,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
        **kwargs,
    ):
        """Initialize the caching TTS.

        Args:
            inner: The local/HTTP ``TTSService`` that does the synthesis.
            cache_key: Stable identity for the inner config (see
                :func:`tts_cache_key`); combined with the text to key the cache.
            cache_dir: Where to store cached audio. Defaults to
                ``<user-cache-dir>/pipecat/evals/tts`` (or
                ``$PIPECAT_EVALS_CACHE_DIR``).
            use_cache: When False, ignore cached audio and don't write new files.
            **kwargs: Additional arguments passed to ``TTSService``.

        Raises:
            ValueError: If ``inner`` is a websocket-streaming TTS service (its
                ``run_tts`` doesn't yield audio); use a local or HTTP service.
        """
        if isinstance(inner, WebsocketService):
            raise ValueError(
                f"CachingTTSService supports only local or HTTP TTS services, not the "
                f"websocket-streaming {type(inner).__name__}. Use an HTTP variant "
                "(e.g. CartesiaHttpTTSService) or a local service (e.g. KokoroTTSService)."
            )
        super().__init__(**kwargs)
        self._inner = inner
        self._cache_key = cache_key
        self._cache_dir_override = cache_dir
        self._use_cache = use_cache

    async def setup(self, setup):
        """Set up this service and forward setup to the inner service."""
        await super().setup(setup)
        await self._inner.setup(setup)

    async def start(self, frame: StartFrame):
        """Start this service and the inner one, the inner with metrics off since it has no downstream."""
        await super().start(frame)
        inner_start = StartFrame(audio_out_sample_rate=self.sample_rate, enable_metrics=False)
        await self._inner.start(inner_start)

    async def stop(self, frame):
        """Stop the inner service, then this one."""
        await self._inner.stop(frame)
        await super().stop(frame)

    async def cancel(self, frame):
        """Cancel the inner service, then this one."""
        await self._inner.cancel(frame)
        await super().cancel(frame)

    async def cleanup(self):
        """Clean up the inner service, then this one."""
        await self._inner.cleanup()
        await super().cleanup()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Emit the audio for ``text`` from cache, or by driving the inner service.

        Args:
            text: The utterance to synthesize.
            context_id: TTS context id (forwarded to the inner service).

        Yields:
            ``TTSStartedFrame`` / ``TTSAudioRawFrame`` / ``TTSStoppedFrame`` (cache
            hit), or the inner service's own frames (cache miss).
        """
        cache_file = self._cache_file(text) if self._use_cache else None

        if cache_file is not None and cache_file.exists():
            try:
                pcm, cached_sr = _read_wav(cache_file)
                if cached_sr == self.sample_rate:
                    yield TTSStartedFrame()
                    yield TTSAudioRawFrame(audio=pcm, sample_rate=self.sample_rate, num_channels=1)
                    yield TTSStoppedFrame()
                    return
                logger.info(
                    f"Cache SR {cached_sr} != requested {self.sample_rate} "
                    f"for {text!r} — regenerating"
                )
            except Exception as e:
                logger.warning(f"Cache read failed for {cache_file}: {e} — regenerating")

        # Cache miss: drive the inner service, capturing its audio as we forward it.
        captured = bytearray()
        inner_frames = cast(
            AsyncGenerator[Frame | None, None], self._inner.run_tts(text, context_id)
        )
        async for frame in inner_frames:
            if isinstance(frame, TTSAudioRawFrame):
                captured.extend(frame.audio)
            elif isinstance(frame, ErrorFrame):
                logger.warning(f"TTS error during synthesis of {text!r}: {frame.error}")
            yield frame

        if cache_file is not None and captured:
            _write_wav(cache_file, bytes(captured), self.sample_rate)

    def _cache_file(self, text: str) -> Path:
        """Resolve the WAV cache path for ``text`` (keyed by config identity + text)."""
        h = hashlib.sha256()
        for part in (self._cache_key, text):
            # Lower-case so trivial casing differences hit the same slot.
            h.update(part.lower().encode("utf-8"))
            h.update(b"\x00")
        return _resolve_cache_dir(self._cache_dir_override) / f"{h.hexdigest()}.wav"
