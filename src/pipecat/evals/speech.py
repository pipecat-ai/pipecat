#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User-audio generation for the eval harness.

When a scenario defines a ``user_audio:`` block, the harness generates raw audio
for each user turn using that TTS config and streams it to the bot as RTVI
``raw-audio`` messages. The bot's STT then processes it for real, exercising the
full input audio path.

:class:`EvalSpeech` sets up an already-built ``TTSService`` with a minimal
frame-processor lifecycle (no pipeline or worker) and calls its ``run_tts``
directly for each turn, collecting the ``TTSAudioRawFrame``s. Only local (e.g.
Kokoro) or HTTP (e.g. ``CartesiaHttpTTSService``) services are supported —
websocket-streaming TTS needs a pipeline to manage its connection. Generated
audio is cached at ``<cache-dir>/<sha256>.wav`` (keyed by ``cache_key`` +
lower-cased text); the cache file's actual sample rate is checked on load and
regenerated on mismatch, so you can experiment with sample rates without bloating
the cache.

:meth:`EvalSpeech.from_config` constructs the service from a ``user_audio``
mapping (dispatch + a ``user_audio.factory`` escape hatch) and wraps it;
:func:`tts_cache_key` and :func:`tts_sample_rate` derive the cache identity and
rate.
"""

import asyncio
import hashlib
import importlib
import os
import wave
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import cast

from loguru import logger

from pipecat.evals.services import cartesia_service, kokoro_service
from pipecat.services.tts_service import TTSService
from pipecat.services.websocket_service import WebsocketService


# Default location for cached audio: <user-cache-dir>/pipecat/tts. Callers can
# override per-run with an explicit cache_dir or the PIPECAT_EVALS_CACHE_DIR env
# var (resolved in _cache_dir).
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
    """A stable identity for a ``user_audio`` config, for caching synthesized audio.

    Covers the audio's semantic identity (service, voice, model) but not the
    sample rate, so different rates reuse the same slot (a mismatch just triggers
    regeneration in :meth:`EvalSpeech.generate`).
    """
    service = str(voice_cfg.get("service", "")).lower()
    voice = str(voice_cfg.get("voice", ""))
    model = str(voice_cfg.get("model", ""))
    return "\x00".join((service, voice, model))


class EvalSpeech:
    """Generates user audio for a scenario by calling a TTS service's ``run_tts``.

    Takes an already-built ``TTSService``; :meth:`from_config` builds one from a
    scenario's ``user_audio`` mapping. :meth:`start` sets the service up with a
    minimal frame-processor lifecycle (no pipeline or worker — just a task manager,
    a clock, and a ``StartFrame``), then :meth:`generate` calls ``run_tts`` directly
    and collects the audio. Synthesized audio is cached on disk (keyed by
    ``cache_key`` + text), so re-runs skip synthesis entirely.

    Only **local** (e.g. Kokoro) or **HTTP** (e.g. ``CartesiaHttpTTSService``)
    services are supported. Websocket-streaming TTS needs a running pipeline to
    manage its connection lifecycle, which ``run_tts`` alone doesn't drive, so it
    is rejected (see :meth:`__init__`).

    Use as an async context manager::

        async with EvalSpeech(service, sample_rate=16000, cache_key="...") as speech:
            pcm, sample_rate = await speech.generate("hello world")
    """

    def __init__(
        self,
        service: TTSService,
        *,
        sample_rate: int,
        cache_key: str,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
    ):
        """Initialize the generator.

        Args:
            service: A constructed local or HTTP ``TTSService`` (e.g. from
                :meth:`from_config`).
            sample_rate: Output sample rate the service produces.
            cache_key: Stable identity for the service config (see
                :func:`tts_cache_key`); combined with the text to key the cache.
            cache_dir: Where to store cached audio. Defaults to
                ``<user-cache-dir>/pipecat/tts`` (or ``$PIPECAT_EVALS_CACHE_DIR``).
            use_cache: When False, ignore any cached audio and don't write new
                cache files — every utterance is freshly synthesized.

        Raises:
            ValueError: If ``service`` is a websocket-streaming TTS service
                (``run_tts`` can't be driven without a pipeline to manage its
                connection); use a local or HTTP service instead.
        """
        if isinstance(service, WebsocketService):
            raise ValueError(
                f"EvalSpeech supports only local or HTTP TTS services, not the "
                f"websocket-streaming {type(service).__name__}. Use an HTTP variant "
                "(e.g. CartesiaHttpTTSService) or a local service (e.g. KokoroTTSService)."
            )
        self._service = service
        self._sample_rate = sample_rate
        self._cache_key = cache_key
        self._cache_dir_override = Path(cache_dir) if cache_dir else None
        self._use_cache = use_cache
        self._started = False

    @property
    def sample_rate(self) -> int:
        """Sample rate (Hz) of the audio this generates."""
        return self._sample_rate

    @classmethod
    def from_config(
        cls,
        voice_cfg: dict,
        *,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
    ) -> "EvalSpeech":
        """Build an :class:`EvalSpeech` from a scenario's ``user_audio`` mapping.

        Honors a custom ``factory`` (dotted path to a callable taking
        ``(voice_cfg, sample_rate)`` and returning a ``TTSService``); otherwise
        dispatches on the ``service`` name. Add providers by extending this. To
        use a fully custom setup, construct ``EvalSpeech`` directly with your own
        ``TTSService`` and pass it to :meth:`pipecat.evals.harness.EvalSession.from_scenario`.

        Args:
            voice_cfg: ``user_audio`` mapping — ``service`` (``kokoro`` or
                ``cartesia``) and ``voice`` at minimum; optional ``sample_rate``
                (default 16 kHz), plus ``model`` / ``api_key`` (defaults to
                $CARTESIA_API_KEY) for Cartesia.
            cache_dir: Where to store cached audio (see :meth:`__init__`).
            use_cache: When False, force fresh synthesis (see :meth:`__init__`).

        Returns:
            A configured EvalSpeech (not yet started).

        Example::

            # In the scenario: user_audio.factory: "my_pkg.make_tts"
            def make_tts(voice_cfg, sample_rate):
                return RimeTTSService(...)
        """
        sample_rate = tts_sample_rate(voice_cfg)

        custom = voice_cfg.get("factory")
        if custom:
            module_name, _, attr = custom.rpartition(".")
            if not module_name:
                raise ValueError(f"user_audio.factory must be a dotted path: {custom!r}")
            factory = getattr(importlib.import_module(module_name), attr)
            service = factory(voice_cfg, sample_rate)
        else:
            name = str(voice_cfg.get("service", "")).lower()
            voice = str(voice_cfg.get("voice", ""))
            if not name or not voice:
                raise ValueError("user_audio config requires at least 'service' and 'voice'")
            if name == "kokoro":
                service = kokoro_service(voice_cfg, sample_rate)
            elif name == "cartesia":
                service = cartesia_service(voice_cfg, sample_rate)
            else:
                raise ValueError(
                    f"Unknown TTS service: {name!r}. Known: kokoro, cartesia. "
                    "Or set user_audio.factory to a 'module.func' returning a TTSService."
                )

        return cls(
            service,
            sample_rate=sample_rate,
            cache_key=tts_cache_key(voice_cfg),
            cache_dir=cache_dir,
            use_cache=use_cache,
        )

    async def start(self) -> None:
        """Set the service up so :meth:`generate` can call ``run_tts`` directly.

        A TTS service normally gets its task manager, clock, and output sample rate
        from the pipeline that hosts it. Provide a minimal stand-in here (no
        pipeline, no worker): a ``StartFrame`` sets the sample rate and opens any
        HTTP session the service needs. ``enable_metrics`` is off so ``run_tts``
        never tries to push a ``MetricsFrame`` (there is no downstream).
        """
        from pipecat.clocks.system_clock import SystemClock
        from pipecat.frames.frames import StartFrame
        from pipecat.processors.frame_processor import FrameProcessorSetup
        from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

        task_manager = TaskManager()
        task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        clock = SystemClock()
        clock.start()
        # There deliberately is no PipelineWorker: the service runs out-of-pipeline,
        # and the FrameProcessor.pipeline_worker property keeps raising if touched.
        await self._service.setup(
            FrameProcessorSetup(
                clock=clock,
                task_manager=task_manager,
                pipeline_worker=None,  # pyright: ignore[reportArgumentType]
            )
        )
        await self._service.start(
            StartFrame(audio_out_sample_rate=self._sample_rate, enable_metrics=False)
        )
        self._started = True

    async def generate(self, text: str) -> tuple[bytes, int]:
        """Return ``(pcm, sample_rate)`` for ``text`` — from cache or freshly synthesized.

        Call serially: each call runs one ``run_tts`` on the shared service and
        collects its audio.

        Args:
            text: The utterance to synthesize.

        Returns:
            Tuple of ``(pcm_bytes, sample_rate)`` — raw 16-bit little-endian mono PCM.
        """
        cache_file = self._cache_file(text) if self._use_cache else None

        if cache_file is not None and cache_file.exists():
            try:
                pcm, cached_sr = self._read_wav(cache_file)
                if cached_sr == self._sample_rate:
                    return pcm, self._sample_rate
                logger.info(
                    f"Cache SR {cached_sr} ≠ requested {self._sample_rate} "
                    f"for {text!r} — regenerating"
                )
            except Exception as e:
                logger.warning(f"Cache read failed for {cache_file}: {e} — regenerating")

        pcm = await self._synthesize(text)
        if cache_file is not None:
            self._write_wav(cache_file, pcm, self._sample_rate)
        return pcm, self._sample_rate

    async def _synthesize(self, text: str) -> bytes:
        """Run one ``run_tts`` and collect its audio frames into PCM bytes."""
        from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame

        if not self._started:
            raise RuntimeError("EvalSpeech.start() was not called")

        pcm = bytearray()
        # run_tts's base signature types it as a coroutine, but every concrete TTS
        # overrides it as an async generator; iterate it as such (mirrors
        # EvalTranscriber's run_stt handling).
        frames = cast(AsyncGenerator[object, None], self._service.run_tts(text, context_id="eval"))
        async for frame in frames:
            if isinstance(frame, TTSAudioRawFrame):
                pcm.extend(frame.audio)
            elif isinstance(frame, ErrorFrame):
                raise RuntimeError(f"TTS error during synthesis: {frame.error}")
        if not pcm:
            raise RuntimeError(f"TTS produced no audio for {text!r}")
        return bytes(pcm)

    async def aclose(self) -> None:
        """Stop and clean up the TTS service (closes any HTTP session)."""
        from pipecat.frames.frames import EndFrame

        if self._started:
            await self._service.stop(EndFrame())
            await self._service.cleanup()
            self._started = False

    async def __aenter__(self) -> "EvalSpeech":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    #
    # Audio cache
    #

    def _cache_file(self, text: str) -> Path:
        """Resolve the cache path for ``text``.

        Keyed by ``cache_key`` (the service config's semantic identity) plus the
        text. Sample rate is excluded, so different rates reuse the same slot and
        a mismatch just triggers regeneration in :meth:`generate`.
        """
        h = hashlib.sha256()
        for part in (self._cache_key, text):
            # Lower-case so trivial casing differences ("Hello" vs "hello") hit
            # the same cache slot.
            h.update(part.lower().encode("utf-8"))
            h.update(b"\x00")
        return self._cache_dir() / f"{h.hexdigest()}.wav"

    def _cache_dir(self) -> Path:
        """Resolve the cache directory: explicit cache_dir > env var > default."""
        if self._cache_dir_override is not None:
            base = self._cache_dir_override
        elif os.environ.get("PIPECAT_EVALS_CACHE_DIR"):
            base = Path(os.environ["PIPECAT_EVALS_CACHE_DIR"])
        else:
            base = DEFAULT_CACHE_DIR
        base.mkdir(parents=True, exist_ok=True)
        return base

    @staticmethod
    def _read_wav(path: Path) -> tuple[bytes, int]:
        """Read a mono 16-bit PCM WAV file. Returns (pcm_bytes, sample_rate)."""
        with wave.open(str(path), "rb") as wf:
            if wf.getnchannels() != 1:
                raise ValueError(f"{path}: expected mono, got {wf.getnchannels()} channels")
            if wf.getsampwidth() != 2:
                raise ValueError(f"{path}: expected 16-bit, got {wf.getsampwidth() * 8}-bit")
            return wf.readframes(wf.getnframes()), wf.getframerate()

    @staticmethod
    def _write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
        """Write mono 16-bit PCM to a WAV file."""
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
