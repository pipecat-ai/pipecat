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

:class:`EvalVoice` owns this: it builds one persistent pipeline running the
configured TTS service and reuses it across the scenario's audio turns, so any
TTS service works (HTTP or streaming/WebSocket) — the audio comes back as
``TTSAudioRawFrame``s flowing through the pipeline. Generated audio is cached at
``.cache/evals/tts/<sha256>.wav`` (keyed by text/service/voice/model); the cache
file's actual sample rate is checked on load and regenerated on mismatch, so you
can experiment with sample rates without bloating the cache.

Built-in service support lives in ``EvalVoice._build_service``. Escape hatch:
``user_audio.factory: "my_pkg.my_func"`` — an importable callable taking
``(voice_cfg, sample_rate)`` and returning a ``TTSService``.
"""

import asyncio
import hashlib
import importlib
import os
import wave
from pathlib import Path

from loguru import logger

from pipecat.frames.frames import Frame
from pipecat.serializers.base_serializer import FrameSerializer

# Where cached audio lives. Override via the PIPECAT_EVALS_CACHE_DIR env var.
DEFAULT_CACHE_DIR = Path(".cache/evals/tts")

# Default sample rate for generated audio (matches pipecat input default).
DEFAULT_SAMPLE_RATE = 16000


class _IdentitySerializer(FrameSerializer):
    """Passes frames through unchanged so ``AsyncGeneratorProcessor`` yields the frames.

    ``AsyncGeneratorProcessor`` serializes each frame and exposes the results via
    an async generator; an identity serializer lets :class:`EvalVoice` receive
    the actual frames and pick out the audio.
    """

    async def serialize(self, frame: Frame):
        """Return the frame unchanged."""
        return frame

    async def deserialize(self, data):
        """Unused — we only consume the generator."""
        return None


class EvalVoice:
    """Generates user audio for a scenario from one persistent TTS pipeline.

    A scenario's ``user_audio:`` block configures a single TTS service. This
    builds one pipeline running that service and reuses it across all the
    scenario's audio turns — connecting the service once instead of per turn.
    Synthesized audio is cached on disk, so re-runs skip synthesis entirely.

    Use as an async context manager::

        async with EvalVoice(voice_cfg) as voice:
            pcm, sample_rate = await voice.generate("hello world")
    """

    def __init__(self, voice_cfg: dict):
        """Initialize the generator.

        Args:
            voice_cfg: ``user_audio`` mapping — ``service`` and ``voice`` at
                minimum; optional ``model``, ``sample_rate`` (default 16000),
                ``api_key`` (defaults to $CARTESIA_API_KEY), and ``factory`` (custom
                service factory).
        """
        service = str(voice_cfg.get("service", "")).lower()
        voice = str(voice_cfg.get("voice", ""))
        if not service or not voice:
            raise ValueError("user_audio config requires at least 'service' and 'voice'")

        self._voice_cfg = voice_cfg
        self._service_name = service
        self._voice = voice
        self._model = str(voice_cfg.get("model", ""))
        self._sample_rate = int(voice_cfg.get("sample_rate", DEFAULT_SAMPLE_RATE))

        self._worker = None
        self._runner_task = None
        self._output_generator = None

    async def start(self) -> None:
        """Build and start the persistent TTS pipeline."""
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.worker import PipelineParams, PipelineWorker
        from pipecat.processors.async_generator import AsyncGeneratorProcessor
        from pipecat.workers.runner import WorkerRunner

        service = self._build_service()
        grab = AsyncGeneratorProcessor(serializer=_IdentitySerializer())
        self._worker = PipelineWorker(
            Pipeline([service, grab]),
            params=PipelineParams(audio_out_sample_rate=self._sample_rate),
            # Persistent: the pipeline idles between turns and must not self-cancel.
            idle_timeout_secs=None,
            # This is an internal synthesis pipeline; no RTVI machinery.
            enable_rtvi=False,
        )
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(self._worker)
        self._runner_task = asyncio.create_task(runner.run())
        self._output_generator = grab.generator()

    async def generate(self, text: str) -> tuple[bytes, int]:
        """Return ``(pcm, sample_rate)`` for ``text`` — from cache or freshly synthesized.

        Call serially: each call queues one utterance on the shared pipeline and
        reads its audio off the single output generator until ``TTSStoppedFrame``.

        Args:
            text: The utterance to synthesize.

        Returns:
            Tuple of ``(pcm_bytes, sample_rate)`` — raw 16-bit little-endian mono PCM.
        """
        cache_file = self._cache_file(text)

        if cache_file.exists():
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
        self._write_wav(cache_file, pcm, self._sample_rate)
        return pcm, self._sample_rate

    async def _synthesize(self, text: str) -> bytes:
        """Queue one utterance and collect its audio from the shared pipeline."""
        from pipecat.frames.frames import (
            ErrorFrame,
            TTSAudioRawFrame,
            TTSSpeakFrame,
            TTSStoppedFrame,
        )

        if self._worker is None or self._output_generator is None:
            raise RuntimeError("EvalVoice.start() was not called")

        await self._worker.queue_frames([TTSSpeakFrame(text)])

        pcm = bytearray()
        async for frame in self._output_generator:
            if isinstance(frame, TTSStoppedFrame):
                break
            if isinstance(frame, TTSAudioRawFrame):
                pcm.extend(frame.audio)
            elif isinstance(frame, ErrorFrame):
                raise RuntimeError(f"TTS error during synthesis: {frame.error}")
        if not pcm:
            raise RuntimeError(f"TTS produced no audio for {text!r}")
        return bytes(pcm)

    async def aclose(self) -> None:
        """Stop the pipeline and release the TTS service."""
        if self._worker is not None:
            await self._worker.cancel()
        if self._runner_task is not None:
            try:
                await self._runner_task
            except (asyncio.CancelledError, Exception):
                pass
        self._worker = None
        self._runner_task = None
        self._output_generator = None

    async def __aenter__(self) -> "EvalVoice":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    #
    # TTS service construction
    #

    def _build_service(self):
        """Build the configured TTS service.

        Honors a custom ``user_audio.factory`` (dotted path to a callable taking
        ``(voice_cfg, sample_rate)`` and returning a ``TTSService``); otherwise
        dispatches on the ``service`` name. Add providers by extending this.
        """
        custom = self._voice_cfg.get("factory")
        if custom:
            module_name, _, attr = custom.rpartition(".")
            if not module_name:
                raise ValueError(f"user_audio.factory must be a dotted path: {custom!r}")
            factory = getattr(importlib.import_module(module_name), attr)
            return factory(self._voice_cfg, self._sample_rate)

        if self._service_name == "cartesia":
            return self._cartesia()

        raise ValueError(
            f"Unknown TTS service: {self._service_name!r}. Known: cartesia. "
            "Or set user_audio.factory to a 'module.func' returning a TTSService."
        )

    def _cartesia(self):
        """Build a Cartesia TTS service from the ``user_audio`` config."""
        from pipecat.services.cartesia.tts import CartesiaHttpTTSService

        # Prefer an explicit api_key in the config; fall back to the env var so
        # committed scenarios don't carry secrets.
        api_key = self._voice_cfg.get("api_key") or os.environ.get("CARTESIA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Cartesia API key not found — set $CARTESIA_API_KEY or user_audio.api_key"
            )

        return CartesiaHttpTTSService(
            api_key=api_key,
            settings=CartesiaHttpTTSService.Settings(
                voice=self._voice,
                model=self._voice_cfg.get("model") or "sonic-2",
            ),
            sample_rate=self._sample_rate,
        )

    #
    # Audio cache
    #

    def _cache_file(self, text: str) -> Path:
        """Resolve the cache path for ``text``.

        Keyed by the audio's semantic identity (service, voice, model, text) —
        sample rate is excluded so different rates reuse the same slot and a
        mismatch just triggers regeneration in :meth:`generate`.
        """
        h = hashlib.sha256()
        for part in (self._service_name, self._voice, self._model, text):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return self._cache_dir() / f"{h.hexdigest()}.wav"

    @staticmethod
    def _cache_dir() -> Path:
        """Resolve the cache directory, honoring the env var override."""
        override = os.environ.get("PIPECAT_EVALS_CACHE_DIR")
        base = Path(override) if override else DEFAULT_CACHE_DIR
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
