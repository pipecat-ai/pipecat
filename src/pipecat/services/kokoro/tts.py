#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Kokoro TTS service implementation using kokoro-onnx."""

import asyncio
import os
import threading
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.deprecation import deprecated
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given, require_given

try:
    import requests
    from kokoro_onnx import Kokoro
    from kokoro_onnx.chunker import pause_after, split_phonemes
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Kokoro, you need to `uv add "pipecat-ai[kokoro]"`.')
    raise ImportError(f"Missing module: {e}") from e

KOKORO_CACHE_DIR = Path(os.path.expanduser("~/.cache/pipecat/kokoro-onnx"))
KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
)

# Pause lengths spliced between streamed chunks, matching the pauses
# kokoro-onnx itself inserts when joining its internal synthesis batches.
KOKORO_SENTENCE_PAUSE = 0.25
KOKORO_CLAUSE_PAUSE = 0.1


def _download_file(url: str, dest: Path):
    """Download a file from a URL to a destination path."""
    logger.debug(f"Downloading {url} to {dest}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.debug(f"Downloaded {dest}")


def _ensure_model_files(model_path: Path, voices_path: Path):
    """Download model files if they don't exist."""
    if not model_path.exists():
        _download_file(KOKORO_MODEL_URL, model_path)
    if not voices_path.exists():
        _download_file(KOKORO_VOICES_URL, voices_path)


def language_to_kokoro_language(language: Language) -> str:
    """Convert a Language enum to kokoro-onnx language code.

    kokoro-onnx phonemizes through espeak-ng, so these are espeak-ng voice names
    rather than ISO codes. They differ for Mandarin (``cmn``, not ``zh``) and
    French, which espeak-ng only offers per-region (``fr-fr``, no bare ``fr``);
    an unsupported name fails at synthesis time.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding kokoro-onnx locale code.

    """
    LANGUAGE_MAP = {
        Language.EN: "en-us",
        Language.EN_US: "en-us",
        Language.EN_GB: "en-gb",
        Language.ES: "es",
        Language.FR: "fr-fr",
        Language.FR_BE: "fr-be",
        Language.FR_CA: "fr-fr",
        Language.FR_CH: "fr-ch",
        Language.FR_FR: "fr-fr",
        Language.HI: "hi",
        Language.IT: "it",
        Language.JA: "ja",
        Language.PT: "pt",
        Language.PT_BR: "pt-br",
        Language.CMN: "cmn",
        Language.YUE: "yue",
        Language.ZH: "cmn",
        Language.ZH_CN: "cmn",
        Language.ZH_HK: "cmn",
        Language.ZH_TW: "cmn",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


@dataclass
class KokoroTTSSettings(TTSSettings):
    """Settings for KokoroTTSService.

    Parameters:
        speed: Speech rate multiplier (0.5-2.0). Defaults to 1.0.
        stream_chunk_phonemes: Maximum phonemes synthesized per streamed chunk.
            kokoro-onnx passes up to 510 phonemes through the model in one
            batch, so without chunking a long reply's first audio only arrives
            after one inference over the whole text — measured at 11s+ for a
            ~286-phoneme greeting on a CPU-only machine, well past the
            stop-frame timeout, which surfaces as the context "completing with
            no audio". Sentence-sized chunks keep the first audio one short
            inference away. Defaults to 60.
        warmup: Run a throwaway synthesis on a daemon thread at construction,
            priming espeak-ng and the ONNX session's first inference while the
            transport is still connecting. Defaults to True.
    """

    speed: float | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    stream_chunk_phonemes: int | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    warmup: bool | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class KokoroTTSService(TTSService):
    """Kokoro TTS service implementation.

    Provides local text-to-speech synthesis using kokoro-onnx.
    Automatically downloads model files on first use.

    Text is phonemized once and then synthesized as a chain of sentence-sized
    chunk streams, so the first audio arrives after one short inference and the
    rest is synthesized while the earlier audio plays.
    """

    Settings = KokoroTTSSettings
    _settings: Settings

    @deprecated(
        "`KokoroTTSService.InputParams` is deprecated since 0.0.105 and will be removed in 2.0.0. "
        "Use `KokoroTTSService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Input parameters for Kokoro TTS configuration.

        .. deprecated:: 0.0.105
            Use ``KokoroTTSService.Settings`` directly via the ``settings`` parameter instead.
            Will be removed in 2.0.0.

        Parameters:
            language: Language to use for synthesis.
        """

        language: Language = Language.EN

    def __init__(
        self,
        *,
        voice_id: str | None = None,
        model_path: str | None = None,
        voices_path: str | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        stop_frame_timeout_s: float = 10.0,
        **kwargs,
    ):
        """Initialize the Kokoro TTS service.

        Args:
            voice_id: Voice identifier to use for synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=KokoroTTSService.Settings(voice=...)`` instead.
                    Will be removed in 2.0.0.

            model_path: Path to the kokoro ONNX model file. Defaults to auto-downloaded file.
            voices_path: Path to the voices binary file. Defaults to auto-downloaded file.
            params: Configuration parameters for synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=KokoroTTSService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            stop_frame_timeout_s: Idle time before pushing TTSStoppedFrame. Local
                synthesis doesn't stall the way a network provider can — real
                failures raise and surface as their own error frames — so a slow
                first chunk gets a real chance instead of the 3s network default,
                which misreported a merely slow context as "completed with no
                audio". Defaults to 10.0.
            **kwargs: Additional arguments passed to parent `TTSService`.

        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model=None,
            voice=None,
            language=Language.EN,
            speed=1.0,
            stream_chunk_phonemes=60,
            warmup=True,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.language = params.language

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            stop_frame_timeout_s=stop_frame_timeout_s,
            **kwargs,
        )

        model_file = Path(model_path) if model_path else KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"
        voices = Path(voices_path) if voices_path else KOKORO_CACHE_DIR / "voices-v1.0.bin"

        _ensure_model_files(model_file, voices)

        self._kokoro = Kokoro(str(model_file), str(voices))

        voice = require_given(self._settings.voice, "Kokoro TTS voice")
        if voice not in self._kokoro.voices:
            raise ValueError(f"Kokoro TTS voice '{voice}' is not in the voices file")

        self._resampler = create_stream_resampler()

        # Prime espeak-ng and the ONNX session's first inference while the
        # transport handshake and the first LLM call are still in flight.
        # espeak phonemization is lock-guarded inside kokoro-onnx and ONNX
        # Runtime session.run is thread-safe, so this can race a real
        # synthesis; a failure only means the first reply synthesizes cold.
        if assert_given(self._settings.warmup):
            threading.Thread(target=self._warm_up, name="kokoro-warmup", daemon=True).start()

    def _warm_up(self) -> None:
        """Synthesize a throwaway phrase once, to prime the first inference."""
        try:
            started = time.perf_counter()
            voice = require_given(self._settings.voice, "Kokoro TTS voice")
            lang = require_given(self._settings.language, "Kokoro TTS language")
            speed = assert_given(self._settings.speed)
            self._kokoro.create("Ready.", voice=voice, speed=speed, lang=lang)
            logger.info(f"{self} warm-up synthesis done in {time.perf_counter() - started:.1f}s")
        except Exception as e:
            logger.warning(f"{self} warm-up failed (first reply will synthesize cold): {e}")

    def _prepare_stream_pieces(self, text: str, lang: str) -> list[str]:
        """Phonemize ``text`` once and split it into stream-sized pieces.

        Phonemization is synchronous CPU work (espeak-ng), so callers run it in
        a thread. The split uses kokoro-onnx's own punctuation-aware, balanced
        splitter with a much smaller cap than the model context, so the first
        audio arrives after one short inference instead of one pass over the
        whole text.
        """
        chunk_phonemes = assert_given(self._settings.stream_chunk_phonemes)
        phonemes = " ".join(self._kokoro.tokenizer.phonemize(text, lang).split())
        if not phonemes:
            raise ValueError(f"Nothing to synthesize, {text!r} produced no phonemes")
        pieces = [
            piece
            for piece in split_phonemes(phonemes, chunk_phonemes)
            if self._kokoro.tokenizer.tokenize(piece)
        ]
        if not pieces:
            raise ValueError(f"No phonemes of {text!r} are in the model vocabulary")
        return pieces

    def can_generate_metrics(self) -> bool:
        """Indicate that this service supports TTFB and usage metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str:
        """Convert a Language enum to kokoro-onnx language format.

        Args:
            language: The language to convert.

        Returns:
            The kokoro-onnx language code.
        """
        return language_to_kokoro_language(language)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Synthesize speech from text using kokoro-onnx.

        The text is phonemized once and then synthesized as a chain of
        sentence-sized chunk streams rather than one stream over the whole
        text. kokoro-onnx passes up to 510 phonemes through the model in one
        batch, so without chunking a long reply's first (and only) audio chunk
        takes one full-text inference — 11s+ for a ~286-phoneme greeting on a
        CPU-only machine, warm — which blows past the stop-frame timeout and
        surfaces as the context "completing with no audio" plus an error frame.
        Chunked, the first audio arrives after one short inference and the rest
        is synthesized while the earlier audio plays. Chunks are cut at
        punctuation via kokoro-onnx's own splitter and joined with its own
        pause lengths, exactly how it joins its internal batches.

        Args:
            text: The text to synthesize.
            context_id: Unique identifier for this TTS context.

        """
        try:
            await self.start_tts_usage_metrics(text)

            voice = require_given(self._settings.voice, "Kokoro TTS voice")
            lang = require_given(self._settings.language, "Kokoro TTS language")
            speed = assert_given(self._settings.speed)

            pieces = await asyncio.to_thread(self._prepare_stream_pieces, text, lang)
            if len(pieces) > 1:
                logger.debug(f"{self} streaming {text!r} as {len(pieces)} chunks")

            for i, piece in enumerate(pieces):
                stream = self._kokoro.create_stream(
                    piece, voice=voice, lang=lang, speed=speed, is_phonemes=True
                )
                async for samples, sample_rate in stream:
                    await self.stop_ttfb_metrics()

                    # Each chunk is the final batch of its own stream, so the
                    # pause its trailing punctuation calls for is not added by
                    # kokoro-onnx; splice it in for every chunk but the last.
                    if i < len(pieces) - 1:
                        pause = pause_after(piece, KOKORO_SENTENCE_PAUSE, KOKORO_CLAUSE_PAUSE)
                        if pause > 0:
                            samples = np.concatenate(
                                [
                                    samples,
                                    np.zeros(int(pause * sample_rate), dtype=samples.dtype),
                                ]
                            )

                    audio_int16 = (samples * 32767).astype(np.int16).tobytes()
                    audio_data = await self._resampler.resample(
                        audio_int16, sample_rate, self.sample_rate
                    )

                    yield TTSAudioRawFrame(
                        audio=audio_data,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            await self.stop_ttfb_metrics()
