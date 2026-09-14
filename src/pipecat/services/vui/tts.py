#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Vui TTS service implementation using the `vui-tts` engine."""

import asyncio
import threading
from collections.abc import AsyncGenerator, AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import require_given

try:
    import torch
    from vui.engine import Engine, GenConfig, Segment
    from vui.prompt_files import hub_prompt, hub_prompt_transcript, prompt_transcript
    from vui.qwen_codec import SAMPLE_RATE as VUI_SAMPLE_RATE
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Vui, you need to `uv add "pipecat-ai[vui]"`.')
    raise ImportError(f"Missing module: {e}") from e

#: Voices shipped with the model on the Hugging Face Hub (fluxions/vui).
VUI_BUILTIN_VOICES = ("maeve", "abraham", "rhian", "harry")


@dataclass
class VuiTTSSettings(TTSSettings):
    """Settings for VuiTTSService."""

    pass


class VuiTTSService(TTSService):
    """Vui TTS service implementation.

    Provides local text-to-speech with `Vui Nano <https://github.com/fluxions-ai/vui>`_,
    a small (219M active / 305M total parameters, Apache 2.0) context-aware
    model trained on real conversations. The engine runs in-process — on CUDA,
    or on MLX on Apple Silicon — and streams 24 kHz audio into the pipeline as
    each frame is decoded, so the first ``TTSAudioRawFrame`` arrives after the
    model's first frame rather than after the whole utterance. Model weights and
    voice prompts download from Hugging Face on first use, which may block for
    a while. On CUDA, install a flash-attn wheel matching your torch/CUDA for
    full speed; without it the engine uses PyTorch's SDPA fallback (same
    output, roughly a quarter of the throughput). Apple Silicon needs nothing
    extra.

    A *voice* is a reference clip the model clones: one of the shipped voices
    (``"maeve"``, ``"abraham"``, ``"rhian"``, ``"harry"``), a path to a prompt
    ``.safetensors`` baked with Vui's ``scripts/build_prompts.py``, or a path to
    a ``.wav`` (with a sibling ``.txt`` transcript, else transcribed on the
    fly). The voice can be changed at runtime; the checkpoint cannot.
    """

    Settings = VuiTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        checkpoint: str = "vui-nano-1.1",
        gen_config: Any | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Vui TTS service.

        Args:
            checkpoint: Vui checkpoint to load — a name from ``Engine.NAMES``
                (``"vui-nano-1.1"``, ``"vui-190k"``, ``"vui-nano"``), a file in
                the ``fluxions/vui`` Hub repo, or a local path.
            gen_config: Optional ``vui.engine.GenConfig`` overriding the
                sampling defaults (temperature 0.7, top_k, repetition penalty, ...).
            settings: Runtime-updatable settings. Defaults to the ``"maeve"``
                voice.
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        default_settings = self.Settings(model=checkpoint, voice="maeve", language=None)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        # Temperature 0.7 rather than the engine default 0.9: over repeated
        # renders 0.9 drops or swaps the odd word where 0.7 is clean.
        self._gen_config = gen_config if gen_config is not None else GenConfig(temperature=0.7)
        # Everything that touches the engine — load, prefill, decode — runs on
        # this one thread: the CUDA graphs and MLX's streams are bound to the
        # thread that created them.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vui-tts")
        self._lock = asyncio.Lock()

        logger.debug(f"Loading Vui checkpoint '{checkpoint}'")

        def _load():
            self._engine = Engine(checkpoint, max_rows=1)
            self._row = self._engine.new_row()

        self._executor.submit(_load).result()
        logger.debug(f"Loaded Vui checkpoint '{checkpoint}'")

        # The voice prompt is prefilled into the row once and the KV cache is
        # rewound to the end of it after every utterance; None forces a
        # re-prefill on the next utterance (voice changed).
        self._voice_loaded: str | None = None

    def can_generate_metrics(self) -> bool:
        """Indicate that this service supports TTFB and usage metrics."""
        return True

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """Apply a settings delta.

        Voice changes take effect on the next utterance. Model changes are
        stored but not applied, since they would require reloading weights.
        """
        changed = await super()._update_settings(delta)
        if not changed:
            return changed
        if "voice" in changed:
            self._voice_loaded = None
        unhandled = {k: v for k, v in changed.items() if k != "voice"}
        if unhandled:
            self._warn_unhandled_updated_settings(unhandled)
        return changed

    # ------------------------------------------------------------------
    # Voice prompts
    # ------------------------------------------------------------------

    def _resolve_voice(self, voice: str) -> Segment:
        """Turn a voice spec into a ``(transcript, codes)`` prompt segment."""
        from safetensors.torch import load_file

        path = Path(voice)
        if voice in VUI_BUILTIN_VOICES and not path.exists():
            st = hub_prompt(voice, self._engine.checkpoint)
            return Segment(hub_prompt_transcript(voice, st), load_file(st)["codes"].long())
        if path.suffix == ".safetensors":
            text = prompt_transcript(path)
            if not text:
                raise ValueError(f"{path}: no transcript in metadata or sibling .txt")
            return Segment(text, load_file(str(path))["codes"].long())
        if path.suffix.lower() == ".wav":
            return self._encode_wav(path)
        raise ValueError(
            f"Unknown Vui voice {voice!r}: expected one of {VUI_BUILTIN_VOICES}, "
            "a prompt .safetensors, or a .wav"
        )

    def _encode_wav(self, path: Path) -> Segment:
        """Encode a reference wav to codec codes; transcript from a sibling file or ASR."""
        from julius.resample import resample_frac
        from torchcodec.decoders import AudioDecoder
        from vui.qwen_codec import QwenCodecEncoder

        wav_16k = (
            AudioDecoder(str(path), sample_rate=16000, num_channels=1)
            .get_all_samples()
            .data.squeeze(0)
        )
        text = prompt_transcript(path)
        if not text:
            # Transcribing the reference needs openai-whisper, which is part of
            # the `vui-tts[server]` extra, not of `pipecat-ai[vui]`.
            try:
                from vui.inference import asr
            except ImportError as e:
                raise ValueError(
                    f"{path}: no transcript found. Put the exact transcript in a sibling "
                    f"{path.with_suffix('.txt').name}, bake a prompt .safetensors with Vui's "
                    "scripts/build_prompts.py, or install `vui-tts[server]` for automatic "
                    "transcription."
                ) from e
            text = asr(wav_16k)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        enc = QwenCodecEncoder.from_pretrained().to(dev).float().eval()
        with torch.inference_mode():
            wav_24k = resample_frac(wav_16k.unsqueeze(0), 16000, VUI_SAMPLE_RATE)
            codes = enc.encode(wav_24k.float().to(dev).unsqueeze(0))
        return Segment(text, codes[0, : self._engine.Q].T.long().cpu())

    def _ensure_voice(self) -> None:
        """Prefill the row with the current voice if it isn't already."""
        voice = require_given(self._settings.voice, "Vui voice")
        if self._voice_loaded == voice:
            return
        logger.debug(f"{self}: loading voice [{voice}]")
        segment = self._resolve_voice(voice)
        with torch.inference_mode():
            self._row.reset()
            self._row.prefill([segment])
        self._voice_loaded = voice

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def _synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Drive one Vui turn on the worker thread, yielding int16 PCM chunks.

        ``Row.stream`` is a blocking generator yielding ``(1, 1, N)`` float
        audio tensors at 24 kHz; it runs on the executor and frames cross back
        to the event loop through a queue. Closing this generator early (an
        interruption) sets ``cancel`` so the decode loop exits at the next
        frame; the KV cache is rewound to the voice prompt either way.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        cancel = threading.Event()
        sentinel = object()

        def _worker() -> None:
            try:
                self._ensure_voice()
                with torch.inference_mode():
                    for frame in self._row.stream(text, self._gen_config, cancel=cancel):
                        pcm = (
                            frame.reshape(-1)
                            .float()
                            .clamp(-1.0, 1.0)
                            .mul(32767.0)
                            .to(torch.int16)
                            .cpu()
                            .numpy()
                            .tobytes()
                        )
                        loop.call_soon_threadsafe(queue.put_nowait, pcm)
            except Exception as e:  # surfaced to the awaiting coroutine
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                try:
                    self._row.rewind()
                except Exception as e:  # a failed prefill leaves nothing to rewind to
                    logger.warning(f"{self}: rewind after turn failed: {e}")
                    self._voice_loaded = None
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        async with self._lock:
            fut = loop.run_in_executor(self._executor, _worker)
            try:
                while True:
                    item = await queue.get()
                    if item is sentinel:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                cancel.set()
                await fut

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Vui.

        Args:
            text: The text to convert to speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            await self.start_tts_usage_metrics(text)
            async for frame in self._stream_audio_frames_from_iterator(
                self._synthesize(text),
                in_sample_rate=VUI_SAMPLE_RATE,
                context_id=context_id,
            ):
                await self.stop_ttfb_metrics()
                yield frame
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
