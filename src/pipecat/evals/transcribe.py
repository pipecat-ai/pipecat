#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bot-audio transcription for the eval harness.

When a scenario judges a spoken response, the harness needs the text of what the
bot *actually said* — the transcription of its synthesized audio, not the text
fed to the TTS. :class:`EvalTranscriber` provides that by calling an STT
service's ``run_stt()`` directly on the captured audio, mirroring how
:class:`~pipecat.evals.judge.EvalJudge` calls ``run_inference()`` — there is no
pipeline to run.

This works with any STT whose ``run_stt(audio)`` transcribes the buffer and
returns: local models like Whisper (which loads in its constructor) and HTTP
services. A live/streaming STT (e.g. Deepgram's WebSocket service) does *not*
fit — its ``run_stt`` ships audio to a socket and the results arrive out of band,
so nothing is yielded.

The transcriber takes an already-built ``STTService``;
:meth:`EvalTranscriber.from_config` constructs one from a scenario's
``judge.transcription:`` mapping — ``service`` (default ``"moonshine"``, a local
model), ``model``, and optional ``padding_secs``. The escape hatch is
``transcription.factory: "my_pkg.my_func"`` — an
importable callable taking ``(config, sample_rate)`` and returning an
``STTService``. Audio is resampled to 16 kHz before transcription, a rate STT
services expect.
"""

import importlib
from collections.abc import AsyncGenerator, Callable
from typing import cast

from loguru import logger

from pipecat.evals.services import moonshine_service, whisper_service
from pipecat.services.stt_service import STTService

# STT services expect 16 kHz mono audio.
STT_SAMPLE_RATE = 16000

# Silence padded onto each captured segment before transcription. The bot-speech
# buffer starts right at the bot's first audio frame (``bot-started-speaking``),
# an abrupt onset that makes Whisper drop a short leading word — e.g. a terse
# "Four." answer transcribes as just the trailing clause. A bit of leading and
# trailing silence gives the STT a clean lead-in/lead-out so it keeps the onset
# and offset words. Kept short so it can't trigger silence hallucinations.
SILENCE_PAD_S = 2


class EvalTranscriber:
    """Transcribes bot audio by calling an STT service's ``run_stt()`` directly.

    Takes an already-built ``STTService``; :meth:`from_config` builds one from a
    scenario's ``judge.transcription:`` mapping. Use as an async context manager::

        async with EvalTranscriber.from_config({"model": "base"}) as t:
            text = await t.transcribe(pcm, sample_rate=24000)

    Only STTs whose ``run_stt(audio)`` transcribes the buffer and returns are
    supported (local models like Whisper, HTTP services) — not live/streaming
    ones, whose results arrive out of band.
    """

    def __init__(self, service: STTService, *, padding_secs: float = SILENCE_PAD_S):
        """Initialize the transcriber.

        Args:
            service: A constructed ``STTService`` (e.g. from :meth:`from_config`).
            padding_secs: Silence padded onto each side of the segment before
                transcription, giving the STT a clean lead-in/lead-out so it
                keeps onset/offset words (see :data:`SILENCE_PAD_S`). ``0``
                disables padding.
        """
        self._service = service
        self._padding_secs = padding_secs
        self._resampler = None
        # Optional sink for timing diagnostics; the harness points this at its
        # per-scenario debug trace.
        self.debug: Callable[[str], None] = lambda _msg: None

    @classmethod
    def from_config(cls, config: dict | None) -> "EvalTranscriber":
        """Build an :class:`EvalTranscriber` from a scenario's ``judge.transcription:`` mapping.

        Honors a custom ``factory`` (dotted path to a callable taking
        ``(config, sample_rate)`` and returning an ``STTService``); otherwise
        dispatches on the ``service`` name (default ``"moonshine"``). Add providers
        by extending this. To use a fully custom setup, construct
        ``EvalTranscriber`` directly with your own ``STTService`` and pass it to
        :meth:`pipecat.evals.harness.EvalSession.from_scenario`.

        Args:
            config: ``transcription`` mapping, or ``None`` for the Moonshine default.
                An optional ``padding_secs`` overrides the silence padding
                (default :data:`SILENCE_PAD_S`; see :meth:`__init__`).

        Returns:
            A configured EvalTranscriber (not yet started).

        Example::

            # In the scenario: transcription.factory: "my_pkg.make_stt"
            def make_stt(config, sample_rate):
                return WhisperSTTService(...)
        """
        config = config or {}
        padding = config.get("padding_secs")
        padding_secs = SILENCE_PAD_S if padding is None else float(padding)

        custom = config.get("factory")
        if custom:
            module_name, _, attr = custom.rpartition(".")
            if not module_name:
                raise ValueError(f"transcription.factory must be a dotted path: {custom!r}")
            factory = getattr(importlib.import_module(module_name), attr)
            return cls(factory(config, STT_SAMPLE_RATE), padding_secs=padding_secs)

        name = str(config.get("service", "moonshine")).lower()
        if name == "whisper":
            return cls(whisper_service(config), padding_secs=padding_secs)
        if name == "moonshine":
            return cls(moonshine_service(config), padding_secs=padding_secs)

        raise ValueError(
            f"Unknown STT service: {name!r}. Known: whisper, moonshine. "
            "Or set transcription.factory to a 'module.func' returning an STTService."
        )

    async def start(self) -> None:
        """Prepare the transcriber. The STT service loads its model on construction."""
        from pipecat.audio.utils import create_file_resampler

        self._resampler = create_file_resampler()

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        """Transcribe one audio segment to text.

        Calls the STT service's ``run_stt()`` on the resampled, padded audio and
        joins the ``TranscriptionFrame``s it yields. Returns ``""`` when the audio
        contains no recognizable speech.

        Args:
            pcm: Raw 16-bit little-endian mono PCM.
            sample_rate: Sample rate of ``pcm``; resampled to 16 kHz if needed.

        Returns:
            The transcribed text, stripped, or ``""`` for silence.
        """
        from pipecat.frames.frames import ErrorFrame, TranscriptionFrame

        if self._resampler is None:
            raise RuntimeError("EvalTranscriber.start() was not called")
        if not pcm:
            return ""

        if sample_rate != STT_SAMPLE_RATE:
            pcm = await self._resampler.resample(pcm, sample_rate, STT_SAMPLE_RATE)

        # Pad with silence so the STT has a clean lead-in/lead-out and doesn't drop
        # a short onset/offset word (see SILENCE_PAD_S); padding_secs=0 disables it.
        if self._padding_secs:
            pad = b"\x00\x00" * int(STT_SAMPLE_RATE * self._padding_secs)
            pcm = pad + pcm + pad

        # run_stt transcribes the whole buffer and yields its TranscriptionFrame(s),
        # then the generator ends — so we just collect and join. No pipeline, no
        # timeout: the generator returning is the "done" signal.
        # run_stt's base signature types it as a coroutine, but every concrete STT
        # overrides it as an async generator; iterate it as such.
        parts: list[str] = []
        async for frame in cast(AsyncGenerator[object, None], self._service.run_stt(pcm)):
            if isinstance(frame, TranscriptionFrame):
                parts.append(frame.text.strip())
            elif isinstance(frame, ErrorFrame):
                self.debug(f"transcribe: ErrorFrame {frame.error}")
                logger.warning(f"EvalTranscriber: transcription error: {frame.error}")

        text = " ".join(p for p in parts if p).strip()
        self.debug(f"transcribe: {len(parts)} frame(s) -> {text!r}")
        return text

    async def aclose(self) -> None:
        """Release resources. The STT service holds no pipeline-managed state here."""
        self._resampler = None

    async def __aenter__(self) -> "EvalTranscriber":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()
