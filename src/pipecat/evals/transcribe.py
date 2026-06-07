#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bot-audio transcription for the eval harness.

When a scenario asserts on ``tts_response``, the harness needs the text of what
the bot *actually said* — the transcription of its synthesized audio, not the
text fed to the TTS. :class:`EvalTranscriber` provides that: it runs an STT
service in one persistent pipeline (mirroring
:class:`~pipecat.evals.voice.EvalVoice`) and transcribes buffered audio segments
on demand.

The transcriber takes an already-built ``STTService``;
:meth:`EvalTranscriber.from_config` constructs one from a scenario's
``bot_audio`` mapping — ``service`` (default ``"whisper"``, a local model) and
``model`` — much like :meth:`pipecat.evals.judge.EvalJudge.from_config`. The
escape hatch is ``bot_audio.factory: "my_pkg.my_func"`` — an importable callable
taking ``(config, sample_rate)`` and returning an ``STTService``. Audio is
resampled to 16 kHz before transcription, a rate STT services expect.
"""

import asyncio
import importlib
from collections.abc import Callable

from loguru import logger

from pipecat.evals.voice import _IdentitySerializer
from pipecat.services.stt_service import STTService

# STT services expect 16 kHz mono audio.
STT_SAMPLE_RATE = 16000

# Upper bound on a single transcription; also the silence fallback (no
# TranscriptionFrame is emitted for non-speech, so we time out and return "").
TRANSCRIBE_TIMEOUT_S = 30.0


def _whisper_service(config: dict):
    """Build a local Whisper STT service from the ``bot_audio`` config.

    The eval transcribes audio it already knows is the bot speaking (the harness
    captures it between ``bot-started-speaking`` and ``bot-stopped-speaking``), so
    Whisper's non-speech filter is counterproductive here: the default
    ``no_speech_prob=0.4`` drops correct transcriptions of synthetic/TTS speech,
    whose ``no_speech_prob`` jitters across ~0.4-0.6 run to run (a dropped segment
    yields no ``TranscriptionFrame``, so the harness then waits out the whole
    transcription timeout). Disable the filter with a permissive threshold.
    """
    from pipecat.services.whisper.stt import WhisperSTTService

    kwargs: dict = {"no_speech_prob": 1.0}
    model = config.get("model")
    if model:
        kwargs["model"] = str(model)
    return WhisperSTTService(settings=WhisperSTTService.Settings(**kwargs))


class EvalTranscriber:
    """Transcribes bot audio with an STT service from one persistent pipeline.

    Takes an already-built ``STTService``; :meth:`from_config` builds one from a
    scenario's ``bot_audio`` mapping. Use as an async context manager::

        async with EvalTranscriber.from_config({"model": "base"}) as t:
            text = await t.transcribe(pcm, sample_rate=24000)
    """

    def __init__(self, service: STTService):
        """Initialize the transcriber.

        Args:
            service: A constructed ``STTService`` (e.g. from :meth:`from_config`).
        """
        self._service = service
        self._worker = None
        self._runner_task = None
        self._output_generator = None
        self._resampler = None
        # Optional sink for timing diagnostics; the harness points this at its
        # per-scenario debug trace so a hung transcription is visible.
        self.debug: Callable[[str], None] = lambda _msg: None

    @classmethod
    def from_config(cls, config: dict | None) -> "EvalTranscriber":
        """Build an :class:`EvalTranscriber` from a scenario's ``bot_audio`` mapping.

        Honors a custom ``factory`` (dotted path to a callable taking
        ``(config, sample_rate)`` and returning an ``STTService``); otherwise
        dispatches on the ``service`` name (default ``"whisper"``). Add providers
        by extending this. To use a fully custom setup, construct
        ``EvalTranscriber`` directly with your own ``STTService`` and pass it to
        :func:`pipecat.evals.harness.run_scenario`.

        Args:
            config: ``bot_audio`` mapping, or ``None`` for the Whisper default.

        Returns:
            A configured EvalTranscriber (not yet started).

        Example::

            # In the scenario: bot_audio.factory: "my_pkg.make_stt"
            def make_stt(config, sample_rate):
                return DeepgramSTTService(...)
        """
        config = config or {}

        custom = config.get("factory")
        if custom:
            module_name, _, attr = custom.rpartition(".")
            if not module_name:
                raise ValueError(f"bot_audio.factory must be a dotted path: {custom!r}")
            factory = getattr(importlib.import_module(module_name), attr)
            return cls(factory(config, STT_SAMPLE_RATE))

        name = str(config.get("service", "whisper")).lower()
        if name == "whisper":
            return cls(_whisper_service(config))

        raise ValueError(
            f"Unknown STT service: {name!r}. Known: whisper. "
            "Or set bot_audio.factory to a 'module.func' returning an STTService."
        )

    async def start(self) -> None:
        """Build and start the persistent transcription pipeline."""
        from pipecat.audio.utils import create_file_resampler
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.worker import PipelineParams, PipelineWorker
        from pipecat.processors.async_generator import AsyncGeneratorProcessor
        from pipecat.workers.runner import WorkerRunner

        grab = AsyncGeneratorProcessor(serializer=_IdentitySerializer())
        self._worker = PipelineWorker(
            Pipeline([self._service, grab]),
            params=PipelineParams(audio_in_sample_rate=STT_SAMPLE_RATE),
            # Persistent: idles between segments and must not self-cancel.
            idle_timeout_secs=None,
            enable_rtvi=False,
        )
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(self._worker)
        self._runner_task = asyncio.create_task(runner.run())
        self._output_generator = grab.generator()
        self._resampler = create_file_resampler()

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        """Transcribe one audio segment to text.

        Call serially: each call drives the shared pipeline through one
        speak/stop segment and reads its transcription. Returns ``""`` when the
        audio contains no recognizable speech.

        Args:
            pcm: Raw 16-bit little-endian mono PCM.
            sample_rate: Sample rate of ``pcm``; resampled to 16 kHz if needed.

        Returns:
            The transcribed text, stripped, or ``""`` for silence.
        """
        from pipecat.frames.frames import (
            ErrorFrame,
            InputAudioRawFrame,
            TranscriptionFrame,
            VADUserStartedSpeakingFrame,
            VADUserStoppedSpeakingFrame,
        )

        if self._worker is None or self._output_generator is None or self._resampler is None:
            raise RuntimeError("EvalTranscriber.start() was not called")
        if not pcm:
            return ""

        if sample_rate != STT_SAMPLE_RATE:
            pcm = await self._resampler.resample(pcm, sample_rate, STT_SAMPLE_RATE)
        self.debug(f"transcribe: resampled {sample_rate}->{STT_SAMPLE_RATE}Hz, {len(pcm)}B")

        # VAD start/stop bracket the audio so the SegmentedSTTService buffers it
        # and transcribes the whole segment on stop.
        await self._worker.queue_frames(
            [
                VADUserStartedSpeakingFrame(),
                InputAudioRawFrame(audio=pcm, sample_rate=STT_SAMPLE_RATE, num_channels=1),
                VADUserStoppedSpeakingFrame(),
            ]
        )
        self.debug("transcribe: frames queued, awaiting transcription")

        try:
            async with asyncio.timeout(TRANSCRIBE_TIMEOUT_S):
                async for frame in self._output_generator:
                    if isinstance(frame, TranscriptionFrame):
                        return frame.text.strip()
                    if isinstance(frame, ErrorFrame):
                        self.debug(f"transcribe: ErrorFrame {frame.error}")
                        logger.warning(f"EvalTranscriber: transcription error: {frame.error}")
                        return ""
        except TimeoutError:
            self.debug(f"transcribe: TIMEOUT after {TRANSCRIBE_TIMEOUT_S}s — no TranscriptionFrame")
            self._dump_threads()
            return ""  # no speech detected in the segment
        return ""

    def _dump_threads(self) -> None:
        """Dump thread stacks AND asyncio task stacks to the debug trace.

        Used on a transcription timeout to find where the STT inference is parked.
        Threads catch a hang in C (e.g. ``ctranslate2``); asyncio tasks catch a
        suspended coroutine (e.g. the STT pipeline worker awaiting something that
        never arrives), which a thread dump cannot show.
        """
        import io
        import sys
        import threading
        import traceback

        names = {t.ident: t.name for t in threading.enumerate()}
        for tid, frame in sys._current_frames().items():
            stack = "".join(traceback.format_stack(frame)).rstrip()
            self.debug(f"--- thread {names.get(tid, '?')} ({tid}) ---\n{stack}")

        try:
            tasks = asyncio.all_tasks()
        except RuntimeError:
            tasks = set()
        for task in tasks:
            buf = io.StringIO()
            task.print_stack(file=buf)
            self.debug(f"--- task {task.get_name()} ---\n{buf.getvalue().rstrip()}")

    async def aclose(self) -> None:
        """Stop the pipeline and release the model."""
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

    async def __aenter__(self) -> "EvalTranscriber":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()
