#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bot-audio transcription for the eval harness.

When a scenario asserts on ``tts_response``, the harness needs the text of what
the bot *actually said* — the transcription of its synthesized audio, not the
text fed to the TTS. :class:`EvalTranscriber` provides that: it runs a local
Whisper model in one persistent pipeline (mirroring
:class:`~pipecat.evals.voice.EvalVoice`) and transcribes buffered audio segments
on demand.

The model is configured by a scenario's ``bot_audio.model`` (defaults to the
``WhisperSTTService`` default). Audio is resampled to 16 kHz before
transcription, the rate Whisper expects.
"""

import asyncio

from loguru import logger

from pipecat.evals.voice import _IdentitySerializer

# Whisper expects 16 kHz mono audio.
WHISPER_SAMPLE_RATE = 16000

# Upper bound on a single transcription; also the silence fallback (no
# TranscriptionFrame is emitted for non-speech, so we time out and return "").
TRANSCRIBE_TIMEOUT_S = 30.0


class EvalTranscriber:
    """Transcribes bot audio with a local Whisper model from one persistent pipeline.

    Use as an async context manager::

        async with EvalTranscriber({"model": "base"}) as t:
            text = await t.transcribe(pcm, sample_rate=24000)
    """

    def __init__(self, config: dict | None = None):
        """Initialize the transcriber.

        Args:
            config: ``bot_audio`` mapping; only ``model`` is read (the local
                Whisper model name, e.g. ``"base"`` or a HF path). Omit to use
                the ``WhisperSTTService`` default.
        """
        config = config or {}
        self._model = str(config["model"]) if config.get("model") else None
        self._worker = None
        self._runner_task = None
        self._output_generator = None
        self._resampler = None

    async def start(self) -> None:
        """Build and start the persistent Whisper pipeline."""
        from pipecat.audio.utils import create_file_resampler
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.worker import PipelineParams, PipelineWorker
        from pipecat.processors.async_generator import AsyncGeneratorProcessor
        from pipecat.services.whisper.stt import WhisperSTTService
        from pipecat.workers.runner import WorkerRunner

        if self._model:
            service = WhisperSTTService(settings=WhisperSTTService.Settings(model=self._model))
        else:
            service = WhisperSTTService()

        grab = AsyncGeneratorProcessor(serializer=_IdentitySerializer())
        self._worker = PipelineWorker(
            Pipeline([service, grab]),
            params=PipelineParams(audio_in_sample_rate=WHISPER_SAMPLE_RATE),
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

        if sample_rate != WHISPER_SAMPLE_RATE:
            pcm = await self._resampler.resample(pcm, sample_rate, WHISPER_SAMPLE_RATE)

        # VAD start/stop bracket the audio so the SegmentedSTTService buffers it
        # and transcribes the whole segment on stop.
        await self._worker.queue_frames(
            [
                VADUserStartedSpeakingFrame(),
                InputAudioRawFrame(audio=pcm, sample_rate=WHISPER_SAMPLE_RATE, num_channels=1),
                VADUserStoppedSpeakingFrame(),
            ]
        )

        try:
            async with asyncio.timeout(TRANSCRIBE_TIMEOUT_S):
                async for frame in self._output_generator:
                    if isinstance(frame, TranscriptionFrame):
                        return frame.text.strip()
                    if isinstance(frame, ErrorFrame):
                        logger.warning(f"EvalTranscriber: transcription error: {frame.error}")
                        return ""
        except TimeoutError:
            return ""  # no speech detected in the segment
        return ""

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
