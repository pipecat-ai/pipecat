#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audio quality analysis for Pipecat backed by the ai-coustics Tyto model.

Classes:
    AICAudioQualityScores: Audio quality scores for one window of user audio.
    AICAudioQualitySummary: Summary of the audio quality windows of one call.
    AICTytoObserver: Scores the user's audio during a call and summarizes the
        scores when the call ends.
"""

import asyncio
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from aic_sdk import (
    AnalysisResult,
    Analyzer,
    Collector,
    Model,
    ProcessorConfig,
    analyzer_pair,
    # Exported at runtime but absent from aic-sdk's type stub.
    set_sdk_id,  # type: ignore[attr-defined]
)
from loguru import logger
from pydantic import BaseModel

from pipecat.audio.filters.aic_filter import _AIC_SDK_PIPECAT_ID, AICFilter, AICModelManager
from pipecat.frames.frames import CancelFrame, EndFrame, InputAudioRawFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.utils.asyncio.task_manager import BaseTaskManager

DEFAULT_TYTO_MODEL_ID = "tyto-1.1-l-16khz"

# Tyto analyzes a fixed 5 s window and fills a shorter window with silence.
# The observer analyzes once per window, so the windows do not overlap.
_TYTO_WINDOW_SECS = 5.0

_INT16_SCALE = 32768.0


class AICAudioQualityScores(BaseModel):
    """Audio quality scores for one window of user audio.

    All scores are in the range 0.0 to 1.0. A higher value means more
    problematic audio.

    Parameters:
        risk_score: Likelihood that downstream models (STT, VAD, turn-taking,
            speech-to-speech) fail on this audio.
        noise: Ambient or environmental noise.
        speaker_reverb: Speaker distance and reverberation.
        speaker_loudness: Level of the main speaker. A neutral level, not a
            degradation.
        interfering_speech: Speech from other speakers, live or from a device.
        packet_loss: Dropouts or discontinuities in the stream.
        codec_degradation: Artifacts from lossy speech codecs.
    """

    risk_score: float
    noise: float
    speaker_reverb: float
    speaker_loudness: float
    interfering_speech: float
    packet_loss: float
    codec_degradation: float


class AICAudioQualitySummary(BaseModel):
    """Summary of the audio quality windows of one call.

    Parameters:
        windows: Number of analyzed windows.
        mean: Mean of each score over all windows.
        p95: 95th percentile of each score over all windows.
        max: Maximum of each score over all windows.
        degraded_threshold: ``risk_score`` value at or above which a window
            counts as degraded.
        degraded_fraction: Fraction of windows that are degraded.
        dominant_dimension: Name of the score with the highest mean, without
            ``risk_score`` and ``speaker_loudness``. It shows the largest
            degradation, not the cause of the risk score.
    """

    windows: int
    mean: AICAudioQualityScores
    p95: AICAudioQualityScores
    max: AICAudioQualityScores
    degraded_threshold: float
    degraded_fraction: float
    dominant_dimension: str


_SCORE_FIELDS = tuple(AICAudioQualityScores.model_fields)
# risk_score combines the other scores. speaker_loudness is a level, not a degradation.
_DEGRADATION_FIELDS = tuple(f for f in _SCORE_FIELDS if f not in ("risk_score", "speaker_loudness"))


class AICTytoObserver(BaseObserver):
    """Scores the quality of the user's audio with the ai-coustics Tyto model.

    The observer buffers the user's audio and analyzes the latest 5 s window
    in a background thread after every 5 s of received audio. Each result goes
    to the ``on_analysis_result`` event. When the call ends, the observer
    combines the results into one summary and gives it to the
    ``on_call_summary`` event.

    Tyto must score the audio before enhancement. When :class:`AICFilter` is the
    transport's ``audio_in_filter``, give the same filter as ``audio_filter``.
    The filter then feeds each block to the analysis before it enhances the
    block. Without a filter, the observer analyzes the ``InputAudioRawFrame``
    audio that the input transport pushes.

    Event handlers available:

    - on_analysis_result: Called with an :class:`AICAudioQualityScores` and the
      seconds of user audio received when the window was analyzed.
    - on_call_summary: Called with an :class:`AICAudioQualitySummary` when the
      call ends. Not called when no window was analyzed.

    Example::

        aic_filter = AICFilter(license_key=key, model_id="quail-vf-2.2-l-16khz")
        tyto = AICTytoObserver(license_key=key, audio_filter=aic_filter)

        @tyto.event_handler("on_call_summary")
        async def on_call_summary(observer, summary):
            logger.info(f"Mean risk score: {summary.mean.risk_score}")

        worker = PipelineWorker(pipeline, observers=[tyto])
    """

    def __init__(
        self,
        *,
        license_key: str,
        model_id: str | None = DEFAULT_TYTO_MODEL_ID,
        model_path: Path | None = None,
        model_download_dir: Path | None = None,
        degraded_threshold: float = 0.3,
        audio_filter: AICFilter | None = None,
        **kwargs,
    ):
        """Initialize the Tyto observer.

        Args:
            license_key: ai-coustics SDK license key.
            model_id: Tyto model identifier. Defaults to ``"tyto-1.1-l-16khz"``.
                Ignored if ``model_path`` is set.
            model_path: Optional path to a local ``.aicmodel`` file.
            model_download_dir: Directory for downloaded models. Defaults to
                ``~/.cache/pipecat/aic-models``.
            degraded_threshold: ``risk_score`` value at or above which the
                summary counts a window as degraded. Calibrate it against the
                failures that you see in your application.
            audio_filter: The transport's :class:`AICFilter`. If set, the filter
                feeds the analysis with the audio before enhancement.
            **kwargs: Additional arguments passed to parent class.

        Raises:
            ValueError: If neither ``model_id`` nor ``model_path`` is set, or
                if ``degraded_threshold`` is not between 0.0 and 1.0.
        """
        if model_id is None and model_path is None:
            raise ValueError(
                "Either 'model_id' or 'model_path' must be provided. "
                "See https://artifacts.ai-coustics.io/ for available models."
            )

        if not 0.0 <= degraded_threshold <= 1.0:
            raise ValueError("'degraded_threshold' must be between 0.0 and 1.0.")

        # Each audio frame is buffered once, on its push from the input transport.
        super().__init__(observe_every_push=False, **kwargs)

        set_sdk_id(_AIC_SDK_PIPECAT_ID)

        self._license_key = license_key
        self._model_id = model_id
        self._model_path = model_path
        self._model_download_dir = model_download_dir or (
            Path.home() / ".cache" / "pipecat" / "aic-models"
        )
        self._degraded_threshold = degraded_threshold
        self._audio_filter = audio_filter

        self._model: Model | None = None
        self._model_cache_key: str | None = None
        self._collector: Collector | None = None
        self._analyzer: Analyzer | None = None
        # The SDK requires serialized calls on one analyzer.
        self._executor: ThreadPoolExecutor | None = None
        self._analysis_task: asyncio.Task | None = None

        self._reset_session()

        self._register_event_handler("on_analysis_result")
        self._register_event_handler("on_call_summary")

    async def setup(self, task_manager: BaseTaskManager):
        """Load the Tyto model and start a new analysis session.

        A failure to load the model is logged, and the call goes on without
        analysis.

        Args:
            task_manager: The task manager that runs the analysis tasks.
        """
        await super().setup(task_manager)

        self._reset_session()

        try:
            self._model, self._model_cache_key = await AICModelManager.acquire(
                model_path=self._model_path,
                model_id=self._model_id,
                model_download_dir=self._model_download_dir,
            )
            self._collector, self._analyzer = analyzer_pair(self._model, self._license_key)
        except Exception as e:
            logger.error(f"{self}: Tyto setup failed, no audio quality analysis: {e}")
            self._release_model()
            return

        self._executor = ThreadPoolExecutor(max_workers=1)

        if self._audio_filter is not None:
            self._audio_filter.set_pre_process_tap(self._on_filter_block)

    async def cleanup(self):
        """Send the summary, end the analysis session and release the model."""
        # The worker stops the observer's frame queue at shutdown, so an
        # EndFrame or CancelFrame still in the queue never reaches
        # on_push_frame.
        await self._finish()

        if self._audio_filter is not None:
            self._audio_filter.set_pre_process_tap(None)

        if self._analyzer is not None and self._executor is not None:
            # On the analysis executor, so it cannot overlap an analysis that
            # is still running after its task was cancelled.
            try:
                await asyncio.get_running_loop().run_in_executor(
                    self._executor, self._analyzer.terminate_session
                )
            except Exception as e:  # noqa: BLE001 - teardown is best-effort
                logger.debug(f"{self}: Tyto session termination failed: {e}")

        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

        self._collector = None
        self._analyzer = None
        self._release_model()

        await super().cleanup()

    async def on_push_frame(self, data: FramePushed):
        """Buffer the user's audio and send the summary when the call ends.

        Args:
            data: The event data containing details about the frame transfer.
        """
        frame = data.frame
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._finish()
        elif isinstance(frame, InputAudioRawFrame) and self._audio_filter is None:
            self._handle_audio(frame)

    def _reset_session(self):
        # Zero until the first block of the session initializes the collector.
        self._frames_per_block = 0
        self._audio_buffer = bytearray()

        self._audio_secs = 0.0
        # Scores of a window that is part silence padding do not describe the
        # call, so the first analysis waits for a full window.
        self._next_analysis_secs = _TYTO_WINDOW_SECS
        self._windows: list[AICAudioQualityScores] = []
        self._finished = False

    def _release_model(self):
        self._model = None
        if self._model_cache_key is not None:
            AICModelManager.release(self._model_cache_key)
            self._model_cache_key = None

    def _start_analysis(self):
        if self._finished:
            return
        if self._analysis_task is not None and not self._analysis_task.done():
            return
        if self._audio_secs < self._next_analysis_secs:
            return
        self._next_analysis_secs = self._audio_secs + _TYTO_WINDOW_SECS
        self._analysis_task = self.create_task(self._analyze(self._audio_secs), "tyto_analysis")

    async def _stop_analysis(self):
        if self._analysis_task is not None:
            await self.cancel_task(self._analysis_task)
            self._analysis_task = None

    # Used without AICFilter. Buffers and counts the frame audio.
    def _handle_audio(self, frame: InputAudioRawFrame):
        # None when setup failed.
        if self._collector is None or self._model is None:
            return

        self._audio_secs += frame.num_frames / frame.sample_rate

        # The transport can override the pipeline's input sample rate, so the
        # first frame sets the collector configuration.
        if self._frames_per_block == 0:
            self._frames_per_block = self._model.get_optimal_block_size(frame.sample_rate)
            self._collector.initialize(ProcessorConfig(frame.sample_rate, self._frames_per_block))

        self._audio_buffer.extend(frame.audio)
        block_bytes = self._frames_per_block * 2
        num_blocks = len(self._audio_buffer) // block_bytes
        if num_blocks > 0:
            total_bytes = num_blocks * block_bytes
            samples = np.frombuffer(self._audio_buffer[:total_bytes], dtype=np.int16)
            del self._audio_buffer[:total_bytes]

            blocks = samples.reshape(num_blocks, self._frames_per_block).astype(np.float32)
            blocks /= _INT16_SCALE
            for block in blocks:
                self._collector.buffer(block)

        self._start_analysis()

    # Used with AICFilter. The filter calls this tap with each block before
    # enhancement. Buffers and counts the block. The filter does not call the
    # tap when it fails to initialize, so the observer then analyzes nothing.
    def _on_filter_block(self, block: np.ndarray, sample_rate: int):
        # None after cleanup.
        if self._collector is None:
            return

        if self._frames_per_block == 0:
            self._frames_per_block = len(block)
            self._collector.initialize(ProcessorConfig(sample_rate, self._frames_per_block))

        self._collector.buffer(block)
        self._audio_secs += len(block) / sample_rate
        self._start_analysis()

    async def _analyze(self, audio_secs: float):
        assert self._analyzer is not None
        started = time.monotonic()
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                self._executor, self._analyzer.analyze_buffered
            )
        except Exception as e:
            self._next_analysis_secs = math.inf
            logger.error(f"{self}: Tyto analysis failed, analysis stopped: {e}")
            return

        elapsed = time.monotonic() - started
        if elapsed > _TYTO_WINDOW_SECS:
            logger.warning(
                f"{self}: Tyto analysis took {elapsed:.2f}s, longer than its "
                f"{_TYTO_WINDOW_SECS}s interval. Reduce the CPU load."
            )

        await self._handle_result(result, audio_secs)

    async def _handle_result(self, result: AnalysisResult, audio_secs: float):
        scores = AICAudioQualityScores(**{f: getattr(result, f) for f in _SCORE_FIELDS})
        self._windows.append(scores)
        await self._call_event_handler("on_analysis_result", scores, audio_secs)

    async def _finish(self):
        if self._finished:
            return
        self._finished = True

        await self._stop_analysis()

        if not self._windows:
            logger.debug(f"{self}: no audio quality window analyzed, no summary")
            return

        await self._call_event_handler("on_call_summary", self._summarize())

    def _summarize(self) -> AICAudioQualitySummary:
        values = np.array([[getattr(w, f) for f in _SCORE_FIELDS] for w in self._windows])

        def to_scores(row: np.ndarray) -> AICAudioQualityScores:
            return AICAudioQualityScores(**dict(zip(_SCORE_FIELDS, row.tolist())))

        mean = values.mean(axis=0)
        risk = values[:, _SCORE_FIELDS.index("risk_score")]
        return AICAudioQualitySummary(
            windows=len(self._windows),
            mean=to_scores(mean),
            p95=to_scores(np.percentile(values, 95, axis=0)),
            max=to_scores(values.max(axis=0)),
            degraded_threshold=self._degraded_threshold,
            degraded_fraction=float(np.mean(risk >= self._degraded_threshold)),
            dominant_dimension=max(_DEGRADATION_FIELDS, key=lambda f: mean[_SCORE_FIELDS.index(f)]),
        )
