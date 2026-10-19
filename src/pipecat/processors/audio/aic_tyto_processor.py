#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audio quality analysis for Pipecat backed by the ai-coustics Tyto model.

Classes:
    AICTytoProcessor: Scores the user's audio during a call and summarizes the
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
    # Exported at runtime but absent from aic-sdk 3.0.0's type stub.
    set_sdk_id,  # type: ignore[attr-defined]
)
from loguru import logger

from pipecat.audio.filters.aic_filter import _AIC_SDK_PIPECAT_ID, AICFilter, AICModelManager
from pipecat.frames.frames import (
    AudioQualityFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    MetricsFrame,
    StartFrame,
)
from pipecat.metrics.metrics import (
    AudioQualityMetricsData,
    AudioQualityScores,
    AudioQualitySummary,
    AudioQualitySummaryMetricsData,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup

DEFAULT_TYTO_MODEL_ID = "tyto-1.1-l-16khz"

# Tyto analyzes a fixed 5 s window and fills a shorter window with silence.
# The processor analyzes once per window, so the windows do not overlap.
_TYTO_WINDOW_SECS = 5.0

_INT16_SCALE = 32768.0

_SCORE_FIELDS = tuple(AudioQualityScores.model_fields)
# risk_score combines the other scores. speaker_loudness is a level, not a degradation.
_DEGRADATION_FIELDS = tuple(f for f in _SCORE_FIELDS if f not in ("risk_score", "speaker_loudness"))


class AICTytoProcessor(FrameProcessor):
    """Scores the quality of the user's audio with the ai-coustics Tyto model.

    The processor buffers the user's audio and analyzes the latest 5 s window
    in a background thread after every 5 s of received audio. Each
    ``StartFrame`` starts a new session with no results. Each result goes to the
    ``on_analysis_result`` event and downstream in an
    :class:`~pipecat.frames.frames.AudioQualityFrame`. When the call ends, the processor combines
    the results into one summary and gives it to the ``on_call_summary``
    event. When pipeline metrics are enabled, the processor also pushes each
    result and the summary in a ``MetricsFrame``. On ``CancelFrame`` it calls
    the event but pushes no frame.

    Place the processor after ``transport.input()``. Tyto must score the audio
    before enhancement. When :class:`AICFilter` is the transport's
    ``audio_in_filter``, give the same filter as ``audio_filter``. The filter
    then feeds each block to the analysis before it enhances the block.

    Event handlers available:

    - on_analysis_result: Called with an :class:`AudioQualityMetricsData` for
      each analyzed window.
    - on_call_summary: Called with an :class:`AudioQualitySummaryMetricsData`
      when the call ends. Not called when no window was analyzed.

    Example::

        aic_filter = AICFilter(license_key=key, model_id="quail-vf-2.2-l-16khz")
        tyto = AICTytoProcessor(license_key=key, audio_filter=aic_filter)

        @tyto.event_handler("on_call_summary")
        async def on_call_summary(processor, data):
            logger.info(f"Mean risk score: {data.value.mean.risk_score}")
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
        """Initialize the Tyto processor.

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

        super().__init__(**kwargs)

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

    async def setup(self, setup: FrameProcessorSetup):
        """Load the Tyto model and create the analysis session.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._model, self._model_cache_key = await AICModelManager.acquire(
            model_path=self._model_path,
            model_id=self._model_id,
            model_download_dir=self._model_download_dir,
        )
        self._collector, self._analyzer = analyzer_pair(self._model, self._license_key)

    async def cleanup(self):
        """Stop the analysis, end the analysis session and release the model."""
        await self._stop_analysis()
        await super().cleanup()

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
        self._model = None

        if self._model_cache_key is not None:
            AICModelManager.release(self._model_cache_key)
            self._model_cache_key = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame and forward it.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        # The summary goes before the EndFrame so that it reaches the output
        # transport before the transport stops.
        if isinstance(frame, EndFrame):
            await self._stop_analysis()
            await self._send_summary(push_metrics=True)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, CancelFrame):
            await self._stop_analysis()
            await self._send_summary(push_metrics=False)
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._stop_analysis()
            self._reset_session()
            if self._audio_filter is not None:
                self._audio_filter.set_pre_process_tap(self._on_filter_block)
        elif isinstance(frame, InputAudioRawFrame):
            self._handle_audio(frame)
            self._start_analysis()

    def _reset_session(self):
        # Zero until the first block of the session initializes the collector.
        self._frames_per_block = 0
        self._audio_buffer = bytearray()

        self._audio_secs = 0.0
        # Scores of a window that is part silence padding do not describe the
        # call, so the first analysis waits for a full window.
        self._next_analysis_secs = _TYTO_WINDOW_SECS
        self._windows: list[AudioQualityScores] = []
        self._summary_sent = False

    def _start_analysis(self):
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

    def _handle_audio(self, frame: InputAudioRawFrame):
        # None when setup failed.
        if self._collector is None or self._model is None:
            return

        self._audio_secs += frame.num_frames / frame.sample_rate

        if self._audio_filter is not None:
            return

        # The transport can override the pipeline's input sample rate, so the
        # first frame sets the collector configuration.
        if self._frames_per_block == 0:
            self._frames_per_block = self._model.get_optimal_block_size(frame.sample_rate)
            self._collector.initialize(ProcessorConfig(frame.sample_rate, self._frames_per_block))

        self._audio_buffer.extend(frame.audio)
        block_bytes = self._frames_per_block * 2
        num_blocks = len(self._audio_buffer) // block_bytes
        if num_blocks == 0:
            return

        total_bytes = num_blocks * block_bytes
        samples = np.frombuffer(self._audio_buffer[:total_bytes], dtype=np.int16)
        del self._audio_buffer[:total_bytes]

        blocks = samples.reshape(num_blocks, self._frames_per_block).astype(np.float32)
        blocks /= _INT16_SCALE
        for block in blocks:
            self._collector.buffer(block)

    def _on_filter_block(self, block: np.ndarray, sample_rate: int):
        # None when setup failed.
        if self._collector is None:
            return

        if self._frames_per_block == 0:
            self._frames_per_block = len(block)
            self._collector.initialize(ProcessorConfig(sample_rate, self._frames_per_block))

        self._collector.buffer(block)

    async def _analyze(self, audio_secs: float):
        assert self._analyzer is not None
        started = time.monotonic()
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                self._executor, self._analyzer.analyze_buffered
            )
        except Exception as e:
            self._next_analysis_secs = math.inf
            await self.push_error(f"Tyto analysis failed, analysis stopped: {e}", exception=e)
            return

        elapsed = time.monotonic() - started
        if elapsed > _TYTO_WINDOW_SECS:
            logger.warning(
                f"{self}: Tyto analysis took {elapsed:.2f}s, longer than its "
                f"{_TYTO_WINDOW_SECS}s interval. Reduce the CPU load."
            )

        await self._handle_result(result, audio_secs)

    async def _handle_result(self, result: AnalysisResult, audio_secs: float):
        assert self._model is not None
        scores = AudioQualityScores(**{f: getattr(result, f) for f in _SCORE_FIELDS})
        self._windows.append(scores)

        data = AudioQualityMetricsData(
            processor=self.name, model=self._model.get_id(), value=scores, audio_secs=audio_secs
        )
        await self._call_event_handler("on_analysis_result", data)
        await self.push_frame(
            AudioQualityFrame(
                risk_score=scores.risk_score,
                degradations={f: getattr(scores, f) for f in _DEGRADATION_FIELDS},
                audio_secs=audio_secs,
            )
        )
        if self.metrics_enabled:
            await self.push_frame(MetricsFrame(data=[data]))

    async def _send_summary(self, *, push_metrics: bool):
        if self._summary_sent:
            return
        self._summary_sent = True

        if not self._windows:
            logger.debug(f"{self}: no audio quality window analyzed, no summary")
            return

        assert self._model is not None
        data = AudioQualitySummaryMetricsData(
            processor=self.name, model=self._model.get_id(), value=self._summarize()
        )
        await self._call_event_handler("on_call_summary", data)
        if push_metrics and self.metrics_enabled:
            await self.push_frame(MetricsFrame(data=[data]))

    def _summarize(self) -> AudioQualitySummary:
        values = np.array([[getattr(w, f) for f in _SCORE_FIELDS] for w in self._windows])

        def to_scores(row: np.ndarray) -> AudioQualityScores:
            return AudioQualityScores(**dict(zip(_SCORE_FIELDS, row.tolist())))

        mean = values.mean(axis=0)
        risk = values[:, _SCORE_FIELDS.index("risk_score")]
        return AudioQualitySummary(
            windows=len(self._windows),
            mean=to_scores(mean),
            p95=to_scores(np.percentile(values, 95, axis=0)),
            max=to_scores(values.max(axis=0)),
            degraded_threshold=self._degraded_threshold,
            degraded_fraction=float(np.mean(risk >= self._degraded_threshold)),
            dominant_dimension=max(_DEGRADATION_FIELDS, key=lambda f: mean[_SCORE_FIELDS.index(f)]),
        )
