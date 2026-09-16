#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Real-time audio-quality analyzer powered by the ai-coustics Tyto model.

The Tyto analysis model scores incoming audio to predict how likely it is to
degrade downstream models (speech-to-text, VAD, turn-taking, speech-to-speech).
:class:`AICTytoAnalyzer` taps the pipeline's input audio, buffers it into the
SDK's :class:`aic_sdk.Collector`, and periodically runs the (computationally
expensive, non-real-time-safe) analysis off the event loop, emitting an
:class:`pipecat.metrics.metrics.AICAudioQualityMetricsData` via a
:class:`MetricsFrame` and an ``on_audio_analysis`` event.

Classes:
    AICTytoAnalyzer: Periodic audio-quality analysis FrameProcessor.
"""

from __future__ import annotations

import asyncio
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from aic_sdk import (
    Model,
    ProcessorConfig,
    analyzer_pair,
    # Exported at runtime but absent from the SDK type stub.
    set_sdk_id,  # type: ignore[attr-defined]
)
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    FilterControlFrame,
    Frame,
    InputAudioRawFrame,
    MetricsFrame,
    StartFrame,
    StopFrame,
)
from pipecat.metrics.metrics import AICAudioQualityMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup

if TYPE_CHECKING:
    from aic_sdk import AnalysisResult, Analyzer, Collector

DEFAULT_TYTO_MODEL_ID = "tyto-1.1-l-16khz"

# Telemetry identifier registered with the AIC SDK; identifies pipecat to the
# vendor's usage pipeline. Mirrors the value used by AICFilter / AICQuailVADAnalyzer.
_AIC_SDK_PIPECAT_ID = 6

# 2^15: normalizes int16 samples (-32768..32767) to float32 (-1.0..0.99997).
_INT16_DTYPE = np.int16
_INT16_SCALE = 32768.0


class AICTytoAnalyzer(FrameProcessor):
    """Periodic audio-quality analysis using the ai-coustics Tyto model.

    The processor is a passive tap: every frame it receives is forwarded
    unchanged in its original direction. On the side, it converts each
    :class:`InputAudioRawFrame` to float32 and buffers it into the SDK
    :class:`aic_sdk.Collector` (audio-thread safe), while a background task runs
    :meth:`aic_sdk.Analyzer.analyze_buffered` every ``analysis_interval`` seconds
    on a dedicated thread (the analysis is not real-time safe). Each result is
    published as an :class:`AICAudioQualityMetricsData` in a :class:`MetricsFrame`
    and dispatched to ``on_audio_analysis`` handlers.

    Place it after ``transport.input()`` to score the transport's output audio.
    To score original audio before enhancement, install :meth:`as_input_filter`
    as the transport's ``audio_in_filter`` and also keep this processor in the
    pipeline. Use one analyzer per input stream. Results begin after five seconds
    of collected audio; scores describe overlapping five-second windows.

    Event handlers:

    - on_audio_analysis: Called with the :class:`AICAudioQualityMetricsData` for
      each completed analysis.

    Example::

        analyzer = AICTytoAnalyzer(license_key=os.environ["AIC_SDK_LICENSE"])

        @analyzer.event_handler("on_audio_analysis")
        async def on_audio_analysis(processor, scores):
            logger.info(f"risk={scores.risk_score:.2f} noise={scores.noise:.2f}")

        pipeline = Pipeline([transport.input(), analyzer, ...])
    """

    def __init__(
        self,
        *,
        license_key: str,
        model_id: str | None = DEFAULT_TYTO_MODEL_ID,
        model_path: Path | None = None,
        model_download_dir: Path | None = None,
        analysis_interval: float = 1.0,
        **kwargs,
    ) -> None:
        """Initialize the Tyto audio-quality analyzer.

        Loads the model eagerly so the cold-start CDN download happens at
        construction time (typically before the event loop starts) rather than
        on the first audio frame.

        Args:
            license_key: ai-coustics SDK license key.
            model_id: Tyto analysis model identifier. Defaults to
                ``"tyto-1.1-l-16khz"``. See https://artifacts.ai-coustics.io/ for the
                catalogue. Ignored if ``model_path`` is provided.
            model_path: Optional path to a local ``.aicmodel`` file. Overrides
                ``model_id`` when set.
            model_download_dir: Directory for downloaded models. Defaults to
                ``~/.cache/pipecat/aic-models``.
            analysis_interval: Seconds between analysis runs. Defaults to 1.0.
            **kwargs: Additional arguments passed to :class:`FrameProcessor`.

        Raises:
            ValueError: If no model is provided or the analysis interval is not
                finite and positive.
        """
        if model_id is None and model_path is None:
            raise ValueError(
                "Either 'model_id' or 'model_path' must be provided. "
                "See https://artifacts.ai-coustics.io/ for available models."
            )

        if not math.isfinite(analysis_interval) or analysis_interval <= 0:
            raise ValueError("analysis_interval must be finite and greater than zero")

        super().__init__(**kwargs)

        self._license_key = license_key
        self._model_id = model_id
        self._model_path = model_path
        self._model_download_dir = model_download_dir or (
            Path.home() / ".cache" / "pipecat" / "aic-models"
        )
        self._analysis_interval = analysis_interval

        self._model: Model | None = None
        self._collector: Collector | None = None
        self._analyzer: Analyzer | None = None
        self._sample_rate = 0
        self._num_channels = 0
        self._block_size = 0
        self._analysis_task: asyncio.Task | None = None
        self._input_filter: _TytoInputFilter | None = None
        self._buffered_samples = 0
        self._last_analyzed_samples = 0
        self._generation = 0
        self._inference: asyncio.Future | None = None
        self._sequence = 0
        self._closed = False
        # Latch: log analysis errors at ERROR once, then DEBUG until a success
        # re-arms it (so a recovery followed by a new failure surfaces again).
        self._analysis_error_logged = False

        # Blocking analysis runs here, off the event loop. analyze_buffered() is
        # not real-time safe, so it must never run on the audio/event-loop path.
        self._executor = ThreadPoolExecutor(max_workers=1)

        self._register_event_handler("on_audio_analysis")

        # Eager model load shifts the CDN download out of the hot path. If it
        # raises, shut down the executor so the half-constructed instance does
        # not leak its worker thread, then propagate.
        try:
            set_sdk_id(_AIC_SDK_PIPECAT_ID)
            self._ensure_model_loaded()
        except Exception:
            try:
                self._executor.shutdown(wait=False)
            except Exception as e:  # noqa: BLE001 - executor cleanup is best-effort
                logger.debug(f"AICTytoAnalyzer executor shutdown failed: {e}")
            raise

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        if self._model_path is not None:
            logger.debug(f"Loading Tyto model from file: {self._model_path}")
            self._model = Model.from_file(str(self._model_path))
            return
        # model_id path (validated in __init__).
        assert self._model_id is not None
        self._model_download_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Downloading Tyto model {self._model_id!r} to {self._model_download_dir}")
        model_path = Model.download(self._model_id, str(self._model_download_dir))
        self._model = Model.from_file(model_path)

    async def setup(self, setup: FrameProcessorSetup) -> None:
        """Initialize the SDK outside the audio-processing path."""
        await super().setup(setup)
        if self._input_filter is None:
            await asyncio.get_running_loop().run_in_executor(
                self._executor, self._initialize_collector, setup.audio_in_sample_rate, 1
            )

    def _initialize_collector(self, sample_rate: int, num_channels: int) -> None:
        self._ensure_model_loaded()
        assert self._model is not None

        if self._collector is None:
            self._collector, self._analyzer = analyzer_pair(self._model, self._license_key)
        collector = self._collector
        assert collector is not None
        # Short blocks are accepted; larger transport frames are split below.
        config = ProcessorConfig.optimal(
            self._model,
            sample_rate=sample_rate,
            variable_block_size=True,
        )
        collector.initialize(config)
        self._block_size = config.block_size

        self.reset()
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._analysis_error_logged = False
        logger.debug(f"AICTytoAnalyzer initialized at {sample_rate} Hz, {num_channels} channel(s)")

    def as_input_filter(
        self, audio_filter: BaseAudioFilter | None = None, *, num_channels: int = 1
    ) -> BaseAudioFilter:
        """Collect original transport audio before an optional enhancement filter.

        The analyzer must also appear in the pipeline to own background analysis
        and publish metrics. Configure this adapter before starting the pipeline.

        Args:
            audio_filter: Optional filter to run after collecting original audio.
            num_channels: Transport input channel count.

        Returns:
            Adapter for ``TransportParams.audio_in_filter``.
        """
        if self._input_filter is not None:
            raise ValueError("An input filter is already attached to this analyzer")
        if num_channels < 1:
            raise ValueError("num_channels must be positive")
        self._input_filter = _TytoInputFilter(self, audio_filter, num_channels)
        return self._input_filter

    def reset(self) -> None:
        """Clear audio history after a stream discontinuity or between calls."""
        if self._analyzer is not None:
            self._analyzer.reset()
        self._generation += 1
        self._buffered_samples = 0
        self._last_analyzed_samples = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Forward every frame unchanged and tap input audio for analysis.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._start()
        elif isinstance(frame, (EndFrame, CancelFrame, StopFrame)):
            await self._stop_analysis()
            self.reset()
        elif (
            isinstance(frame, InputAudioRawFrame)
            and direction == FrameDirection.DOWNSTREAM
            and self._input_filter is None
        ):
            self._buffer_audio(frame)
        await self.push_frame(frame, direction)

    def _start(self) -> None:
        if self._analysis_task is None:
            self._analysis_task = self.create_task(self._analysis_loop(), f"{self}::analysis_loop")

    def _buffer_audio(self, frame: InputAudioRawFrame) -> None:
        if self._closed:
            return
        try:
            self._collect_audio(frame)
        except Exception as e:
            if not self._analysis_error_logged:
                logger.error(f"Tyto buffering error: {e}")
                self._analysis_error_logged = True
            else:
                logger.debug(f"Tyto buffering error: {e}")

    def _collect_audio(self, frame: InputAudioRawFrame) -> None:
        channels = frame.num_channels or 1
        # Lazily (re)initialize once the concrete rate/channel layout is known.
        if (
            self._collector is None
            or frame.sample_rate != self._sample_rate
            or channels != self._num_channels
        ):
            self._initialize_collector(frame.sample_rate, channels)
        assert self._collector is not None

        samples = np.frombuffer(frame.audio, dtype=_INT16_DTYPE).astype(np.float32)
        samples /= _INT16_SCALE
        # The SDK accepts mono audio only.
        audio = samples.reshape(-1, channels).mean(axis=1) if channels > 1 else samples
        for offset in range(0, len(audio), self._block_size):
            block = audio[offset : offset + self._block_size]
            self._collector.buffer(block)
            self._buffered_samples += len(block)

    async def _analysis_loop(self) -> None:
        while True:
            await asyncio.sleep(self._analysis_interval)
            await self._analyze_once()

    async def _analyze_once(self) -> None:
        """Run one analysis pass and publish the result.

        The analysis runs off the event loop (it is not real-time safe); SDK
        errors are latched and swallowed so the pipeline stays alive.
        """
        analyzer = self._analyzer
        if (
            analyzer is None
            or self._sample_rate <= 0
            or self._buffered_samples < 5 * self._sample_rate
            or self._buffered_samples == self._last_analyzed_samples
        ):
            return
        generation = self._generation
        self._last_analyzed_samples = self._buffered_samples
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        try:
            self._inference = loop.run_in_executor(self._executor, analyzer.analyze_buffered)
            result: AnalysisResult = await asyncio.shield(self._inference)
            # Successful analysis re-arms the error latch.
            self._analysis_error_logged = False
        except Exception as e:  # noqa: BLE001 - keep the pipeline alive on SDK errors
            if not self._analysis_error_logged:
                logger.error(f"Tyto analysis error: {e}")
                self._analysis_error_logged = True
            else:
                logger.debug(f"Tyto analysis error: {e}")
            return

        if generation != self._generation:
            return
        self._sequence += 1
        data = self._build_metrics(result)
        data.sequence = self._sequence
        data.timestamp = time.time()
        data.inference_duration = time.perf_counter() - started
        await self.push_frame(MetricsFrame(data=[data]))
        await self._call_event_handler("on_audio_analysis", data)

    def _build_metrics(self, result: AnalysisResult) -> AICAudioQualityMetricsData:
        return AICAudioQualityMetricsData(
            processor=self.name,
            model=self._model.get_id() if self._model else self._model_id,
            risk_score=result.risk_score,
            speaker_reverb=result.speaker_reverb,
            speaker_loudness=result.speaker_loudness,
            interfering_speech=result.interfering_speech,
            codec_degradation=result.codec_degradation,
            noise=result.noise,
            packet_loss=result.packet_loss,
        )

    async def _stop_analysis(self) -> None:
        if self._analysis_task is not None:
            await self.cancel_task(self._analysis_task)
            self._analysis_task = None
        if self._inference is not None:
            try:
                await asyncio.shield(self._inference)
            except Exception:
                pass
            self._inference = None

    async def cleanup(self) -> None:
        """Drain native inference before terminating the SDK session."""
        self._closed = True
        await self._stop_analysis()
        if self._analyzer is not None:
            try:
                await asyncio.get_running_loop().run_in_executor(
                    self._executor, self._analyzer.terminate_session
                )
            except Exception as e:
                logger.debug(f"Tyto session termination failed: {e}")
        try:
            self._executor.shutdown(wait=False)
        except Exception as e:  # noqa: BLE001 - cleanup is best-effort
            logger.debug(f"AICTytoAnalyzer executor shutdown failed: {e}")
        self._collector = None
        self._analyzer = None
        self._model = None
        await super().cleanup()


class _TytoInputFilter(BaseAudioFilter):
    """Transport adapter that collects PCM before optional enhancement."""

    def __init__(
        self, analyzer: AICTytoAnalyzer, audio_filter: BaseAudioFilter | None, num_channels: int
    ) -> None:
        self._analyzer = analyzer
        self._filter = audio_filter
        self._num_channels = num_channels
        self._sample_rate = 0

    async def start(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        await asyncio.get_running_loop().run_in_executor(
            self._analyzer._executor,
            self._analyzer._initialize_collector,
            sample_rate,
            self._num_channels,
        )
        if self._filter:
            await self._filter.start(sample_rate)

    async def stop(self) -> None:
        if self._filter:
            await self._filter.stop()

    async def process_frame(self, frame: FilterControlFrame) -> None:
        if self._filter:
            await self._filter.process_frame(frame)

    async def filter(self, audio: bytes) -> bytes:
        self._analyzer._buffer_audio(
            InputAudioRawFrame(
                audio=audio, sample_rate=self._sample_rate, num_channels=self._num_channels
            )
        )
        return await self._filter.filter(audio) if self._filter else audio
