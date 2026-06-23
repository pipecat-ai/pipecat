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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from aic_sdk import (
    Model,
    ProcessorConfig,
    analyzer_pair,
    set_sdk_id,
)
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    MetricsFrame,
    StartFrame,
)
from pipecat.metrics.metrics import AICAudioQualityMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

if TYPE_CHECKING:
    from aic_sdk import AnalysisResult, Analyzer, Collector

DEFAULT_TYTO_MODEL_ID = "tyto-l-16khz"

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

    Place it wherever the audio you want to score flows: right after
    ``transport.input()`` to score the raw microphone signal, or after an
    :class:`pipecat.audio.filters.aic_filter.AICFilter` to score enhanced audio.

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
                ``"tyto-l-16khz"``. See https://artifacts.ai-coustics.io/ for the
                catalogue. Ignored if ``model_path`` is provided.
            model_path: Optional path to a local ``.aicmodel`` file. Overrides
                ``model_id`` when set.
            model_download_dir: Directory for downloaded models. Defaults to
                ``~/.cache/pipecat/aic-models``.
            analysis_interval: Seconds between analysis runs. Defaults to 1.0.
            **kwargs: Additional arguments passed to :class:`FrameProcessor`.

        Raises:
            ValueError: If neither ``model_id`` nor ``model_path`` is provided.
        """
        if model_id is None and model_path is None:
            raise ValueError(
                "Either 'model_id' or 'model_path' must be provided. "
                "See https://artifacts.ai-coustics.io/ for available models."
            )

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
        self._analysis_task: asyncio.Task | None = None
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

    def _initialize_collector(self, sample_rate: int, num_channels: int) -> None:
        self._ensure_model_loaded()
        assert self._model is not None

        collector, analyzer = analyzer_pair(self._model, self._license_key)
        # allow_variable_frames so we can buffer whatever chunk size the
        # transport delivers per InputAudioRawFrame without re-blocking.
        config = ProcessorConfig.optimal(
            self._model,
            sample_rate=sample_rate,
            num_channels=num_channels,
            allow_variable_frames=True,
        )
        collector.initialize(config)

        self._collector = collector
        self._analyzer = analyzer
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._analysis_error_logged = False
        logger.debug(f"AICTytoAnalyzer initialized at {sample_rate} Hz, {num_channels} channel(s)")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Forward every frame unchanged and tap input audio for analysis.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        # Passive tap: forward first so audio/control flow is never delayed by
        # the analysis side-channel.
        await self.push_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._start()
        elif isinstance(frame, InputAudioRawFrame):
            self._buffer_audio(frame)

    def _start(self) -> None:
        if self._analysis_task is None:
            self._analysis_task = self.create_task(self._analysis_loop(), f"{self}::analysis_loop")

    def _buffer_audio(self, frame: InputAudioRawFrame) -> None:
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
        # Collector expects a 2D (channels, frames) array; de-interleave for
        # multi-channel input (the model mixes to mono internally).
        if channels > 1:
            audio = samples.reshape(-1, channels).T
        else:
            audio = samples.reshape(1, -1)
        try:
            self._collector.buffer(audio)
        except Exception as e:  # noqa: BLE001 - keep the pipeline alive on SDK errors
            if not self._analysis_error_logged:
                logger.error(f"Tyto buffering error: {e}")
                self._analysis_error_logged = True
            else:
                logger.debug(f"Tyto buffering error: {e}")

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
        if analyzer is None:
            return
        loop = asyncio.get_running_loop()
        try:
            result: AnalysisResult = await loop.run_in_executor(
                self._executor, analyzer.analyze_buffered
            )
            # Successful analysis re-arms the error latch.
            self._analysis_error_logged = False
        except Exception as e:  # noqa: BLE001 - keep the pipeline alive on SDK errors
            if not self._analysis_error_logged:
                logger.error(f"Tyto analysis error: {e}")
                self._analysis_error_logged = True
            else:
                logger.debug(f"Tyto analysis error: {e}")
            return

        data = self._build_metrics(result)
        await self.push_frame(MetricsFrame(data=[data]))
        await self._call_event_handler("on_audio_analysis", data)

    def _build_metrics(self, result: AnalysisResult) -> AICAudioQualityMetricsData:
        return AICAudioQualityMetricsData(
            processor=self.name,
            model=self._model_id,
            risk_score=result.risk_score,
            speaker_reverb=result.speaker_reverb,
            speaker_loudness=result.speaker_loudness,
            interfering_speech=result.interfering_speech,
            media_speech=result.media_speech,
            noise=result.noise,
            packet_loss=result.packet_loss,
        )

    async def cleanup(self) -> None:
        """Cancel the analysis task and release the SDK handles."""
        await super().cleanup()
        if self._analysis_task is not None:
            await self.cancel_task(self._analysis_task)
            self._analysis_task = None
        try:
            self._executor.shutdown(wait=False)
        except Exception as e:  # noqa: BLE001 - cleanup is best-effort
            logger.debug(f"AICTytoAnalyzer executor shutdown failed: {e}")
        self._collector = None
        self._analyzer = None
        self._model = None
