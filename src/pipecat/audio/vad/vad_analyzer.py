#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice Activity Detection (VAD) analyzer base classes and utilities.

This module provides the abstract base class for VAD analyzers and associated
data structures for voice activity detection in audio streams. Includes state
management, parameter configuration, and audio analysis framework.
"""

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

from pipecat.audio.utils import exp_smoothing
from pipecat.audio.volume import AudioVolumeTracker

VAD_CONFIDENCE = 0.7
VAD_START_SECS = 0.2
VAD_STOP_SECS = 0.2
VAD_MIN_VOLUME = 0.6


class VADState(Enum):
    """Voice Activity Detection states.

    Parameters:
        QUIET: No voice activity detected.
        STARTING: Voice activity beginning, transitioning from quiet.
        SPEAKING: Active voice detected and confirmed.
        STOPPING: Voice activity ending, transitioning to quiet.
    """

    QUIET = 1
    STARTING = 2
    SPEAKING = 3
    STOPPING = 4


class VADParams(BaseModel):
    """Configuration parameters for Voice Activity Detection.

    ``hold_confidence`` adds hysteresis. Speech has to reach ``confidence``
    to start, then only has to stay at or above the lower ``hold_confidence``
    to continue, so a segment survives confidence flickering around the start
    threshold. Leave ``hold_confidence`` unset to use ``confidence`` and
    ``min_volume`` for the whole segment.

    Assigning either threshold checks it against the other field's current
    value. Pass a new ``VADParams`` to :meth:`VADAnalyzer.set_params` to change
    them together.

    Parameters:
        confidence: Confidence required to begin speech.
        hold_confidence: Confidence required to keep a segment going after it
            has started. ``None`` uses ``confidence`` instead. When set, the
            value must satisfy ``0.0 <= hold_confidence <= confidence <= 1.0``.
            ``min_volume`` is not applied for the rest of that segment.
        start_secs: How long speech must hold before it counts as started.
        stop_secs: How long speech must stay below the stop threshold before
            it counts as stopped.
        min_volume: Volume required to begin speech. Also required to continue
            speech when ``hold_confidence`` is unset.

    Example::

        params = VADParams(
            confidence=0.55,
            hold_confidence=0.35,
            start_secs=0.2,
            stop_secs=0.8,
            min_volume=0.6,
        )
    """

    model_config = ConfigDict(validate_assignment=True)

    confidence: float = VAD_CONFIDENCE
    hold_confidence: float | None = None
    start_secs: float = VAD_START_SECS
    stop_secs: float = VAD_STOP_SECS
    min_volume: float = VAD_MIN_VOLUME

    @field_validator("confidence")
    @classmethod
    def _check_confidence(cls, value: float, info: ValidationInfo) -> float:
        """Reject a start threshold that the hold threshold sits outside.

        ``hold_confidence`` is defined after this field, so on construction it
        is not available yet and the hold validator performs the joint check.
        Assignment of ``confidence`` sees the current hold threshold.

        Args:
            value: The confidence value being set.
            info: The other fields already validated on this model.

        Returns:
            The accepted confidence.

        Raises:
            ValueError: If ``hold_confidence`` is already set and the new
                confidence breaks ``0.0 <= hold_confidence <= confidence <= 1.0``.
        """
        hold = info.data.get("hold_confidence")
        if hold is not None and not 0.0 <= hold <= value <= 1.0:
            raise ValueError(
                "VADParams.hold_confidence must satisfy "
                "0.0 <= hold_confidence <= confidence <= 1.0 "
                f"(hold_confidence={hold}, confidence={value})"
            )
        return value

    @field_validator("hold_confidence")
    @classmethod
    def _check_hold_confidence(cls, value: float | None, info: ValidationInfo) -> float | None:
        """Reject a hold threshold outside the start threshold.

        Hysteresis is optional. With ``hold_confidence`` unset, ``confidence``
        is not range-checked.

        Args:
            value: The hold threshold being set, or ``None`` when hysteresis
                is off.
            info: The other fields already validated on this model.

        Returns:
            The accepted hold threshold.

        Raises:
            ValueError: If ``value`` is set and does not satisfy
                ``0.0 <= hold_confidence <= confidence <= 1.0``.
        """
        if value is None:
            return value
        confidence = info.data.get("confidence")
        if confidence is not None and not 0.0 <= value <= confidence <= 1.0:
            raise ValueError(
                "VADParams.hold_confidence must satisfy "
                "0.0 <= hold_confidence <= confidence <= 1.0 "
                f"(hold_confidence={value}, confidence={confidence})"
            )
        return value


class VADAnalyzer(ABC):
    """Abstract base class for Voice Activity Detection analyzers.

    Provides the framework for implementing VAD analysis with configurable
    parameters, state management, and audio processing capabilities.
    Subclasses implement voice confidence. This class decides when speech
    starts and stops.

    From quiet, confidence and volume must both stay at or above ``confidence``
    and ``min_volume`` for ``start_secs`` (passing through ``STARTING``) before
    the state becomes speaking. It then stays speaking while each window is at
    or above the stop threshold, and becomes quiet after windows stay below
    that threshold for ``stop_secs`` (passing through ``STOPPING``). A window
    at or above the threshold during ``STOPPING`` returns to speaking
    immediately.

    The stop threshold is ``hold_confidence`` when that parameter is set, and
    ``confidence`` together with ``min_volume`` otherwise. While
    ``hold_confidence`` is in use, volume does not end a segment that has
    already started.
    """

    def __init__(self, *, sample_rate: int | None = None, params: VADParams | None = None):
        """Initialize the VAD analyzer.

        Args:
            sample_rate: Audio sample rate in Hz. If None, will be set later.
            params: VAD parameters for detection configuration.
        """
        self._init_sample_rate = sample_rate
        self._sample_rate = 0
        self._params = params or VADParams()
        self._num_channels = 1

        self._vad_buffer = b""
        self._volume_tracker = AudioVolumeTracker()

        # Volume exponential smoothing
        self._smoothing_factor = 0.2
        self._prev_volume = 0.0

        # Thread executor that will run the model. We only need one thread per
        # analyzer because one analyzer just handles one audio stream.
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def sample_rate(self) -> int:
        """Get the current sample rate.

        Returns:
            Current audio sample rate in Hz.
        """
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        """Get the number of audio channels.

        Returns:
            Number of audio channels (always 1 for mono).
        """
        return self._num_channels

    @property
    def params(self) -> VADParams:
        """Get the current VAD parameters.

        Returns:
            Current VAD configuration parameters.
        """
        return self._params

    @abstractmethod
    def num_frames_required(self) -> int:
        """Get the number of audio frames required for analysis.

        Returns:
            Number of frames needed for VAD processing.
        """
        pass

    @abstractmethod
    def voice_confidence(self, buffer: bytes) -> float:
        """Calculate voice activity confidence for the given audio buffer.

        Args:
            buffer: Audio buffer to analyze.

        Returns:
            Voice confidence score between 0.0 and 1.0.
        """
        pass

    def set_sample_rate(self, sample_rate: int):
        """Set the sample rate for audio processing.

        Args:
            sample_rate: Audio sample rate in Hz.
        """
        self._sample_rate = self._init_sample_rate or sample_rate
        self.set_params(self._params)

    def set_params(self, params: VADParams):
        """Set VAD parameters and recalculate internal values.

        Args:
            params: VAD parameters for detection configuration.

        Raises:
            ValidationError: If ``hold_confidence`` is set and does not satisfy
                ``0.0 <= hold_confidence <= confidence <= 1.0``. A rejected
                update leaves the analyzer's current parameters in place.
        """
        # model_validate returns an existing VADParams without checking it.
        # A dump runs the thresholds through validation again, which catches an
        # instance built with model_construct.
        VADParams.model_validate(params.model_dump())
        logger.debug(f"Setting VAD params to: {params}")
        self._params = params
        self._vad_frames = self.num_frames_required()
        self._vad_frames_num_bytes = self._vad_frames * self._num_channels * 2

        vad_frames_per_sec = self._vad_frames / self.sample_rate

        self._vad_start_frames = round(self._params.start_secs / vad_frames_per_sec)
        self._vad_stop_frames = round(self._params.stop_secs / vad_frames_per_sec)
        # VAD state resets, but volume state doesn't: the rolling window and its
        # smoothing follow the audio stream, which is continuous across
        # parameter changes.
        self._vad_starting_count = 0
        self._vad_stopping_count = 0
        self._vad_state: VADState = VADState.QUIET

    def _get_smoothed_volume(self, audio: bytes) -> float:
        """Calculate smoothed audio volume using exponential smoothing."""
        self._volume_tracker.update(audio, self.sample_rate)
        return exp_smoothing(self._volume_tracker.volume, self._prev_volume, self._smoothing_factor)

    async def analyze_audio(self, buffer: bytes) -> VADState:
        """Analyze audio buffer and return current VAD state.

        Processes incoming audio and updates the voice state from confidence
        and volume. See the class docstring for how the thresholds apply in
        each state.

        Args:
            buffer: Audio buffer to analyze.

        Returns:
            Current VAD state after processing the buffer.
        """
        loop = asyncio.get_running_loop()
        state = await loop.run_in_executor(self._executor, self._run_analyzer, buffer)
        return state

    def _run_analyzer(self, buffer: bytes) -> VADState:
        """Analyze audio buffer and return current VAD state."""
        self._vad_buffer += buffer

        num_required_bytes = self._vad_frames_num_bytes
        if len(self._vad_buffer) < num_required_bytes:
            return self._vad_state

        while len(self._vad_buffer) >= num_required_bytes:
            audio_frames = self._vad_buffer[:num_required_bytes]
            self._vad_buffer = self._vad_buffer[num_required_bytes:]

            confidence = self.voice_confidence(audio_frames)

            # Volume is smoothed on every window, including windows that ignore
            # it, so the value is current when the next segment starts.
            volume = self._get_smoothed_volume(audio_frames)
            self._prev_volume = volume

            speaking = self._meets_speech_threshold(confidence, volume)

            if speaking:
                match self._vad_state:
                    case VADState.QUIET:
                        self._vad_state = VADState.STARTING
                        self._vad_starting_count = 1
                    case VADState.STARTING:
                        self._vad_starting_count += 1
                    case VADState.STOPPING:
                        self._vad_state = VADState.SPEAKING
                        self._vad_stopping_count = 0
            else:
                match self._vad_state:
                    case VADState.STARTING:
                        self._vad_state = VADState.QUIET
                        self._vad_starting_count = 0
                    case VADState.SPEAKING:
                        self._vad_state = VADState.STOPPING
                        self._vad_stopping_count = 1
                    case VADState.STOPPING:
                        self._vad_stopping_count += 1

        if (
            self._vad_state == VADState.STARTING
            and self._vad_starting_count >= self._vad_start_frames
        ):
            self._vad_state = VADState.SPEAKING
            self._vad_starting_count = 0

        if (
            self._vad_state == VADState.STOPPING
            and self._vad_stopping_count >= self._vad_stop_frames
        ):
            self._vad_state = VADState.QUIET
            self._vad_stopping_count = 0

        return self._vad_state

    def _meets_speech_threshold(self, confidence: float, volume: float) -> bool:
        """Return whether this window counts as speech in the current state.

        A segment that has not started needs confidence at or above
        ``confidence`` and volume at or above ``min_volume``. Once the state
        is speaking or stopping and ``hold_confidence`` is set, confidence at
        or above that lower value is enough. Without ``hold_confidence``,
        every state uses the start thresholds.

        Args:
            confidence: Voice confidence for the window.
            volume: Smoothed volume for the window.

        Returns:
            True when the window counts as speech.
        """
        hold = self._params.hold_confidence
        if hold is not None and self._vad_state in (VADState.SPEAKING, VADState.STOPPING):
            return confidence >= hold
        return confidence >= self._params.confidence and volume >= self._params.min_volume

    async def cleanup(self):
        """Release the analyzer's resources, including the thread it runs the model on.

        This method should be called when the object is no longer needed.
        """
        self._executor.shutdown(wait=False)
