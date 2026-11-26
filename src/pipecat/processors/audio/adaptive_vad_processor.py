#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Noise-adaptive VAD parameter processor.

This module provides a processor that tracks the background noise floor of an
audio stream and raises or lowers the VAD volume threshold to match it, by
emitting :class:`VADParamsUpdateFrame` upstream for a :class:`VADController` to
apply. It lets voice activity detection hold up across environments that are
quiet at one moment and noisy the next, without hand-tuning ``min_volume``.
"""

import time

from loguru import logger

from pipecat.audio.utils import exp_smoothing
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.volume import AudioVolumeTracker
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    SpeechControlParamsFrame,
    VADParamsUpdateFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AdaptiveVADProcessor(FrameProcessor):
    """Adapts the VAD volume threshold to the background noise floor.

    Measures the loudness of the audio stream while the user is not speaking,
    smooths it into a running noise-floor estimate, and sets the VAD
    ``min_volume`` a fixed margin above that floor. The new value is applied by
    emitting a :class:`VADParamsUpdateFrame` upstream, which a
    :class:`~pipecat.audio.vad.vad_controller.VADController` (for example the one
    inside a :class:`~pipecat.processors.audio.vad_processor.VADProcessor`) turns
    into a ``set_params`` call. Place this processor downstream of the VAD
    processor so it can observe the ``VADUserStartedSpeakingFrame`` and
    ``VADUserStoppedSpeakingFrame`` it broadcasts.

    Updates are only emitted while the user is not speaking, so applying them
    never resets an in-progress speech detection. Only ``min_volume`` is
    adjusted; ``confidence``, ``start_secs`` and ``stop_secs`` are left as
    configured. Adding this processor is opt-in: a pipeline without it keeps the
    static VAD threshold.

    Example::

        pipeline = Pipeline([
            transport.input(),
            VADProcessor(vad_analyzer=SileroVADAnalyzer()),
            AdaptiveVADProcessor(),
            ...
        ])
    """

    def __init__(
        self,
        *,
        margin: float = 0.1,
        smoothing_factor: float = 0.05,
        update_period_secs: float = 2.0,
        hysteresis: float = 0.05,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9,
        **kwargs,
    ):
        """Initialize the adaptive VAD processor.

        Args:
            margin: How far above the noise floor to set ``min_volume``, in
                normalized loudness units (0-1). Defaults to 0.1.
            smoothing_factor: Exponential smoothing factor for the noise-floor
                estimate, between 0 and 1. Lower values adapt more slowly and
                reject transients better. Defaults to 0.05.
            update_period_secs: Minimum interval between emitted parameter
                updates, in seconds. Defaults to 2.0.
            hysteresis: Minimum change in the target ``min_volume`` required
                before a new update is emitted, in normalized loudness units.
                Prevents a flutter of tiny updates. Defaults to 0.05.
            min_threshold: Lower clamp for the adapted ``min_volume``. Defaults
                to 0.1.
            max_threshold: Upper clamp for the adapted ``min_volume``. Defaults
                to 0.9.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._margin = margin
        self._smoothing_factor = smoothing_factor
        self._update_period_secs = update_period_secs
        self._hysteresis = hysteresis
        self._min_threshold = min_threshold
        self._max_threshold = max_threshold

        self._volume_tracker = AudioVolumeTracker()
        self._noise_floor: float | None = None
        self._speaking = False
        self._last_update_time = 0.0
        # Current VAD params, learned from SpeechControlParamsFrame so updates
        # build on whatever the analyzer (or another source) last applied.
        self._current_params: VADParams | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Observe audio and speech state, and emit adapted VAD params.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if isinstance(frame, SpeechControlParamsFrame):
            if frame.vad_params is not None:
                self._current_params = frame.vad_params
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._speaking = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speaking = False
        elif isinstance(frame, InputAudioRawFrame):
            self._handle_audio(frame)
            await self._maybe_update()

    def _handle_audio(self, frame: InputAudioRawFrame):
        """Fold a non-speech audio chunk into the noise-floor estimate."""
        self._volume_tracker.update(frame.audio, frame.sample_rate)
        if self._speaking:
            return
        volume = self._volume_tracker.volume
        if self._noise_floor is None:
            self._noise_floor = volume
        else:
            self._noise_floor = exp_smoothing(volume, self._noise_floor, self._smoothing_factor)

    async def _maybe_update(self):
        """Emit a VADParamsUpdateFrame if the target threshold has drifted."""
        if self._speaking or self._noise_floor is None or self._current_params is None:
            return

        now = time.monotonic()
        if now - self._last_update_time < self._update_period_secs:
            return

        target = min(
            self._max_threshold, max(self._min_threshold, self._noise_floor + self._margin)
        )
        if abs(target - self._current_params.min_volume) < self._hysteresis:
            return

        new_params = self._current_params.model_copy(update={"min_volume": target})
        self._current_params = new_params
        self._last_update_time = now
        logger.debug(
            f"{self}: adapting VAD min_volume to {target:.3f} (noise floor {self._noise_floor:.3f})"
        )
        await self.push_frame(VADParamsUpdateFrame(params=new_params), FrameDirection.UPSTREAM)
