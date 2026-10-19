#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tells the LLM when the user's audio quality degrades or recovers.

Classes:
    AudioQualityNudger: Adds a developer message to the LLM context when the
        smoothed audio quality score crosses a threshold.
"""

from loguru import logger

from pipecat.frames.frames import AudioQualityFrame, Frame, LLMMessagesAppendFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

DEFAULT_DEGRADATION_DESCRIPTIONS = {
    "noise": "loud background noise",
    "speaker_reverb": "the user is far from the microphone or in an echoing room",
    "interfering_speech": "other people talk near the user",
    "packet_loss": "a poor connection that drops parts of the audio",
    "codec_degradation": "a poor connection that distorts the audio",
}

DEFAULT_DEGRADED_MESSAGE = (
    "The user's audio quality is poor, most likely because of {description}. "
    "The transcription of what the user says can contain errors. "
    "Confirm requests that seem unclear or unlikely before you act on them, and keep your answers "
    "short. Tell the user about the audio problem one time and suggest how to fix it."
)

DEFAULT_RECOVERED_MESSAGE = (
    "The user's audio quality is good again. Do not mention the audio problem anymore."
)

_UNKNOWN_DESCRIPTION = "an unknown problem"


class AudioQualityNudger(FrameProcessor):
    """Tells the LLM when the user's audio quality degrades or recovers.

    The processor reads each :class:`~pipecat.frames.frames.AudioQualityFrame`
    and keeps an exponential moving average of each score, so one bad window
    does not trigger it. When the smoothed ``risk_score`` reaches
    ``degraded_threshold``, it adds a developer message to the LLM context
    that names the largest degradation. When the smoothed score then falls
    below ``recovered_threshold``, it adds a message that the audio is good
    again. The gap between the two thresholds stops repeated messages when
    the score stays near one threshold.

    The messages do not run the LLM. The agent uses them at its next response.

    Place the processor after the audio quality analyzer and before the user
    context aggregator.

    Example::

        nudger = AudioQualityNudger(
            degraded_message="The user's audio is poor because of {description}. "
            "Ask the user to repeat unclear requests.",
            recovered_message="The user's audio is good again.",
            degradation_descriptions={"noise": "a noisy room"},
        )

        pipeline = Pipeline([transport.input(), analyzer, nudger, stt, user_aggregator, ...])
    """

    def __init__(
        self,
        *,
        degraded_threshold: float = 0.5,
        recovered_threshold: float = 0.3,
        smoothing: float = 0.3,
        degraded_message: str = DEFAULT_DEGRADED_MESSAGE,
        recovered_message: str = DEFAULT_RECOVERED_MESSAGE,
        degradation_descriptions: dict[str, str] | None = None,
        **kwargs,
    ):
        """Initialize the audio quality nudger.

        Args:
            degraded_threshold: Smoothed ``risk_score`` at or above which the
                audio counts as degraded. Calibrate it against the failures
                that you see in your application.
            recovered_threshold: Smoothed ``risk_score`` below which degraded
                audio counts as good again.
            smoothing: Weight of the newest window in the moving average.
                Higher values react faster to changes.
            degraded_message: Message for degraded audio. A ``{description}``
                placeholder gets the description of the largest degradation.
            recovered_message: Message for audio that is good again.
            degradation_descriptions: Descriptions for degradation names. They
                replace the entries with the same name in
                ``DEFAULT_DEGRADATION_DESCRIPTIONS``. A name with no
                description goes into the message as it is.
            **kwargs: Additional arguments passed to parent class.

        Raises:
            ValueError: If a threshold is not between 0.0 and 1.0, if
                ``recovered_threshold`` is not less than ``degraded_threshold``,
                or if ``smoothing`` is not in the range (0.0, 1.0].
        """
        if not 0.0 <= recovered_threshold < degraded_threshold <= 1.0:
            raise ValueError(
                "Thresholds must be between 0.0 and 1.0, and 'recovered_threshold' must be "
                "less than 'degraded_threshold'."
            )

        if not 0.0 < smoothing <= 1.0:
            raise ValueError("'smoothing' must be greater than 0.0 and at most 1.0.")

        super().__init__(**kwargs)

        self._degraded_threshold = degraded_threshold
        self._recovered_threshold = recovered_threshold
        self._smoothing = smoothing
        self._degraded_message = degraded_message
        self._recovered_message = recovered_message
        self._descriptions = {
            **DEFAULT_DEGRADATION_DESCRIPTIONS,
            **(degradation_descriptions or {}),
        }

        self._risk: float | None = None
        self._degradations: dict[str, float] = {}
        self._degraded = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame and forward it.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        await self.push_frame(frame, direction)

        if isinstance(frame, AudioQualityFrame):
            await self._handle_scores(frame)

    async def _handle_scores(self, frame: AudioQualityFrame):
        self._risk = self._smooth(self._risk, frame.risk_score)
        for name, value in frame.degradations.items():
            self._degradations[name] = self._smooth(self._degradations.get(name), value)

        if not self._degraded and self._risk >= self._degraded_threshold:
            self._degraded = True
            if self._degradations:
                largest = max(self._degradations, key=lambda n: self._degradations[n])
                description = self._descriptions.get(largest, largest)
            else:
                largest = None
                description = _UNKNOWN_DESCRIPTION
            logger.debug(
                f"{self}: audio degraded at {frame.audio_secs:.0f}s "
                f"(risk {self._risk:.2f}, largest degradation: {largest})"
            )
            await self._push_message(self._degraded_message.format(description=description))
        elif self._degraded and self._risk < self._recovered_threshold:
            self._degraded = False
            logger.debug(
                f"{self}: audio recovered at {frame.audio_secs:.0f}s (risk {self._risk:.2f})"
            )
            await self._push_message(self._recovered_message)

    def _smooth(self, average: float | None, value: float) -> float:
        if average is None:
            return value
        return self._smoothing * value + (1.0 - self._smoothing) * average

    async def _push_message(self, content: str):
        frame = LLMMessagesAppendFrame([{"role": "developer", "content": content}], run_llm=False)
        # An interruption must not drop a state change, or the context keeps a
        # stale audio quality message.
        frame.interruptible = False
        await self.push_frame(frame)
