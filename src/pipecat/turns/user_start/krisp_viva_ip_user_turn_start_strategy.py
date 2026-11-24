#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn start strategy using Krisp Interruption Prediction (IP).

This strategy uses Krisp's IP model to distinguish genuine user interruptions
from backchannels (e.g. "uh-huh", "yeah"). Instead of triggering a user turn
on every VAD speech event, it collects audio after VAD detects speech and runs
the IP model to predict whether the speech is a real interruption.

Only when the IP model's probability exceeds the configured threshold is
``report_user_turn_started()`` called. This prevents the bot from being
interrupted by brief acknowledgements or filler words.

With ``krisp_audio`` 1.14.0 or newer, whose IP ``process()`` accepts a
bot-speaking flag, the strategy is bot-aware: while the bot is not speaking
there is nothing to filter, so a user turn starts as soon as VAD detects
speech. Earlier builds take only a VAD flag, and the strategy classifies every
speech start with the model.
"""

import os

import numpy as np
from loguru import logger

from pipecat.audio.krisp_instance import (
    KrispVivaSDKManager,
    int_to_krisp_frame_duration,
    int_to_krisp_sample_rate,
    krisp_sdk_ip_accepts_bot_speaking,
)
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy

try:
    import krisp_audio
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use KrispVivaIPUserTurnStartStrategy, you need to install krisp_audio."
    )
    raise ImportError(f"Missing module: {e}") from e


class KrispVivaIPUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """User turn start strategy using Krisp VIVA Interruption Prediction.

    When VAD detects user speech, this strategy feeds audio frames into
    the Krisp VIVA IP model. The model outputs a probability indicating
    whether the speech is a genuine interruption (as opposed to a
    backchannel). A user turn is triggered only when this probability exceeds
    the configured threshold. With ``krisp_audio`` 1.14.0 or newer the strategy
    also tracks whether the bot is speaking, and while it is not, a user turn is
    triggered as soon as VAD detects speech.

    This strategy can work alongside other start strategies
    (e.g. ``TranscriptionUserTurnStartStrategy`` as a fallback) via the
    strategy list in ``UserTurnStrategies``.

    Example::

        from pipecat.turns.user_start import KrispVivaIPUserTurnStartStrategy

        strategies = UserTurnStrategies(
            start=[
                KrispVivaIPUserTurnStartStrategy(
                    model_path="/path/to/ip_model.kef",
                    threshold=0.5,
                ),
                TranscriptionUserTurnStartStrategy(),
            ],
        )
    """

    def __init__(
        self,
        *,
        model_path: str | None = None,
        threshold: float = 0.5,
        frame_duration_ms: int = 20,
        api_key: str = "",
        **kwargs,
    ):
        """Initialize the Krisp VIVA IP user turn start strategy.

        Args:
            model_path: Path to the Krisp VIVA IP model file (.kef). If None,
                uses the KRISP_VIVA_IP_MODEL_PATH environment variable.
            threshold: IP probability threshold (0.0 to 1.0). When the model's
                output exceeds this value, the speech is classified as a genuine
                interruption.
            frame_duration_ms: Frame duration in milliseconds for IP processing.
                Supported values: 10, 15, 20, 30, 32.
            api_key: Krisp SDK API key. If empty, falls back to the
                KRISP_VIVA_API_KEY environment variable.
            **kwargs: Additional arguments passed to BaseUserTurnStartStrategy.
        """
        super().__init__(**kwargs)

        self._threshold = threshold
        self._frame_duration_ms = frame_duration_ms
        self._api_key = api_key

        self._model_path = model_path or os.getenv("KRISP_VIVA_IP_MODEL_PATH")
        if not self._model_path:
            raise ValueError(
                "IP model path must be provided via model_path or "
                "KRISP_VIVA_IP_MODEL_PATH environment variable."
            )
        if not self._model_path.endswith(".kef"):
            raise ValueError("Model is expected with .kef extension")
        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(f"IP model file not found: {self._model_path}")

        self._sdk_acquired = False
        self._ip_session = None
        self._samples_per_frame: int | None = None
        self._sample_rate: int | None = None

        # State tracking
        self._speech_active = False
        # Tracked from bot speaking frames, so a strategy added mid-utterance
        # (e.g. via ``update_strategies``) treats the bot as silent until its
        # next ``BotStartedSpeakingFrame``.
        self._bot_speaking = False
        self._bot_aware = krisp_sdk_ip_accepts_bot_speaking()
        self._audio_buffer = bytearray()
        self._decision_made = False

        # Acquire SDK
        try:
            KrispVivaSDKManager.acquire(api_key=api_key)
            self._sdk_acquired = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Krisp SDK: {e}")

    async def cleanup(self):
        """Release Krisp SDK resources."""
        if self._sdk_acquired:
            try:
                self._ip_session = None
                KrispVivaSDKManager.release()
                self._sdk_acquired = False
            except Exception as e:
                logger.error(f"Error cleaning up Krisp VIVA IP strategy: {e}", exc_info=True)

    def _ensure_session(self, sample_rate: int):
        """Create or re-create the IP session when sample rate changes.

        Args:
            sample_rate: Audio sample rate in Hz.
        """
        if self._sample_rate == sample_rate and self._ip_session is not None:
            return

        self._sample_rate = sample_rate
        self._samples_per_frame = int((sample_rate * self._frame_duration_ms) / 1000)

        model_info = krisp_audio.ModelInfo()
        model_info.path = self._model_path

        ip_cfg = krisp_audio.IpSessionConfig()
        ip_cfg.inputSampleRate = int_to_krisp_sample_rate(sample_rate)
        ip_cfg.inputFrameDuration = int_to_krisp_frame_duration(self._frame_duration_ms)
        ip_cfg.modelInfo = model_info

        self._ip_session = krisp_audio.IpFloat.create(ip_cfg)
        logger.debug(f"Krisp VIVA IP session created (sample_rate={sample_rate})")

    def _reset_state(self):
        """Reset speech tracking state for the next candidate interruption."""
        self._speech_active = False
        self._audio_buffer.clear()
        self._decision_made = False

    async def handle_user_turn_started(self):
        """Ready the strategy for a new user turn."""
        self._reset_state()

    async def process_frame(self, frame: Frame):
        """Process a frame to detect genuine user interruptions.

        On ``VADUserStartedSpeakingFrame``, begins collecting audio.
        On ``InputAudioRawFrame``, feeds audio through the IP model and
        triggers a user turn if the interruption probability exceeds the
        threshold.
        On ``BotStartedSpeakingFrame``, marks the bot as speaking.
        On ``BotStoppedSpeakingFrame``, clears it; if the user is still speaking
        with no decision made yet, starts the user turn, otherwise resets the
        candidate state.
        On ``VADUserStoppedSpeakingFrame``, resets the candidate state.

        Args:
            frame: The incoming frame.
        """
        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_vad_started(frame)
        elif isinstance(frame, InputAudioRawFrame):
            await self._handle_audio(frame)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._handle_bot_stopped(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_reset(frame)

    async def _handle_vad_started(self, frame: VADUserStartedSpeakingFrame):
        """Begin collecting audio for interruption classification.

        When the bot is not speaking there is no interruption to classify, so
        the user turn starts immediately.

        Args:
            frame: The VAD speech-start frame.
        """
        self._speech_active = True
        if self._bot_aware and not self._bot_speaking:
            self._decision_made = True
            await self.report_user_turn_started()
            return

        logger.trace("Krisp VIVA IP: VAD speech started, collecting audio for classification")
        self._decision_made = False

    async def _handle_audio(self, frame: InputAudioRawFrame):
        """Feed audio to the IP model and check for genuine interruption.

        Every frame is passed to the IP model regardless of speech state so
        that the model maintains continuous internal state (matching the
        standalone Krisp SDK behaviour). ``_speech_active`` and
        ``_bot_speaking`` are forwarded as the per-frame VAD and bot-speaking
        inputs to the model; the model only predicts while the bot is speaking.
        ``_speech_active`` also gates the threshold evaluation.

        Args:
            frame: Raw audio input frame.
        """
        self._ensure_session(frame.sample_rate)

        if self._ip_session is None or self._samples_per_frame is None:
            logger.warning("IP session not ready, skipping frame")
            return

        self._audio_buffer.extend(frame.audio)

        total_samples = len(self._audio_buffer) // 2  # 2 bytes per int16 sample
        num_complete_frames = total_samples // self._samples_per_frame

        if num_complete_frames == 0:
            return

        complete_samples_count = num_complete_frames * self._samples_per_frame
        bytes_to_process = complete_samples_count * 2

        audio_to_process = bytes(self._audio_buffer[:bytes_to_process])
        self._audio_buffer = self._audio_buffer[bytes_to_process:]

        audio_int16 = np.frombuffer(audio_to_process, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        frames = audio_float32.reshape(-1, self._samples_per_frame)

        for ip_frame in frames:
            if self._bot_aware:
                ip_prob = self._ip_session.process(
                    ip_frame, self._speech_active, self._bot_speaking
                )
            else:
                ip_prob = self._ip_session.process(ip_frame, self._speech_active)

            if self._speech_active and not self._decision_made and ip_prob >= self._threshold:
                logger.debug(
                    f"Krisp VIVA IP: genuine interruption detected (prob={ip_prob:.3f}, "
                    f"threshold={self._threshold})"
                )
                self._decision_made = True
                await self.report_user_turn_started()
                return

    async def _handle_bot_stopped(self, frame: BotStoppedSpeakingFrame):
        """Handle the bot finishing its speech.

        VAD does not fire again for speech already in progress, so with a
        bot-aware SDK a user who is still speaking when the bot stops, with no
        decision made yet, starts a turn here: with the bot silent there is
        nothing left to classify. Otherwise the candidate state is reset.

        Args:
            frame: The bot-stopped frame.
        """
        self._bot_speaking = False
        if self._bot_aware and self._speech_active and not self._decision_made:
            self._decision_made = True
            await self.report_user_turn_started()
            return
        await self._handle_reset(frame)

    async def _handle_reset(self, frame: VADUserStoppedSpeakingFrame | BotStoppedSpeakingFrame):
        """Reset state when the candidate interruption window ends.

        Args:
            frame: The frame signaling end of speech or bot output.
        """
        if self._speech_active:
            logger.trace("Krisp VIVA IP: speech segment ended, resetting state")
            self._reset_state()
