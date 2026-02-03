#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base classes for Speech-to-Text services with continuous and segmented processing."""

import asyncio
import io
import time
import wave
from abc import abstractmethod
from typing import Any, AsyncGenerator, Dict, Mapping, Optional

from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    MetricsFrame,
    SpeechControlParamsFrame,
    StartFrame,
    STTMuteFrame,
    STTUpdateSettingsFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService
from pipecat.services.websocket_service import WebsocketService
from pipecat.transcriptions.language import Language


class STTService(AIService):
    """Base class for speech-to-text services.

    Provides common functionality for STT services including audio passthrough,
    muting, settings management, and audio processing. Subclasses must implement
    the run_stt method to provide actual speech recognition.

    Event handlers:
        on_connected: Called when connected to the STT service.
        on_disconnected: Called when disconnected from the STT service.
        on_connection_error: Called when a connection to the STT service error occurs.

    Example::

        @stt.event_handler("on_connected")
        async def on_connected(stt: STTService):
            logger.debug(f"STT connected")

        @stt.event_handler("on_disconnected")
        async def on_disconnected(stt: STTService):
            logger.debug(f"STT disconnected")

        @stt.event_handler("on_connection_error")
        async def on_connection_error(stt: STTService, error: str):
            logger.error(f"STT connection error: {error}")
    """

    def __init__(
        self,
        audio_passthrough=True,
        # STT input sample rate
        sample_rate: Optional[int] = None,
        # STT TTFB timeout - time to wait after VAD stop before reporting TTFB
        stt_ttfb_timeout: float = 2.0,
        **kwargs,
    ):
        """Initialize the STT service.

        Args:
            audio_passthrough: Whether to pass audio frames downstream after processing.
                Defaults to True.
            sample_rate: The sample rate for audio input. If None, will be determined
                from the start frame.
            stt_ttfb_timeout: Time in seconds to wait after VAD stop before reporting
                TTFB. This delay allows the final transcription to arrive. Defaults to 2.0.
                Note: STT "TTFB" differs from traditional TTFB (which measures from a discrete
                request to first response byte). Since STT receives continuous audio, we measure
                from when the user stops speaking to when the final transcript arrives—capturing
                the latency that matters for voice AI applications.
            **kwargs: Additional arguments passed to the parent AIService.
        """
        super().__init__(**kwargs)
        self._audio_passthrough = audio_passthrough
        self._init_sample_rate = sample_rate
        self._sample_rate = 0
        self._settings: Dict[str, Any] = {}
        self._tracing_enabled: bool = False
        self._muted: bool = False
        self._user_id: str = ""

        # STT TTFB tracking state
        self._stt_ttfb_timeout = stt_ttfb_timeout
        self._ttfb_timeout_task: Optional[asyncio.Task] = None
        self._vad_stop_secs: Optional[float] = None
        self._speech_end_time: Optional[float] = None
        self._user_speaking: bool = False
        self._last_transcription_time: Optional[float] = None
        self._finalize_pending: bool = False
        self._finalize_requested: bool = False

        self._register_event_handler("on_connected")
        self._register_event_handler("on_disconnected")
        self._register_event_handler("on_connection_error")

    @property
    def is_muted(self) -> bool:
        """Check if the STT service is currently muted.

        Returns:
            True if the service is muted and will not process audio.
        """
        return self._muted

    def request_finalize(self):
        """Mark that a finalize request has been sent, awaiting server confirmation.

        For providers that have explicit server confirmation of finalization
        (e.g., Deepgram's from_finalize field), call this when sending the finalize
        request. Then call confirm_finalize() when the server confirms.

        For providers without server confirmation, don't call this method - just
        send the finalize/flush/commit command and rely on the TTFB timeout.
        """
        self._finalize_requested = True

    def confirm_finalize(self):
        """Confirm that the server has acknowledged the finalize request.

        Call this when the server response confirms finalization (e.g., Deepgram's
        from_finalize=True). The next TranscriptionFrame pushed will be marked
        as finalized.

        Only has effect if request_finalize() was previously called.
        """
        if self._finalize_requested:
            self._finalize_pending = True
            self._finalize_requested = False

    @property
    def sample_rate(self) -> int:
        """Get the current sample rate for audio processing.

        Returns:
            The sample rate in Hz.
        """
        return self._sample_rate

    async def set_model(self, model: str):
        """Set the speech recognition model.

        Args:
            model: The name of the model to use for speech recognition.
        """
        self.set_model_name(model)

    async def set_language(self, language: Language):
        """Set the language for speech recognition.

        Args:
            language: The language to use for speech recognition.
        """
        pass

    @abstractmethod
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Run speech-to-text on the provided audio data.

        This method must be implemented by subclasses to provide actual speech
        recognition functionality.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: Frames containing transcription results (typically TextFrame).
        """
        pass

    async def start(self, frame: StartFrame):
        """Start the STT service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._sample_rate = self._init_sample_rate or frame.audio_in_sample_rate
        self._tracing_enabled = frame.enable_tracing

    async def cleanup(self):
        """Clean up STT service resources."""
        await super().cleanup()
        await self._cancel_ttfb_timeout()

    async def _update_settings(self, settings: Mapping[str, Any]):
        logger.info(f"Updating STT settings: {self._settings}")
        for key, value in settings.items():
            if key in self._settings:
                logger.info(f"Updating STT setting {key} to: [{value}]")
                self._settings[key] = value
                if key == "language":
                    await self.set_language(value)
            elif key == "language":
                await self.set_language(value)
            elif key == "model":
                self.set_model_name(value)
            else:
                logger.warning(f"Unknown setting for STT service: {key}")

    async def process_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection):
        """Process an audio frame for speech recognition.

        If the service is muted, this method does nothing. Otherwise, it
        processes the audio frame and runs speech-to-text on it, yielding
        transcription results. If the frame has a user_id, it is stored
        for later use in transcription.

        Args:
            frame: The audio frame to process.
            direction: The direction of frame processing.
        """
        if self._muted:
            return

        # UserAudioRawFrame contains a user_id (e.g. Daily, Livekit)
        if hasattr(frame, "user_id"):
            self._user_id = frame.user_id
        # AudioRawFrame does not have a user_id (e.g. SmallWebRTCTransport, websockets)
        else:
            self._user_id = ""

        if not frame.audio:
            # Ignoring in case we don't have audio to transcribe.
            logger.warning(
                f"Empty audio frame received for STT service: {self.name} {frame.num_frames}"
            )
            return

        await self.process_generator(self.run_stt(frame.audio))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, handling VAD events and audio segmentation.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            # In this service we accumulate audio internally and at the end we
            # push a TextFrame. We also push audio downstream in case someone
            # else needs it.
            await self.process_audio_frame(frame, direction)
            if self._audio_passthrough:
                await self.push_frame(frame, direction)
        elif isinstance(frame, SpeechControlParamsFrame):
            await self._handle_speech_control_params(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_vad_user_started_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_vad_user_stopped_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, STTUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        elif isinstance(frame, STTMuteFrame):
            self._muted = frame.mute
            logger.debug(f"STT service {'muted' if frame.mute else 'unmuted'}")
        elif isinstance(frame, InterruptionFrame):
            await self._reset_stt_ttfb_state()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame downstream, tracking TranscriptionFrame timestamps for TTFB.

        Stores the timestamp of each TranscriptionFrame for TTFB calculation.
        If the frame is marked as finalized (via request_finalize/confirm_finalize),
        reports TTFB immediately and cancels any pending timeout. Otherwise, TTFB is
        reported after a timeout.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        if isinstance(frame, TranscriptionFrame):
            # Store the transcription time for TTFB calculation
            self._last_transcription_time = time.time()

            # Set finalized from pending state and auto-reset
            if self._finalize_pending:
                frame.finalized = True
                self._finalize_pending = False

            # If this is a finalized transcription, report TTFB immediately
            if frame.finalized and self._speech_end_time is not None:
                ttfb = self._last_transcription_time - self._speech_end_time
                await self._emit_stt_ttfb_metric(ttfb)
                # Cancel the timeout since we've already reported
                await self._cancel_ttfb_timeout()
                # Clear state
                self._speech_end_time = None
                self._last_transcription_time = None

        await super().push_frame(frame, direction)

    async def _handle_speech_control_params(self, frame: SpeechControlParamsFrame):
        """Handle speech control parameters frame to extract VAD stop_secs.

        Args:
            frame: The speech control parameters frame.
        """
        if frame.vad_params is not None:
            self._vad_stop_secs = frame.vad_params.stop_secs

    async def _cancel_ttfb_timeout(self):
        """Cancel any pending TTFB timeout task."""
        if self._ttfb_timeout_task:
            await self.cancel_task(self._ttfb_timeout_task)
            self._ttfb_timeout_task = None

    async def _reset_stt_ttfb_state(self):
        """Reset STT TTFB measurement state.

        Called when starting a new utterance or on interruption to ensure
        we don't use stale state for TTFB calculations. This specifically guards
        against the case where a TranscriptionFrame is received without corresponding
        VADUserStartedSpeakingFrame and VADUserStoppedSpeakingFrame frames.

        Note: Does not reset _user_speaking since InterruptionFrame can arrive
        while user is still speaking.
        """
        await self._cancel_ttfb_timeout()
        self._speech_end_time = None
        self._last_transcription_time = None

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        """Handle VAD user started speaking frame to start tracking transcriptions.

        Cancels any pending TTFB timeout, resets TTFB tracking state, and marks user as speaking.
        Also resets finalization state to prevent stale finalization from a previous utterance.

        Args:
            frame: The VAD user started speaking frame.
        """
        await self._reset_stt_ttfb_state()
        self._user_speaking = True
        self._finalize_requested = False
        self._finalize_pending = False

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        """Handle VAD user stopped speaking frame.

        Calculates the actual speech end time and starts a timeout task to wait
        for the final transcription before reporting TTFB.

        Args:
            frame: The VAD user stopped speaking frame.
        """
        self._user_speaking = False

        # Skip TTFB measurement if we don't have VAD params
        if self._vad_stop_secs is None:
            return

        # Calculate the actual speech end time (current time minus VAD stop delay).
        # This approximates when the last user audio was sent to the STT service,
        # which we use to measure against the eventual transcription response.
        self._speech_end_time = time.time() - self._vad_stop_secs

        # Start timeout task (any previous timeout was cancelled by VADUserStartedSpeakingFrame
        # or InterruptionFrame)
        self._ttfb_timeout_task = self.create_task(
            self._ttfb_timeout_handler(), name="stt_ttfb_timeout"
        )

    async def _ttfb_timeout_handler(self):
        """Wait for timeout then report TTFB using the last transcription timestamp.

        This timeout allows the final transcription to arrive before we calculate
        and report TTFB. If no transcription arrived, no TTFB is reported.
        """
        try:
            await asyncio.sleep(self._stt_ttfb_timeout)

            # Report TTFB if we have both speech end time and transcription time
            if self._speech_end_time is not None and self._last_transcription_time is not None:
                ttfb = self._last_transcription_time - self._speech_end_time
                await self._emit_stt_ttfb_metric(ttfb)

            # Clear state after reporting
            self._speech_end_time = None
            self._last_transcription_time = None
        except asyncio.CancelledError:
            # Task was cancelled (new utterance or interruption), which is expected behavior
            pass
        finally:
            self._ttfb_timeout_task = None

    async def _emit_stt_ttfb_metric(self, ttfb: float):
        """Emit STT TTFB metric if value is non-negative.

        Args:
            ttfb: The TTFB value in seconds.
        """
        if ttfb >= 0:
            logger.debug(f"{self} TTFB: {ttfb:.3f}s")
            if self.metrics_enabled:
                ttfb_data = TTFBMetricsData(
                    processor=self.name,
                    model=self.model_name,
                    value=ttfb,
                )
                await super().push_frame(MetricsFrame(data=[ttfb_data]))


class SegmentedSTTService(STTService):
    """STT service that processes speech in segments using VAD events.

    Uses Voice Activity Detection (VAD) events to detect speech segments and runs
    speech-to-text only on those segments, rather than continuously.

    Requires VAD to be enabled in the pipeline to function properly. Maintains a
    small audio buffer to account for the delay between actual speech start and
    VAD detection.
    """

    def __init__(self, *, sample_rate: Optional[int] = None, **kwargs):
        """Initialize the segmented STT service.

        Args:
            sample_rate: The sample rate for audio input. If None, will be determined
                from the start frame.
            **kwargs: Additional arguments passed to the parent STTService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._content = None
        self._wave = None
        self._audio_buffer = bytearray()
        self._audio_buffer_size_1s = 0
        self._user_speaking = False

    async def start(self, frame: StartFrame):
        """Start the segmented STT service and initialize audio buffer.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._audio_buffer_size_1s = self.sample_rate * 2

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame, marking TranscriptionFrames as finalized.

        Segmented STT services process complete speech segments and return a single
        TranscriptionFrame per segment, so every transcription is inherently finalized.

        Args:
            frame: The frame to push.
            direction: The direction of frame flow in the pipeline.
        """
        if isinstance(frame, TranscriptionFrame):
            frame.finalized = True
        await super().push_frame(frame, direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, handling VAD events and audio segmentation."""
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_user_started_speaking(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(frame)

    async def _handle_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        self._user_speaking = True

    async def _handle_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        self._user_speaking = False

        content = io.BytesIO()
        wav = wave.open(content, "wb")
        wav.setsampwidth(2)
        wav.setnchannels(1)
        wav.setframerate(self.sample_rate)
        wav.writeframes(self._audio_buffer)
        wav.close()
        content.seek(0)

        # Start clean.
        self._audio_buffer.clear()

        await self.process_generator(self.run_stt(content.read()))

    async def process_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection):
        """Process audio frames by buffering them for segmented transcription.

        Continuously buffers audio, growing the buffer while user is speaking and
        maintaining a small buffer when not speaking to account for VAD delay.

        If the frame has a user_id, it is stored for later use in transcription.

        Args:
            frame: The audio frame to process.
            direction: The direction of frame processing.
        """
        # UserAudioRawFrame contains a user_id (e.g. Daily, Livekit)
        if hasattr(frame, "user_id"):
            self._user_id = frame.user_id
        # AudioRawFrame does not have a user_id (e.g. SmallWebRTCTransport, websockets)
        else:
            self._user_id = ""

        # If the user is speaking the audio buffer will keep growing.
        self._audio_buffer += frame.audio

        # If the user is not speaking we keep just a little bit of audio.
        if not self._user_speaking and len(self._audio_buffer) > self._audio_buffer_size_1s:
            discarded = len(self._audio_buffer) - self._audio_buffer_size_1s
            self._audio_buffer = self._audio_buffer[discarded:]


class WebsocketSTTService(STTService, WebsocketService):
    """Base class for websocket-based STT services.

    Combines STT functionality with websocket connectivity, providing automatic
    error handling and reconnection capabilities.
    """

    def __init__(self, *, reconnect_on_error: bool = True, **kwargs):
        """Initialize the Websocket STT service.

        Args:
            reconnect_on_error: Whether to automatically reconnect on websocket errors.
            **kwargs: Additional arguments passed to parent classes.
        """
        STTService.__init__(self, **kwargs)
        WebsocketService.__init__(self, reconnect_on_error=reconnect_on_error, **kwargs)

    async def _report_error(self, error: ErrorFrame):
        await self._call_event_handler("on_connection_error", error.error)
        await self.push_error_frame(error)
