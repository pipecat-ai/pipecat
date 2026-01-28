#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice Activity Detection controller for managing speech state transitions.

This module provides a controller that wraps a VADAnalyzer to track speech state
and emit events when speech starts, stops, or is actively detected.
"""

import time
from typing import Type

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADState
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    SpeechControlParamsFrame,
    StartFrame,
    VADParamsUpdateFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.base_object import BaseObject


class VADController(BaseObject):
    """Manages voice activity detection state and emits speech events.

    Wraps a `VADAnalyzer` to process audio and trigger events based on speech
    state transitions. Tracks whether the user is speaking, quiet, or
    transitioning between states.

    Event handlers available:

    - on_speech_started: Called when speech begins.
    - on_speech_stopped: Called when speech ends.
    - on_speech_activity: Called periodically while speech is detected.
    - on_push_frame: Called when the controller wants to push a frame.
    - on_broadcast_frame: Called when the controller wants to broadcast a frame.

    Example::

        @vad_controller.event_handler("on_speech_started")
        async def on_speech_started(controller):
            ...

        @vad_controller.event_handler("on_speech_stopped")
        async def on_speech_stopped(controller):
            ...

        @vad_controller.event_handler("on_speech_activity")
        async def on_speech_activity(controller):
            ...

        @vad_controller.event_handler("on_push_frame")
        async def on_push_frame(controller, frame: Frame, direction: FrameDirection):
            ...

        @vad_controller.event_handler("on_broadcast_frame")
        async def on_broadcast_frame(controller, frame_cls: Type[Frame], **kwargs):
            ...
    """

    def __init__(self, vad_analyzer: VADAnalyzer, *, speech_activity_period: float = 0.2):
        """Initialize the VAD controller.

        Args:
            vad_analyzer: The `VADAnalyzer` instance for processing audio.
            speech_activity_period: Minimum interval in seconds between
                `on_speech_activity` events. Defaults to 0.2.
        """
        super().__init__()
        self._vad_analyzer = vad_analyzer
        self._vad_state: VADState = VADState.QUIET

        # Last time a on_speech_activity was triggered.
        self._speech_activity_time = 0
        # How often a on_speech_activity event should be triggered (value should
        # be greater than the audio chunks to have any effect).
        self._speech_activity_period = speech_activity_period

        self._register_event_handler("on_speech_started", sync=True)
        self._register_event_handler("on_speech_stopped", sync=True)
        self._register_event_handler("on_speech_activity", sync=True)
        self._register_event_handler("on_push_frame", sync=True)
        self._register_event_handler("on_broadcast_frame", sync=True)

    async def process_frame(self, frame: Frame):
        """Process a frame and handle VAD-related events.

        Handles `StartFrame` to initialize the sample rate and `InputAudioRawFrame`
        to analyze audio for voice activity.

        Args:
            frame: The frame to process.
        """
        if isinstance(frame, StartFrame):
            await self._start(frame)
        elif isinstance(frame, InputAudioRawFrame):
            await self._handle_audio(frame)
        elif isinstance(frame, VADParamsUpdateFrame):
            self._vad_analyzer.set_params(frame.params)
            await self.broadcast_frame(SpeechControlParamsFrame, vad_params=frame.params)

    async def _start(self, frame: StartFrame):
        self._vad_analyzer.set_sample_rate(frame.audio_in_sample_rate)

    async def _handle_audio(self, frame: InputAudioRawFrame):
        """Process an audio chunk and emit speech events as needed.

        Analyzes the audio for voice activity and triggers `on_speech_started`,
        `on_speech_stopped`, or `on_speech_activity` events based on state changes.

        Args:
            frame: Audio frame to process.
        """
        self._vad_state = await self._handle_vad(frame.audio, self._vad_state)

        if self._vad_state == VADState.SPEAKING:
            await self._call_event_handler("on_speech_activity")

    async def _handle_vad(self, audio: bytes, vad_state: VADState) -> VADState:
        """Handle Voice Activity Detection results and trigger appropriate events."""
        new_vad_state = await self._vad_analyzer.analyze_audio(audio)
        if (
            new_vad_state != vad_state
            and new_vad_state != VADState.STARTING
            and new_vad_state != VADState.STOPPING
        ):
            if new_vad_state == VADState.SPEAKING:
                await self._call_event_handler("on_speech_started")
            elif new_vad_state == VADState.QUIET:
                await self._call_event_handler("on_speech_stopped")

            vad_state = new_vad_state
        return vad_state

    async def _maybe_speech_activity(self):
        """Handle user speaking frame."""
        diff_time = time.time() - self._speech_activity_time
        if diff_time >= self._speech_activity_period:
            self._speech_activity_time = time.time()
            await self._call_event_handler("on_speech_activity")

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Request a frame to be pushed through the pipeline.

        This emits an on_push_frame event that must be handled by a processor
        to actually push the frame into the pipeline.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await self._call_event_handler("on_push_frame", frame, direction)

    async def broadcast_frame(self, frame_cls: Type[Frame], **kwargs):
        """Request a frame to be broadcast upstream and downstream.

        This emits an on_broadcast_frame event that must be handled by a processor
        to actually broadcast the frame in the pipeline.

        Args:
            frame_cls: The class of the frame to broadcast.
            **kwargs: Arguments to pass to the frame constructor.
        """
        await self._call_event_handler("on_broadcast_frame", frame_cls, **kwargs)
