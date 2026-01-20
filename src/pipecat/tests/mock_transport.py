#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Mock transport implementation for testing Pipecat pipelines.

This module provides a simple mock transport that can be used in tests
to verify pipeline behavior without needing a real transport connection.
"""

from typing import Optional

from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import BaseTransport, TransportParams


class MockInputTransport(FrameProcessor):
    """Mock input transport processor for testing.

    Passes through all frames without modification. Can be extended
    to simulate specific input behaviors.
    """

    def __init__(self, params: Optional[TransportParams] = None, **kwargs):
        """Initialize the mock input transport.

        Args:
            params: Optional transport parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._params = params or TransportParams()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames by passing them through.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class MockOutputTransport(FrameProcessor):
    """Mock output transport processor for testing.

    Simulates bot speaking behavior by emitting BotStartedSpeaking,
    BotSpeaking, and BotStoppedSpeaking frames when TTS frames are received.
    """

    def __init__(
        self,
        params: Optional[TransportParams] = None,
        *,
        emit_bot_speaking: bool = True,
        **kwargs,
    ):
        """Initialize the mock output transport.

        Args:
            params: Optional transport parameters.
            emit_bot_speaking: If True, emits BotSpeakingFrame on TTSAudioRawFrame.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._params = params or TransportParams()
        self._emit_bot_speaking = emit_bot_speaking

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and simulate bot speaking behavior.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            await self.push_frame(BotStartedSpeakingFrame())
            await self.push_frame(
                BotStartedSpeakingFrame(), direction=FrameDirection.UPSTREAM
            )
        elif isinstance(frame, TTSAudioRawFrame):
            if self._emit_bot_speaking:
                await self.push_frame(BotSpeakingFrame())
                await self.push_frame(
                    BotSpeakingFrame(), direction=FrameDirection.UPSTREAM
                )
        elif isinstance(frame, TTSStoppedFrame):
            await self.push_frame(BotStoppedSpeakingFrame())
            await self.push_frame(
                BotStoppedSpeakingFrame(), direction=FrameDirection.UPSTREAM
            )

        await self.push_frame(frame, direction)


class MockTransport(BaseTransport):
    """Mock transport for testing Pipecat pipelines.

    Provides simple input and output transport processors that can be
    used in tests without needing actual WebSocket or WebRTC connections.
    """

    def __init__(
        self,
        params: Optional[TransportParams] = None,
        *,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        emit_bot_speaking: bool = True,
    ):
        """Initialize the mock transport.

        Args:
            params: Optional transport parameters.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
            emit_bot_speaking: If True, output transport emits BotSpeakingFrame.
        """
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params or TransportParams()
        self._input = MockInputTransport(self._params, name=self._input_name)
        self._output = MockOutputTransport(
            self._params,
            emit_bot_speaking=emit_bot_speaking,
            name=self._output_name,
        )

    def input(self) -> FrameProcessor:
        """Get the mock input transport processor.

        Returns:
            The mock input transport instance.
        """
        return self._input

    def output(self) -> FrameProcessor:
        """Get the mock output transport processor.

        Returns:
            The mock output transport instance.
        """
        return self._output
