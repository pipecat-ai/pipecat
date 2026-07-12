#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared lifecycle behavior for telephony frame serializers."""

from abc import abstractmethod

from pipecat.frames.frames import CancelFrame, EndFrame, Frame
from pipecat.serializers.base_serializer import FrameSerializer


class AutoHangupFrameSerializer(FrameSerializer):
    """Base serializer for providers supporting automatic call termination."""

    def __init__(self, params: FrameSerializer.InputParams | None = None, **kwargs):
        """Initialize automatic call termination state.

        Args:
            params: Configuration parameters.
            **kwargs: Additional arguments passed to :class:`FrameSerializer`.
        """
        super().__init__(params=params, **kwargs)
        self._hangup_attempted = False

    async def _maybe_hang_up(self, frame: Frame, *, enabled: bool) -> bool:
        """Attempt to terminate the call for an enabled terminal frame.

        Args:
            frame: Frame being serialized.
            enabled: Whether automatic call termination is enabled.

        Returns:
            True if the terminal frame was consumed.
        """
        if not enabled or not isinstance(frame, (EndFrame, CancelFrame)):
            return False

        if not self._hangup_attempted:
            self._hangup_attempted = True
            await self._hang_up_call()

        return True

    @abstractmethod
    async def _hang_up_call(self) -> None:
        """Terminate the active provider call."""
        pass
