#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Answering the agent's permission requests."""

from loguru import logger

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.acp.frames import ACPClientResponseFrame, ACPPermissionRequestFrame
from pipecat.services.acp.types import (
    ACPErrorData,
    PermissionOptionKind,
    RequestPermissionResult,
)

DEFAULT_PREFERENCE = [
    PermissionOptionKind.ALLOW_ALWAYS,
    PermissionOptionKind.ALLOW_ONCE,
]
"""Option kinds :class:`ACPAutoPermission` will pick, most preferred first."""


class ACPAutoPermission(FrameProcessor):
    """Answers permission requests without asking anyone.

    Useful for development against a scratch repository, and as the reference
    for what a real answerer does: pick an option id, push an
    :class:`~pipecat.services.acp.frames.ACPClientResponseFrame` carrying the
    request id, and let the request frame continue on its way.

    The response is broadcast, so this processor works on either side of the
    service.
    """

    def __init__(self, *, preference: list[PermissionOptionKind] | None = None, **kwargs):
        """Initialize the processor.

        Args:
            preference: Option kinds to pick, most preferred first. A request
                offering none of them is answered with an error.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(**kwargs)
        self._preference = preference or DEFAULT_PREFERENCE

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if isinstance(frame, ACPPermissionRequestFrame) and frame.params:
            await self._answer(frame)

    async def _answer(self, frame: ACPPermissionRequestFrame):
        options = {o.kind: o for o in frame.params.options}
        chosen = next((options[k] for k in self._preference if k in options), None)

        if not chosen:
            offered = [o.kind.value for o in frame.params.options]
            logger.warning(f"{self}: no acceptable permission option in {offered}")
            response = ACPClientResponseFrame(
                request_id=frame.request_id,
                error=ACPErrorData(code=-32000, message="No acceptable permission option"),
            )
        else:
            logger.debug(f"{self}: auto-allowing {frame.params.tool_call.title} ({chosen.kind})")
            response = ACPClientResponseFrame(
                request_id=frame.request_id,
                result=RequestPermissionResult.selected(chosen.option_id).to_wire(),
            )

        await self.broadcast_frame_instance(response)
