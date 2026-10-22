#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base user turn filter for deciding whether a proposed turn end stands."""

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.utils.base_object import BaseObject


class BaseUserTurnFilter(BaseObject):
    """Base class for filters that decide whether a proposed turn end stands.

    A stop strategy proposes that the user is done. With a filter set, the
    :class:`~pipecat.turns.user_turn_controller.UserTurnController` ends the
    turn only once the filter accepts that proposal. A rejection keeps the turn
    open.

    Every proposal is judged against the inference generated for it, named by
    the id of the :class:`~pipecat.frames.frames.LLMContextFrame` that started
    it. Subclasses decide how: from the LLM's own marker, a classifier, or
    anything else, and answer by calling :meth:`accept` or :meth:`reject`. A
    filter that never answers leaves the turn to the controller's turn stop
    timeout, which ends it.

    Events triggered by filters:

      - `on_push_frame`: Indicates the filter wants to push a frame.
      - `on_broadcast_frame`: Indicates the filter wants to broadcast a frame.
      - `on_accept`: The proposed turn end stands.
      - `on_reject`: The user isn't done; the turn stays open.
    """

    def __init__(self, **kwargs):
        """Initialize the base user turn filter.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._register_event_handler("on_push_frame", sync=True)
        self._register_event_handler("on_broadcast_frame", sync=True)
        self._register_event_handler("on_accept", sync=True)
        self._register_event_handler("on_reject", sync=True)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the filter.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup.task_manager)

    async def cleanup(self):
        """Cleanup the filter."""
        await super().cleanup()

    async def handle_user_turn_started(self):
        """Notify the filter that a user turn has started."""
        pass

    async def handle_user_turn_stopped(self):
        """Notify the filter that the user turn has stopped."""
        pass

    async def handle_turn_end_proposed(self, context_frame_id: int):
        """Judge a proposed end of the user turn.

        The filter must eventually call :meth:`accept` or :meth:`reject`.

        Args:
            context_frame_id: The id of the ``LLMContextFrame`` that started the
                inference generated for this proposal.
        """
        pass

    async def process_frame(self, frame: Frame):
        """Process a frame seen by the user turn controller.

        Args:
            frame: The frame to be analyzed.
        """
        pass

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Emit on_push_frame to push a frame using the user aggregator.

        Args:
            frame: The frame to be pushed.
            direction: What direction the frame should be pushed to.
        """
        await self._call_event_handler("on_push_frame", frame, direction)

    async def broadcast_frame(self, frame_cls: type[Frame], **kwargs):
        """Emit on_broadcast_frame to broadcast a frame using the user aggregator.

        Args:
            frame_cls: The class of the frame to be broadcasted.
            **kwargs: Keyword arguments to be passed to the frame's constructor.
        """
        await self._call_event_handler("on_broadcast_frame", frame_cls, **kwargs)

    async def accept(self):
        """Let the proposed turn end stand."""
        await self._call_event_handler("on_accept")

    async def reject(self):
        """Keep the turn open: the user isn't done."""
        await self._call_event_handler("on_reject")
