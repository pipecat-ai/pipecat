#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The LLM extension that keeps a ``UIWorker``'s LLM abreast of the screen."""

from collections.abc import Callable

from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.base_llm_extension import BaseLLMExtension


class UIStateLLMExtension(BaseLLMExtension):
    """Keeps a ``UIWorker``'s LLM abreast of the screen.

    The worker's half of reading the UI: the wire-format guide that lets the
    model parse the ``<ui_state>`` and ``<ui_event>`` messages goes into the
    system instruction, and the latest ``<ui_state>`` snapshot is appended to
    the context as a developer message before each inference that answers a
    user turn, so the model answers against the screen as it is now. The
    worker installs the extension on its LLM and renders the snapshot.
    """

    installed_by = "UIWorker"

    def __init__(self, *, prompt_guide: str | None, render: Callable[[], str] | None, **kwargs):
        """Initialize the extension.

        Args:
            prompt_guide: What the extension adds to the LLM's system
                instruction, or ``None`` for nothing.
            render: Renders the latest snapshot as the content of the
                developer message appended before an inference, or an empty
                string for no snapshot yet; ``None`` appends nothing, for a
                worker that injects the snapshot itself.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._prompt_guide = prompt_guide
        self._render = render

    @property
    def system_instruction(self) -> str | None:
        """The wire-format guide."""
        return self._prompt_guide

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Append the latest snapshot to the context of an inference that answers a user turn.

        Only a user turn's inference gets it, not the follow-up inference the
        LLM runs after a tool result, which would stack a second snapshot
        within the same turn.

        Args:
            frame: The frame the service processes.
            direction: The direction of frame processing.
        """
        if self._render is None or not isinstance(frame, LLMContextFrame):
            return
        if not _is_user_turn(frame.context):
            return
        content = self._render()
        if content:
            frame.context.add_message({"role": "developer", "content": content})


def _is_user_turn(context: LLMContext) -> bool:
    """Whether the context's last message is the user's turn.

    Distinguishes a fresh user-turn inference (tail is the user message) from
    the follow-up inference the LLM runs after a tool result (tail is the tool
    result / assistant output).
    """
    messages = context.messages
    if not messages:
        return False
    last = messages[-1]
    return isinstance(last, dict) and last.get("role") == "user"
