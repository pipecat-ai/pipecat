#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Observer that records a bot's conversation as a conversation file holds it.

It lets a call the harness didn't drive, such as a production one, be judged
with ``pipecat eval judge`` against a simulation's ``success`` and metrics.
"""

from dataclasses import dataclass

from pipecat.evals.conversation import EvalConversation
from pipecat.frames.frames import (
    ExternalFunctionCallCancelFrame,
    ExternalFunctionCallInProgressFrame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    LLMContextAssistantTurnFrame,
    LLMContextUserTurnFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.frames import RTVISendTextFrame


@dataclass
class EvalConversationEndedMessage:
    """A message accompanying ``on_conversation_ended``.

    Parameters:
        conversation: The conversation's messages, as a conversation file holds
            them (see :mod:`pipecat.evals.conversation`).
    """

    conversation: list[dict]


class EvalConversationObserver(BaseObserver):
    """Records a bot's conversation, and reports it once the pipeline is done.

    Each side's turn is what the context aggregators write to the LLM context,
    so the user's turns are what the bot heard and the bot's turns are what it
    said, cut where the user interrupted it. Text a client sends with RTVI's
    ``send-text`` is a user turn too, since it goes to the context as is. Each
    turn is recorded as it is written, so summarizing the context later doesn't
    lose it, and the context's other messages, such as Flows' tool changes, stay
    out.

    It records what passes through the worker it is attached to: in a bot with
    several workers, that is the one holding the conversation's context, and
    calls made in other workers are not in it.

    Events:
        on_conversation_ended(observer, message): Emitted once the pipeline
            is done, with an :class:`EvalConversationEndedMessage` holding the
            conversation.

    Example::

        observer = EvalConversationObserver()

        @observer.event_handler("on_conversation_ended")
        async def on_conversation_ended(observer, message: EvalConversationEndedMessage):
            Path("call.conversation.json").write_text(json.dumps(message.conversation, indent=2))

        worker = PipelineWorker(pipeline, observers=[observer])
    """

    def __init__(self, **kwargs):
        """Initialize the observer.

        Args:
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(observe_every_push=False, **kwargs)
        self._conversation = EvalConversation()

        self._register_event_handler("on_conversation_ended")

    async def on_push_frame(self, data: FramePushed):
        """Add what a frame says about the conversation to it.

        Args:
            data: The frame push event data.
        """
        frame = data.frame

        # These frames are broadcast, arriving as two frames, each pushed for
        # the first time once. Read the downstream one.
        if frame.broadcast_sibling_id is not None and data.direction != FrameDirection.DOWNSTREAM:
            return

        if isinstance(frame, (LLMContextUserTurnFrame, RTVISendTextFrame)):
            self._conversation.add_user_message(frame.text)
        elif isinstance(frame, LLMContextAssistantTurnFrame):
            self._conversation.add_assistant_message(frame.text)
        elif isinstance(frame, (FunctionCallInProgressFrame, ExternalFunctionCallInProgressFrame)):
            self._conversation.add_tool_call(frame.function_name, frame.arguments or {})
        elif isinstance(frame, (FunctionCallCancelFrame, ExternalFunctionCallCancelFrame)):
            self._conversation.cancel_tool_call(frame.function_name)

    async def cleanup(self):
        """Report the conversation: the pipeline is done, and so is the bot's last turn."""
        self._conversation.end_bot_turn()
        message = EvalConversationEndedMessage(conversation=self._conversation.messages)
        await self._call_event_handler("on_conversation_ended", message)
        await super().cleanup()
