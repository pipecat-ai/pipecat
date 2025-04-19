import sys
from enum import Enum
from typing import Any, Dict, List

from openai._types import NotGiven
from pydantic import BaseModel

from pipecat.frames.frames import (
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
)
from pipecat.processors.aggregators.llm_response import (
    LLMUserContextAggregator,
)
from pipecat.processors.frameworks.rtvi import (
    ActionResult,
    RTVIAction,
    RTVIActionArgument,
    RTVIProcessor,
)

if sys.version_info < (3, 11):

    class StrEnum(str, Enum):
        """A string-based Enum class for Python versions < 3.11."""
        def __new__(cls, value):
            """Constructor for StrEnum."""
            obj = str.__new__(cls, value)
            obj._value_ = value
            return obj
else:
    from enum import StrEnum


class RTVILLMActionType(StrEnum):
    """Enum for RTVI LLM action types."""
    APPEND_TO_MESSAGES = "append_to_messages"
    GET_CONTEXT = "get_context"
    SET_CONTEXT = "set_context"
    RUN = "run"


class RTVIHelper(BaseModel):
    """Abstract class for helpers meant to handle various service related requests."""

    def __init__(self, service: str):
        super().__init__()
        self._service = service
        self._actions = []

    def register_actions(self, rtvi: RTVIProcessor):
        """Register the actions for the RTVI LLM helper."""
        for action in self._actions:
            rtvi.register_action(action)


class RTVILLMHelper(RTVIHelper):
    """Helper class for handling RTVI LLM-related requests."""

    def __init__(self, service: str, user_aggregator: LLMUserContextAggregator):
        super().__init__(service)
        self._user_aggregator = user_aggregator
        self.setupActions()

    def setupActions(self):
        """Set up the actions for the RTVI LLM helper."""
        self._actions.append(RTVIAction(
            service=self._service,
            action=RTVILLMActionType.APPEND_TO_MESSAGES,
            result="bool",
            arguments=[
                RTVIActionArgument(name="messages", type="array"),
                RTVIActionArgument(name="run_immediately", type="bool"),
            ],
            handler=self.append_to_messages_handler,
        ))
        self._actions.append(RTVIAction(
            service=self._service,
            action=RTVILLMActionType.GET_CONTEXT,
            result="array",
            handler=self.get_context_handler
        ))
        self._actions.append(RTVIAction(
            service=self._service,
            action=RTVILLMActionType.SET_CONTEXT,
            result="bool",
            arguments=[
                RTVIActionArgument(name="messages", type="array"),
                RTVIActionArgument(name="tools", type="array"),
            ],
            handler=self.set_context_handler,
        ))
        self._actions.append(RTVIAction(
            service=self._service,
            action=RTVILLMActionType.RUN,
            result="bool",
            arguments=[RTVIActionArgument(name="interrupt", type="bool")],
            handler=self.run_handler,
        ))

    async def append_to_messages_handler(
            self, rtvi: RTVIProcessor, service: str, arguments: Dict[str, Any]) -> ActionResult:
        """Handle the LLM append-to-messages action.

        Args:
            rtvi: The RTVIProcessor instance managing the bot's real-time interaction.
            service: The name of the service handling the action.
            arguments: A dictionary of arguments for the action, including 'messages' and 'run_immediately'.

        Returns:
            ActionResult: A boolean indicating the success of the action.
        """
        print('action_llm_append_to_messages_handler', arguments)
        run_immediately = arguments["run_immediately"] if "run_immediately" in arguments else True

        if run_immediately:
            await rtvi.interrupt_bot()

        # We just interrupted the bot so it should be fine to use the
        # context directly instead of through frame.

        if "messages" in arguments and arguments["messages"]:
            frame = LLMMessagesAppendFrame(messages=arguments["messages"])
            await rtvi.push_frame(frame)

        if run_immediately:
            frame = self._user_aggregator.get_context_frame()
            await rtvi.push_frame(frame)

        return True

    async def get_context_handler(
        self, rtvi: RTVIProcessor, service: str, arguments: Dict[str, Any]
    ) -> ActionResult:
        """Handle the RTVI get-context action.

        Args:
            rtvi: The RTVIProcessor instance managing the bot's real-time interaction.
            service: The name of the service handling the action.
            arguments: A dictionary of arguments for the action.

        Returns:
            ActionResult: A dictionary containing the context messages and tools.
        """
        messages = self._user_aggregator.context.messages
        tools = (
            self._user_aggregator.context.tools
            # TODO: Is it ok that we have to depend on an openai type here?
            if not isinstance(self._user_aggregator.context.tools, NotGiven)
            else []
        )
        result = {"messages": messages, "tools": tools}
        return result

    async def set_context_handler(
        self, rtvi: RTVIProcessor, service: str, arguments: Dict[str, Any]
    ) -> ActionResult:
        """Handle the RTVI set-context action.

        Args:
            rtvi: The RTVIProcessor instance managing the bot's real-time interaction.
            service: The name of the service handling the action.
            arguments: A dictionary of arguments for the action, including 'messages' and 'tools'.

        Returns:
            ActionResult: A boolean indicating the success of the action.
        """
        run_immediately = arguments["run_immediately"] if "run_immediately" in arguments else True

        if run_immediately:
            await rtvi.interrupt_bot()

        # We just interrupted the bot so it should be find to use the
        # context directly instead of through frame.

        if "messages" in arguments and arguments["messages"]:
            frame = LLMMessagesUpdateFrame(messages=arguments["messages"])
            await rtvi.push_frame(frame)

        if "tools" in arguments and arguments["tools"]:
            frame = LLMSetToolsFrame(tools=arguments["tools"])
            await rtvi.push_frame(frame)

        if run_immediately:
            frame = self._user_aggregator.get_context_frame()
            await rtvi.push_frame(frame)

        return True

    async def run_handler(
        self, rtvi: RTVIProcessor, service: str, arguments: Dict[str, Any]
    ) -> ActionResult:
        """Handle the RTVI run action.

        Args:
            rtvi: The RTVIProcessor instance managing the bot's real-time interaction.
            service: The name of the service handling the action.
            arguments: A dictionary of arguments for the action, including 'interrupt'.

        Returns:
            ActionResult: A boolean indicating the success of the action.
        """
        interrupt = arguments["interrupt"] if "interrupt" in arguments else True
        if interrupt:
            await rtvi.interrupt_bot()
        frame = self._user_aggregator.get_context_frame()
        await rtvi.push_frame(frame)

        return True
