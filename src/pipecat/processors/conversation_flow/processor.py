from typing import List, Optional

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .flow import ConversationFlow


# processor.py
class ConversationFlowProcessor(FrameProcessor):
    """Processor that manages conversation flow based on function calls"""

    def __init__(self, flow_config: dict):
        super().__init__()
        self.flow = ConversationFlow(flow_config)
        self.initialized = False

    async def initialize(self, initial_messages: List[dict]):
        """Initialize the conversation with starting messages and functions"""
        if not self.initialized:
            messages = initial_messages + [self.flow.get_current_message()]
            logger.info(f"Initializing with messages: {messages}")
            logger.info(f"Initial tools: {self.flow.get_current_functions()}")

            await self.push_frame(LLMMessagesUpdateFrame(messages=messages))
            await self.push_frame(LLMSetToolsFrame(tools=self.flow.get_current_functions()))
            self.initialized = True
            logger.info(f"Initialized conversation flow at node: {self.flow.current_node}")
        else:
            logger.warning("Attempted to initialize ConversationFlowProcessor multiple times")

    async def handle_transition(self, function_name: str):
        """Handle state transition triggered by function call"""
        logger.info(f"Handling transition for function: {function_name}")
        available_functions = self.flow.get_available_function_names()
        logger.info(f"Available functions: {available_functions}")

        if function_name in available_functions:
            new_node = self.flow.transition(function_name)
            if new_node:
                if self.flow.get_current_actions():
                    logger.info(f"Executing actions for node {new_node}")
                    await self._execute_actions(self.flow.get_current_actions())

                current_message = self.flow.get_current_message()
                logger.info(f"New node message: {current_message}")
                logger.info(f"New node functions: {self.flow.get_current_functions()}")

                await self.push_frame(LLMMessagesAppendFrame(messages=[current_message]))
                await self.push_frame(LLMSetToolsFrame(tools=self.flow.get_current_functions()))

                logger.info("Transition complete")
        else:
            logger.warning(
                f"Received invalid function call '{function_name}' for node '{self.flow.current_node}'. "
                f"Available functions are: {available_functions}"
            )

    async def _execute_actions(self, actions: Optional[List[dict]]) -> None:
        """Execute actions specified for the current node"""
        if not actions:
            return

        for action in actions:
            if action["type"] == "tts.say":
                logger.info(f"Executing TTS action: {action['text']}")
                await self.push_frame(TTSSpeakFrame(text=action["text"]))
            else:
                logger.warning(f"Unknown action type: {action['type']}")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process incoming frames and manage state transitions"""
        if not self.initialized:
            logger.warning("ConversationFlowProcessor received frames before initialization")
            await self.push_frame(frame, direction)
            return

        # Pass all frames through
        await self.push_frame(frame, direction)
