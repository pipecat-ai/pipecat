#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import List, Optional

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    FunctionCallResultFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .flow import ConversationFlow


class ConversationFlowProcessor(FrameProcessor):
    """Processor that manages conversation flow based on function calls"""

    def __init__(self, flow_config: dict):
        super().__init__()
        self.flow = ConversationFlow(flow_config)
        self.initialized = False

    async def initialize(self, initial_messages: List[dict]):
        """Initialize the conversation with starting messages and functions"""
        if not self.initialized:
            # Combine initial messages with the first node's message
            # TODO: Not sure if this is needed
            messages = initial_messages + [self.flow.get_current_message()]

            await self.push_frame(LLMMessagesUpdateFrame(messages=initial_messages))
            await self.push_frame(LLMSetToolsFrame(tools=self.flow.get_current_functions()))
            self.initialized = True
            logger.info(f"Initialized conversation flow at node: {self.flow.current_node}")
        else:
            logger.warning("Attempted to initialize ConversationFlowProcessor multiple times")

    async def _execute_actions(self, actions: Optional[List[dict]]) -> None:
        """Execute actions specified for the current node"""
        if not actions:
            return

        for action in actions:
            if action["type"] == "tts.say":
                await self.push_frame(TTSSpeakFrame(text=action["text"]))
            else:
                logger.warning(f"Unknown action type: {action['type']}")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process incoming frames and manage state transitions"""
        if not self.initialized:
            logger.warning("ConversationFlowProcessor received frames before initialization")
            return

        if isinstance(frame, FunctionCallResultFrame):
            available_functions = self.flow.get_available_function_names()
            if frame.function_name in available_functions:
                new_node = self.flow.transition(frame.function_name)
                if new_node:
                    # Execute any entry actions for the new node
                    await self._execute_actions(self.flow.get_current_actions())

                    # Update the LLM context with the new node's message
                    await self.push_frame(
                        LLMMessagesAppendFrame(messages=[self.flow.get_current_message()])
                    )

                    # Update available functions for this node
                    await self.push_frame(LLMSetToolsFrame(tools=self.flow.get_current_functions()))

                    logger.info(f"Transitioned to node: {new_node}")
            else:
                logger.warning(
                    f"Received function call '{frame.function_name}' not in available functions: {available_functions}"
                )
