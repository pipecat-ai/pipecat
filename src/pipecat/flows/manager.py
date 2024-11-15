#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from asyncio import iscoroutinefunction
from typing import Callable, Dict, List, Optional

from loguru import logger

from pipecat.frames.frames import (
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
    TTSSpeakFrame,
)

from .state import FlowState


class FlowManager:
    """Manages conversation flows in a Pipecat pipeline.

    This manager handles the progression through a flow defined by nodes, where each node
    represents a state in the conversation. Each node has:
    - A message for the LLM
    - Available functions that can be called
    - Optional pre-actions to execute before LLM inference
    - Optional post-actions to execute after LLM inference

    The flow is defined by a configuration that specifies:
    - Initial node
    - Available nodes and their configurations
    - Transitions between nodes via function calls
    """

    def __init__(self, flow_config: dict, task, tts=None):
        """Initialize the flow manager.

        Args:
            flow_config: Dictionary containing the complete flow configuration,
                        including initial_node and node configurations
            task: PipelineTask instance used to queue frames into the pipeline
        """
        self.flow = FlowState(flow_config)
        self.initialized = False
        self.task = task
        self.tts = tts
        self.action_handlers: Dict[str, Callable] = {}

        # Register built-in actions
        self.register_action("tts_say", self._handle_tts_action)

    async def initialize(self, initial_messages: List[dict]):
        """Initialize the flow with starting messages and functions.

        This method sets up the initial context, combining any system-level
        messages with the initial node's message and functions.

        Args:
            initial_messages: List of initial messages (typically system messages)
                            to include in the context
        """
        if not self.initialized:
            messages = initial_messages + [self.flow.get_current_message()]
            await self.task.queue_frame(LLMMessagesUpdateFrame(messages=messages))
            await self.task.queue_frame(LLMSetToolsFrame(tools=self.flow.get_current_functions()))
            self.initialized = True
            logger.debug(f"Initialized flow at node: {self.flow.current_node}")
        else:
            logger.warning("Attempted to initialize FlowManager multiple times")

    async def register_functions(self, llm_service):
        """Register all functions from the flow configuration with the LLM service.

        This method sets up function handlers for all functions defined across all nodes.
        When a function is called, it will automatically trigger the appropriate node
        transition.

        Note: This registers handlers for all possible functions, but the LLM's access
        to functions is controlled separately through LLMSetToolsFrame. The LLM will
        only see the functions available in the current node.

        Args:
            llm_service: The LLM service to register functions with
        """

        async def handle_function_call(
            function_name, tool_call_id, arguments, llm, context, result_callback
        ):
            await self.handle_transition(function_name)
            await result_callback("Acknowledged")

        # Register all functions from all nodes
        for node in self.flow.nodes.values():
            for function in node.functions:
                function_name = function["function"]["name"]
                llm_service.register_function(function_name, handle_function_call)

    def register_action(self, action_type: str, handler: Callable):
        """Register a handler for a specific action type.

        Args:
            action_type: String identifier for the action (e.g., "tts_say")
            handler: Async or sync function that handles the action.
                    Should accept action configuration as parameter.
        """
        if not callable(handler):
            raise ValueError("Action handler must be callable")
        self.action_handlers[action_type] = handler

    async def _execute_actions(self, actions: Optional[List[dict]]) -> None:
        """Execute actions specified for the current node.

        Args:
            actions: List of action configurations to execute

        Note:
            Each action must have a 'type' field matching a registered handler
        """
        if not actions:
            return

        for action in actions:
            action_type = action["type"]
            if action_type in self.action_handlers:
                handler = self.action_handlers[action_type]
                try:
                    if iscoroutinefunction(handler):
                        await handler(action)
                    else:
                        handler(action)
                except Exception as e:
                    logger.warning(f"Error executing action {action_type}: {e}")
            else:
                logger.warning(f"No handler registered for action type: {action_type}")

    async def _handle_tts_action(self, action: dict):
        """Built-in handler for TTS actions"""
        if self.tts:
            # Direct call to TTS service to speak text
            await self.tts.say(action["text"])
        else:
            # Fall back to queued TTS if no direct service available
            await self.task.queue_frame(TTSSpeakFrame(text=action["text"]))

    async def handle_transition(self, function_name: str):
        """Handle node transition triggered by a function call.

        This method:
        1. Validates the function call against available functions
        2. Transitions to the new node if appropriate
        3. Executes any pre-actions before updating the LLM context
        4. Updates the LLM context with new messages and available functions
        5. Executes any post-actions after updating the LLM context

        Args:
            function_name: Name of the function that was called

        Raises:
            RuntimeError: If handle_transition is called before initialization
        """
        if not self.initialized:
            raise RuntimeError("FlowManager must be initialized before handling transitions")

        available_functions = self.flow.get_available_function_names()
        current_node = self.flow.get_current_node()

        if function_name in available_functions:
            new_node = self.flow.transition(function_name)
            if new_node:
                # Only execute actions if we actually changed nodes
                is_new_node = new_node != current_node

                # Execute pre-actions before updating LLM context
                if is_new_node and self.flow.get_current_pre_actions():
                    logger.debug(f"Executing pre-actions for node {new_node}")
                    await self._execute_actions(self.flow.get_current_pre_actions())

                # Update LLM context and tools
                current_message = self.flow.get_current_message()
                await self.task.queue_frame(LLMMessagesAppendFrame(messages=[current_message]))
                await self.task.queue_frame(
                    LLMSetToolsFrame(tools=self.flow.get_current_functions())
                )

                # Execute post-actions after updating LLM context
                if is_new_node and self.flow.get_current_post_actions():
                    logger.debug(f"Executing post-actions for node {new_node}")
                    await self._execute_actions(self.flow.get_current_post_actions())

                logger.debug(f"Transition to node {new_node} complete")
        else:
            logger.warning(
                f"Received invalid function call '{function_name}' for node '{self.flow.current_node}'. "
                f"Available functions are: {available_functions}"
            )
