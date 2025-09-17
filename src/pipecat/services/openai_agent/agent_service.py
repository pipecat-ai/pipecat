#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Agents SDK integration service.

Provides integration with the OpenAI Agents SDK for building AI applications
within Pipecat pipelines. This service allows leveraging agent loops, handoffs,
guardrails, sessions, and tools from the OpenAI Agents SDK.
"""

import asyncio
import os
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    override,
    runtime_checkable,
)

from loguru import logger

try:
    from agents import Agent, InputGuardrail, OutputGuardrail, Runner, Tool
    from agents.result import RunResult, RunResultStreaming
    from agents.stream_events import StreamEvent
except ImportError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use OpenAI Agents SDK, you need to `pip install openai-agents`. "
        "Also, set `OPENAI_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TextFrame,
    UserImageRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService


@runtime_checkable
class ToolLike(Protocol):
    """Protocol for tool-like objects."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Tool call interface."""
        ...


@runtime_checkable
class AgentLike(Protocol):
    """Protocol for agent-like objects."""

    name: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Agent call interface."""
        ...


class OpenAIAgentService(AIService):
    """OpenAI Agents SDK service for Pipecat.

    Integrates the OpenAI Agents SDK with Pipecat's pipeline architecture,
    enabling advanced agentic workflows with features like handoffs, guardrails,
    sessions, and tools within real-time conversational AI applications.

    The service processes text input frames and generates streaming responses
    using the agent's configured capabilities.
    """

    def __init__(
        self,
        *,
        agent: Optional[Agent] = None,
        name: str = "Assistant",
        instructions: Union[str, Sequence[str]] = "You are a helpful assistant.",
        handoffs: Optional[Sequence[AgentLike]] = None,
        tools: Optional[Sequence[ToolLike]] = None,
        input_guardrails: Optional[Sequence[InputGuardrail]] = None,
        output_guardrails: Optional[Sequence[OutputGuardrail]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        session_config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        streaming: bool = True,
        **kwargs,
    ):
        """Initialize the OpenAI Agent service.

        Args:
            agent: Pre-configured Agent instance. If provided, other agent configuration
                parameters will be ignored.
            name: Name of the agent for identification and handoffs.
            instructions: System instructions that define the agent's behavior.
            handoffs: List of other agents this agent can hand off to.
            tools: List of callable functions the agent can use as tools.
            input_guardrails: List of input validation guardrails.
            output_guardrails: List of output validation guardrails.
            model_config: Configuration for the underlying language model.
            session_config: Configuration for session management.
            api_key: OpenAI API key. If not provided, will use OPENAI_API_KEY env var.
            streaming: Whether to use streaming responses for real-time output.
            **kwargs: Additional arguments passed to the parent AIService.
        """
        super().__init__(**kwargs)

        # Set up API key
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY"):
            logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")

        # Create or use existing agent
        if agent:
            self._agent = agent
        else:
            # Convert sequences to lists and handle string instructions
            agent_handoffs: List[Any] = list(handoffs) if handoffs else []
            agent_tools: List[Any] = list(tools) if tools else []
            agent_input_guardrails: List[Any] = list(input_guardrails) if input_guardrails else []
            agent_output_guardrails: List[Any] = (
                list(output_guardrails) if output_guardrails else []
            )

            # Handle instructions - convert sequence to string if needed
            if isinstance(instructions, str):
                agent_instructions = instructions
            else:
                agent_instructions = " ".join(str(instr) for instr in instructions)

            self._agent = Agent(
                name=name,
                instructions=agent_instructions,
                handoffs=agent_handoffs,
                tools=agent_tools,
                input_guardrails=agent_input_guardrails,
                output_guardrails=agent_output_guardrails,
                model=model_config.get("model", "gpt-4o") if model_config else "gpt-4o",
            )

        self._streaming = streaming
        self._session_config = session_config or {}
        self._current_session = None
        self._accumulated_text = ""

        # Set model name for metrics
        if model_config and "model" in model_config:
            self.set_model_name(model_config["model"])
        else:
            self.set_model_name("gpt-4o")  # Default model

        logger.info(f"Initialized OpenAI Agent service: {self._agent.name}")

    @property
    def agent(self) -> Agent:
        """Get the underlying OpenAI Agent.

        Returns:
            The configured Agent instance.
        """
        return self._agent

    def update_agent_config(
        self,
        *,
        instructions: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Update agent configuration dynamically.

        Args:
            instructions: New system instructions for the agent.
            model_config: Updated model configuration.
            **kwargs: Additional agent configuration parameters.
        """
        if instructions:
            self._agent.instructions = instructions
            logger.info(f"Updated agent instructions for {self._agent.name}")

        if model_config:
            # Note: OpenAI Agents SDK handles model configuration during agent creation
            # We can't update model_config after agent is created, but we can update our model name
            if "model" in model_config:
                self.set_model_name(model_config["model"])
            logger.info(f"Updated model config for {self._agent.name}")

    async def start(self, frame: StartFrame):
        """Start the OpenAI Agent service.

        Initializes the agent session and prepares for processing.

        Args:
            frame: The start frame containing initialization parameters.
        """
        logger.info(f"Starting OpenAI Agent service: {self._agent.name}")
        await super().start(frame)

    async def stop(self, frame: EndFrame):
        """Stop the OpenAI Agent service.

        Cleans up resources and ends the current session.

        Args:
            frame: The end frame.
        """
        logger.info(f"Stopping OpenAI Agent service: {self._agent.name}")
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the OpenAI Agent service.

        Cancels any ongoing operations.

        Args:
            frame: The cancel frame.
        """
        logger.info(f"Cancelling OpenAI Agent service: {self._agent.name}")
        await super().cancel(frame)

    @override
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process frames and handle agent interactions.

        Processes text input frames by running them through the OpenAI Agent
        and streams the results back as LLM frames.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            # Process text input through the agent directly
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self._process_agent_request(frame.text)
                await self.push_frame(LLMFullResponseEndFrame())
            except Exception as e:
                logger.error(f"Error processing agent request: {e}")
                await self.push_error(ErrorFrame(f"Agent processing error: {e}"))
        else:
            # For frames we don't handle, pass them through with direction
            await self.push_frame(frame, direction)

    async def _process_agent_request(self, input_text: str):
        """Process an agent request and stream the results.

        Args:
            input_text: The user input text to process.
        """
        logger.debug(f"Processing agent request: {input_text}")

        if self._streaming:
            await self._process_streaming_response(input_text)
        else:
            await self._process_non_streaming_response(input_text)

    async def _process_streaming_response(self, input_text: str):
        """Process a streaming agent response.

        Args:
            input_text: The user input text to process.
        """
        try:
            # Run the agent with streaming
            result: RunResultStreaming = Runner.run_streamed(
                self._agent, input_text, context=self._session_config
            )

            # Process the stream events
            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    # Handle token-by-token streaming
                    # Only check for delta on events that are known to have it
                    if hasattr(event.data, "delta") and getattr(event.data, "delta", None):
                        delta_text = getattr(event.data, "delta", "")
                        if delta_text:
                            await self.push_frame(LLMTextFrame(text=delta_text))

                elif event.type == "run_item_stream_event":
                    # Handle completed items
                    if event.item.type == "message_output_item":
                        # Get the complete message text
                        message_text = self._extract_message_text(event.item)
                        if message_text and message_text != self._accumulated_text:
                            # Send any new text that wasn't already streamed
                            new_text = message_text[len(self._accumulated_text) :]
                            if new_text:
                                await self.push_frame(LLMTextFrame(text=new_text))
                            self._accumulated_text = message_text

                    elif event.item.type == "tool_call_item":
                        # Use getattr for safe attribute access
                        tool_name = getattr(event.item, "tool_name", "unknown")
                        logger.debug(f"Tool called: {tool_name}")

                    elif event.item.type == "tool_call_output_item":
                        output = getattr(event.item, "output", "no output")
                        logger.debug(f"Tool output: {output}")

                elif event.type == "agent_updated_stream_event":
                    logger.debug(f"Agent updated: {event.new_agent.name}")

            # Reset accumulated text for next request
            self._accumulated_text = ""

        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            raise

    async def _process_non_streaming_response(self, input_text: str):
        """Process a non-streaming agent response.

        Args:
            input_text: The user input text to process.
        """
        try:
            # Run the agent without streaming
            result: RunResult = await Runner.run(
                self._agent, input_text, context=self._session_config
            )

            # Send the final output
            if result.final_output:
                await self.push_frame(LLMTextFrame(text=result.final_output))

        except Exception as e:
            logger.error(f"Error in non-streaming response: {e}")
            raise

    def _extract_message_text(self, item) -> str:
        """Extract text from a message output item.

        Args:
            item: The message output item from the agent.

        Returns:
            The extracted text content.
        """
        try:
            # Handle different message item formats
            if hasattr(item, "content"):
                if isinstance(item.content, str):
                    return item.content
                elif isinstance(item.content, list):
                    # Extract text from content array
                    text_parts = []
                    for content_part in item.content:
                        if isinstance(content_part, dict) and content_part.get("type") == "text":
                            text_parts.append(content_part.get("text", ""))
                        elif isinstance(content_part, str):
                            text_parts.append(content_part)
                    return "".join(text_parts)

            # Fallback: try to get text through string conversion
            return str(item)

        except Exception as e:
            logger.warning(f"Could not extract text from message item: {e}")
            return ""

    async def add_tool(self, tool_function: ToolLike):
        """Add a tool function to the agent.

        Args:
            tool_function: A callable function or Tool object to add as a tool.
        """
        if hasattr(self._agent, "tools"):
            # Cast to Any to handle the type variance issue
            tools_list: List[Any] = self._agent.tools
            tools_list.append(tool_function)
            tool_name = getattr(
                tool_function, "__name__", getattr(tool_function, "name", "unknown")
            )
            logger.info(f"Added tool {tool_name} to agent {self._agent.name}")

    async def add_handoff_agent(self, agent: AgentLike):
        """Add a handoff agent.

        Args:
            agent: Another Agent instance or handoff object that this agent can hand off to.
        """
        if hasattr(self._agent, "handoffs"):
            # Cast to Any to handle the type variance issue
            handoffs_list: List[Any] = self._agent.handoffs
            handoffs_list.append(agent)
            agent_name = getattr(agent, "name", "unknown")
            logger.info(f"Added handoff agent {agent_name} to agent {self._agent.name}")

    def get_session_context(self) -> Dict[str, Any]:
        """Get the current session context.

        Returns:
            Dictionary containing the current session context.
        """
        return self._session_config.copy()

    def update_session_context(self, context: Dict[str, Any]):
        """Update the session context.

        Args:
            context: Dictionary of context updates to apply.
        """
        self._session_config.update(context)
        logger.debug(f"Updated session context for agent {self._agent.name}")
