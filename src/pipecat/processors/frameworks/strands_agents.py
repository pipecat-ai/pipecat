"""Strands Agent integration for Pipecat.

This module provides integration with Strands Agents for handling conversational AI
interactions. It supports both single agent and multi-agent graphs.
"""

from typing import Optional

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

try:
    from strands import Agent
    from strands.multiagent.graph import Graph
except ModuleNotFoundError as e:
    logger.exception("In order to use Strands Agents, you need to `pip install strands-agents`.")
    raise Exception(f"Missing module: {e}")


class StrandsAgentsProcessor(FrameProcessor):
    """Processor that integrates Strands Agents with Pipecat's frame pipeline.

    This processor takes LLM message frames, extracts the latest user message,
    and processes it through either a single Strands Agent or a multi-agent Graph.
    The response is streamed back as text frames with appropriate response markers.

    Supports both single agent streaming and graph-based multi-agent workflows.
    """

    def __init__(
        self,
        agent: Optional[Agent] = None,
        graph: Optional[Graph] = None,
        graph_exit_node: Optional[str] = None,
    ):
        """Initialize the Strands Agents processor.

        Args:
            agent: The Strands Agent to use for single-agent processing.
            graph: The Strands multi-agent Graph to use for graph-based processing.
            graph_exit_node: The exit node name when using graph-based processing.

        Raises:
            AssertionError: If neither agent nor graph is provided, or if graph is
                          provided without a graph_exit_node.
        """
        super().__init__()
        self.agent = agent
        self.graph = graph
        self.graph_exit_node = graph_exit_node

        assert self.agent or self.graph, "Either agent or graph must be provided"

        if self.graph:
            assert self.graph_exit_node, "graph_exit_node must be provided if graph is provided"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle LLM message frames.

        Args:
            frame: The incoming frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, (LLMContextFrame, OpenAILLMContextFrame)):
            messages = frame.context.get_messages()
            if messages:
                last_message = messages[-1]
                await self._ainvoke(str(last_message["content"]).strip())
        else:
            await self.push_frame(frame, direction)

    async def _ainvoke(self, text: str):
        """Invoke the Strands agent with the provided text and stream results as Pipecat frames.

        Args:
            text: The user input text to process through the agent or graph.
        """
        logger.debug(f"Invoking Strands agent with: {text}")
        ttfb_tracking = True
        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

            if self.graph:
                # Graph does not stream; await full result then emit assistant text
                graph_result = await self.graph.invoke_async(text)
                if ttfb_tracking:
                    await self.stop_ttfb_metrics()
                    ttfb_tracking = False
                try:
                    node_result = graph_result.results[self.graph_exit_node]
                    logger.debug(f"Node result: {node_result}")
                    for agent_result in node_result.get_agent_results():
                        # Push to TTS service
                        message = getattr(agent_result, "message", None)
                        if isinstance(message, dict) and "content" in message:
                            for block in message["content"]:
                                if isinstance(block, dict) and "text" in block:
                                    await self.push_frame(LLMTextFrame(str(block["text"])))
                        # Update usage metrics
                        await self._report_usage_metrics(
                            agent_result.metrics.accumulated_usage.get("inputTokens", 0),
                            agent_result.metrics.accumulated_usage.get("outputTokens", 0),
                            agent_result.metrics.accumulated_usage.get("totalTokens", 0),
                        )
                except Exception as parse_err:
                    logger.warning(f"Failed to extract messages from GraphResult: {parse_err}")
            else:
                # Agent supports streaming events via async iterator
                async for event in self.agent.stream_async(text):
                    # Push to TTS service
                    if isinstance(event, dict) and "data" in event:
                        await self.push_frame(LLMTextFrame(str(event["data"])))
                        if ttfb_tracking:
                            await self.stop_ttfb_metrics()
                            ttfb_tracking = False

                    # Update usage metrics
                    if (
                        isinstance(event, dict)
                        and "event" in event
                        and "metadata" in event["event"]
                    ):
                        if "usage" in event["event"]["metadata"]:
                            usage = event["event"]["metadata"]["usage"]
                            await self._report_usage_metrics(
                                usage.get("inputTokens", 0),
                                usage.get("outputTokens", 0),
                                usage.get("totalTokens", 0),
                            )
        except GeneratorExit:
            logger.warning(f"{self} generator was closed prematurely")
        except Exception as e:
            logger.exception(f"{self} an unknown error occurred: {e}")
        finally:
            if ttfb_tracking:
                await self.stop_ttfb_metrics()
                ttfb_tracking = False
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate performance metrics.

        Returns:
            True as this service supports metrics generation.
        """
        return True

    async def _report_usage_metrics(
        self, prompt_tokens: int, completion_tokens: int, total_tokens: int
    ):
        tokens = LLMTokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        await self.start_llm_usage_metrics(tokens)
