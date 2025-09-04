"""
Strands Agent integration for Pipecat.

This module provides integration with Strands Agents for handling conversational AI
interactions. It supports both single agent and multi-agent graphs.
"""

from typing import Optional

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
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
        if isinstance(frame, OpenAILLMContextFrame):
            text = frame.context.messages[-1]["content"]
            await self._ainvoke(str(text).strip())
        else:
            await self.push_frame(frame, direction)

    async def _ainvoke(self, text: str):
        """Invoke the Strands agent with the provided text and stream results as Pipecat frames.

        Args:
            text: The user input text to process through the agent or graph.
        """
        logger.debug(f"Invoking Strands agent with: {text}")
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            if self.graph:
                # Graph does not stream; await full result then emit assistant text
                graph_result = await self.graph.invoke_async(text)
                try:
                    node_result = graph_result.results[self.graph_exit_node]
                    for agent_result in node_result.get_agent_results():
                        message = getattr(agent_result, "message", None)
                        if isinstance(message, dict) and "content" in message:
                            for block in message["content"]:
                                if isinstance(block, dict) and "text" in block:
                                    await self.push_frame(LLMTextFrame(str(block["text"])))
                except Exception as parse_err:
                    logger.warning(f"Failed to extract messages from GraphResult: {parse_err}")
            else:
                # Agent supports streaming events via async iterator
                async for event in self.agent.stream_async(text):
                    if isinstance(event, dict) and "data" in event:
                        await self.push_frame(LLMTextFrame(str(event["data"])))
        except GeneratorExit:
            logger.warning(f"{self} generator was closed prematurely")
        except Exception as e:
            logger.exception(f"{self} an unknown error occurred: {e}")
        finally:
            await self.push_frame(LLMFullResponseEndFrame())