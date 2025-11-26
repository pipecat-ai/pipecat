#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS AgentCore Processor Module.

This module defines the AWSAgentCoreProcessor, which invokes agents hosted on
Amazon Bedrock AgentCore Runtime and streams their responses as LLMTextFrames.
"""

import asyncio
import json
import os
from typing import Callable, Optional

import aioboto3
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def default_context_to_payload_transformer(
    context: LLMContext | OpenAILLMContext,
) -> Optional[str]:
    """Default transformer to create AgentCore payload from LLM context.

    Extracts the latest user or system message text and wraps it in {"prompt": "<text>"}.

    Args:
        context: The LLM context containing conversation messages.

    Returns:
        A JSON string payload for AgentCore, or None if no valid message found.
    """
    messages = context.messages

    if not messages:
        return None

    last_message = messages[-1]
    if isinstance(last_message, LLMSpecificMessage) or last_message.get("role") not in (
        "user",
        "system",
    ):
        return None

    content = last_message.get("content")
    if not content:
        return None

    if isinstance(content, str):
        prompt = content
    elif isinstance(content, list):
        prompt = " ".join([part.get("text", "") for part in content])
    else:
        return None

    return json.dumps({"prompt": prompt})


def default_response_to_output_transformer(response_line: str) -> Optional[str]:
    """Default transformer to extract output text from AgentCore response.

    Expects responses with {"response": "<text>"} format.

    Args:
        response_line: The raw response line from AgentCore (without "data: " prefix).

    Returns:
        The extracted output text, or None if no text found.
    """
    response_json = json.loads(response_line)
    return response_json.get("response")


class AWSAgentCoreProcessor(FrameProcessor):
    """Processor that runs an Amazon Bedrock AgentCore agent.

    Input:
        - LLMContextFrame: Supplies a context used to invoke the agent.

    Output:
        - LLMTextFrame: The agent's text response(s).
          A single agent invocation may result in multiple text frames.

    This processor transforms the input context to a payload for the AgentCore
    agent, and transforms the agent's response(s) into output text frame(s). Both
    mappings are configurable via transformers. Below is the default behavior.

    Input transformer (context_to_payload_transformer):
        - Grabs the latest user or system message (if it's the latest message)
        - Extracts its text content
        - Constructs a payload that looks like {"prompt": "<text>"}

    Output transformer (response_to_output_transformer):
        - Expects responses that look like {"response": "<text>"}
        - Extracts the text for use in the LLMTextFrame(s)
    """

    def __init__(
        self,
        agentArn: str,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region: Optional[str] = None,
        context_to_payload_transformer: Optional[
            Callable[[LLMContext | OpenAILLMContext], Optional[str]]
        ] = None,
        response_to_output_transformer: Optional[Callable[[str], Optional[str]]] = None,
        **kwargs,
    ):
        """Initialize the AWS AgentCore processor.

        Args:
            agentArn: The Amazon Web Services Resource Name (ARN) of the agent.
            aws_access_key: AWS access key ID. If None, uses default credentials.
            aws_secret_key: AWS secret access key. If None, uses default credentials.
            aws_session_token: AWS session token for temporary credentials.
            aws_region: AWS region.
            context_to_payload_transformer: Optional callable to transform
                LLMContext into AgentCore payload string. If None, uses
                default_context_to_payload_transformer.
            response_to_output_transformer: Optional callable to extract output text
                from AgentCore response. If None, uses
                default_response_to_output_transformer.
            **kwargs: Additional arguments passed to parent FrameProcessor.
        """
        super().__init__(**kwargs)

        self._agentArn = agentArn
        self._aws_session = aioboto3.Session()

        # Store AWS session parameters for creating client in async context
        self._aws_params = {
            "aws_access_key_id": aws_access_key or os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": aws_secret_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
            "aws_session_token": aws_session_token or os.getenv("AWS_SESSION_TOKEN"),
            "region_name": aws_region or os.getenv("AWS_REGION", "us-east-1"),
        }

        # Set transformers with defaults
        self._context_to_payload_transformer = (
            context_to_payload_transformer or default_context_to_payload_transformer
        )
        self._response_to_output_transformer = (
            response_to_output_transformer or default_response_to_output_transformer
        )

        # State for managing output response bookends
        self._output_response_open = False
        self._last_text_frame_time: Optional[float] = None
        self._close_task: Optional[asyncio.Task] = None
        self._output_response_timeout = 1.0  # seconds

    async def _close_output_response_after_timeout(self):
        """Close the output response after timeout if no new text frames arrive."""
        await asyncio.sleep(self._output_response_timeout)
        if self._output_response_open:
            self._output_response_open = False
            await self.push_frame(LLMFullResponseEndFrame())

    async def _push_text_frame(self, text: str):
        """Push a text frame, managing output response bookends."""
        # Cancel any pending close task
        if self._close_task and not self._close_task.done():
            await self.cancel_task(self._close_task)

        # Open output response if needed
        if not self._output_response_open:
            await self.push_frame(LLMFullResponseStartFrame())
            self._output_response_open = True

        # Push the text frame
        await self.push_frame(LLMTextFrame(text))
        self._last_text_frame_time = asyncio.get_event_loop().time()

        # Schedule closing the output response after timeout
        self._close_task = self.create_task(self._close_output_response_after_timeout())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle LLM message frames.

        Args:
            frame: The incoming frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, (LLMContextFrame, OpenAILLMContextFrame)):
            # Create payload to invoke AgentCore agent
            payload = self._context_to_payload_transformer(frame.context)

            if not payload:
                return

            async with self._aws_session.client("bedrock-agentcore", **self._aws_params) as client:
                # Invoke the AgentCore agent
                response = await client.invoke_agent_runtime(
                    agentRuntimeArn=self._agentArn, payload=payload.encode()
                )

                # Determine if this is a streamed multi-part response, which
                # will affect our parsing
                is_multi_part_response = "text/event-stream" in response.get("contentType", "")

                # Handle each response part (there may be one, for single
                # responses, or multiple, for streamed multi-part responses)
                async for part in response.get("response", []):
                    part_string = part.decode("utf-8")

                    # In streamed multi-part responses, each part might have
                    # one or more lines, each of which starts with "data: ".
                    # Treat each line as a response.
                    if is_multi_part_response:
                        for line in part_string.split("\n"):
                            # Get response text from this line
                            if not line:
                                continue
                            if not line.startswith("data: "):
                                logger.warning(f"Expected line to start with 'data: ', got: {line}")
                                continue
                            line = line[6:]  # omit "data: "

                            # Transform response line to output text
                            text = self._response_to_output_transformer(line)
                            if text:
                                await self._push_text_frame(text)

                    # In single-part responses, the whole part is one response
                    # and there's no "data: " prefix
                    else:
                        # Transform response part string to output text
                        text = self._response_to_output_transformer(part_string)
                        if text:
                            await self._push_text_frame(text)

                # Final close if output response is still open after all parts processed
                if self._output_response_open:
                    if self._close_task and not self._close_task.done():
                        await self.cancel_task(self._close_task)
                    self._output_response_open = False
                    await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)
