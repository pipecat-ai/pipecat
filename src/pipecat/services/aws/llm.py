#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS Bedrock integration for Large Language Model services.

This module provides AWS Bedrock LLM service implementation with support for
Amazon Nova and Anthropic Claude models, including vision capabilities and
function calling.
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.adapters.services.bedrock_adapter import (
    AWSBedrockLLMAdapter,
    AWSBedrockLLMInvocationParams,
)
from pipecat.frames.frames import (
    Frame,
    FunctionCallFromLLM,
    LLMContextFrame,
    LLMEnablePromptCachingFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import NOT_GIVEN, LLMSettings, _NotGiven
from pipecat.utils.tracing.service_decorators import traced_llm

try:
    import aioboto3
    from botocore.config import Config
    from botocore.exceptions import ReadTimeoutError
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use AWS services, you need to `pip install pipecat-ai[aws]`. Also, remember to set `AWS_SECRET_ACCESS_KEY`, `AWS_ACCESS_KEY_ID`, and `AWS_REGION` environment variable."
    )
    raise Exception(f"Missing module: {e}")


@dataclass
class AWSBedrockLLMSettings(LLMSettings):
    """Settings for AWSBedrockLLMService.

    Parameters:
        stop_sequences: List of strings that stop generation.
        latency: Performance mode - "standard" or "optimized".
        enable_prompt_caching: Whether to enable prompt caching by adding cachePoint
            markers to system prompts and tool definitions. Can reduce TTFT by up to
            85% for multi-turn conversations. See:
            https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
        additional_model_request_fields: Additional model-specific parameters.
    """

    stop_sequences: list[str] | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    latency: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    enable_prompt_caching: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    additional_model_request_fields: dict[str, Any] | _NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )


class AWSBedrockLLMService(LLMService):
    """AWS Bedrock Large Language Model service implementation.

    Provides inference capabilities for AWS Bedrock models including Amazon Nova
    and Anthropic Claude. Supports streaming responses, function calling, and
    vision capabilities.
    """

    Settings = AWSBedrockLLMSettings
    _settings: Settings

    # Overriding the default adapter to use the Anthropic one.
    adapter_class = AWSBedrockLLMAdapter

    class InputParams(BaseModel):
        """Input parameters for AWS Bedrock LLM service.

        .. deprecated:: 0.0.105
            Use ``AWSBedrockLLMService.Settings`` instead. Pass settings directly via the
            ``settings`` parameter of :class:`AWSBedrockLLMService`.

        Parameters:
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature between 0.0 and 1.0.
            top_p: Nucleus sampling parameter between 0.0 and 1.0.
            stop_sequences: List of strings that stop generation.
            latency: Performance mode - "standard" or "optimized".
            additional_model_request_fields: Additional model-specific parameters.
        """

        max_tokens: int | None = Field(default=None, ge=1)
        temperature: float | None = Field(default=None, ge=0.0, le=1.0)
        top_p: float | None = Field(default=None, ge=0.0, le=1.0)
        stop_sequences: list[str] | None = Field(default_factory=lambda: [])
        latency: str | None = Field(default=None)
        additional_model_request_fields: dict[str, Any] | None = Field(default_factory=dict)

    def __init__(
        self,
        *,
        model: str | None = None,
        aws_access_key: str | None = None,
        aws_secret_key: str | None = None,
        aws_session_token: str | None = None,
        aws_region: str | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        stop_sequences: list[str] | None = None,
        client_config: Config | None = None,
        retry_timeout_secs: float | None = 5.0,
        retry_on_timeout: bool | None = False,
        **kwargs,
    ):
        """Initialize the AWS Bedrock LLM service.

        Args:
            model: The AWS Bedrock model identifier to use.

                .. deprecated:: 0.0.105
                    Use ``settings=AWSBedrockLLMService.Settings(model=...)`` instead.

            aws_access_key: AWS access key ID. If None, uses default credentials.
            aws_secret_key: AWS secret access key. If None, uses default credentials.
            aws_session_token: AWS session token for temporary credentials.
            aws_region: AWS region for the Bedrock service.
            params: Model parameters and configuration.

                .. deprecated:: 0.0.105
                    Use ``settings=AWSBedrockLLMService.Settings(...)`` instead.

            settings: Runtime-updatable settings for this service.  When both
                deprecated parameters and *settings* are provided, *settings*
                values take precedence.
            stop_sequences: List of strings that stop generation.

                .. deprecated:: 0.0.105
                    Use ``settings=AWSBedrockLLMService.Settings(stop_sequences=...)`` instead.

            client_config: Custom boto3 client configuration.
            retry_timeout_secs: Request timeout in seconds for retry logic.
            retry_on_timeout: Whether to retry the request once if it times out.
            **kwargs: Additional arguments passed to parent LLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="us.amazon.nova-lite-v1:0",
            system_instruction=None,
            max_tokens=None,
            temperature=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            stop_sequences=None,
            latency=None,
            enable_prompt_caching=False,
            additional_model_request_fields={},
        )

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model
        if stop_sequences is not None:
            self._warn_init_param_moved_to_settings("stop_sequences", "stop_sequences")
            default_settings.stop_sequences = stop_sequences

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.max_tokens = params.max_tokens
                default_settings.temperature = params.temperature
                default_settings.top_p = params.top_p
                if params.stop_sequences:
                    default_settings.stop_sequences = params.stop_sequences
                default_settings.latency = params.latency
                if isinstance(params.additional_model_request_fields, dict):
                    default_settings.additional_model_request_fields = (
                        params.additional_model_request_fields
                    )

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)

        # Initialize the AWS Bedrock client
        if not client_config:
            client_config = Config(
                connect_timeout=300,  # 5 minutes
                read_timeout=300,  # 5 minutes
                retries={"max_attempts": 3},
            )

        self._aws_session = aioboto3.Session()

        # Store AWS session parameters for creating client in async context
        self._aws_params = {
            "aws_access_key_id": aws_access_key or os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": aws_secret_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
            "aws_session_token": aws_session_token or os.getenv("AWS_SESSION_TOKEN"),
            "region_name": aws_region or os.getenv("AWS_REGION", "us-east-1"),
            "config": client_config,
        }

        self._retry_timeout_secs = retry_timeout_secs
        self._retry_on_timeout = retry_on_timeout

        logger.info(f"Using AWS Bedrock model: {self._settings.model}")
        if self._settings.system_instruction:
            logger.debug(f"{self}: Using system instruction: {self._settings.system_instruction}")

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate usage metrics.

        Returns:
            True if metrics generation is supported.
        """
        return True

    def _build_inference_config(self) -> dict[str, Any]:
        """Build inference config with only the parameters that are set.

        This prevents conflicts with models (e.g., Claude Sonnet 4.5) that don't
        allow certain parameter combinations like temperature and top_p together.

        Returns:
            Dictionary containing only the inference parameters that are not None.
        """
        inference_config = {}
        if self._settings.max_tokens is not None:
            inference_config["maxTokens"] = self._settings.max_tokens
        if self._settings.temperature is not None:
            inference_config["temperature"] = self._settings.temperature
        if self._settings.top_p is not None:
            inference_config["topP"] = self._settings.top_p
        if self._settings.stop_sequences:
            inference_config["stopSequences"] = self._settings.stop_sequences
        return inference_config

    async def run_inference(
        self,
        context: LLMContext,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> str | None:
        """Run a one-shot, out-of-band (i.e. out-of-pipeline) inference with the given LLM context.

        Args:
            context: The LLM context containing conversation history.
            max_tokens: Optional maximum number of tokens to generate. If provided,
                overrides the service's default max_tokens setting.
            system_instruction: Optional system instruction to use for this inference.
                If provided, overrides any system instruction in the context.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        messages = []
        system = []
        effective_instruction = system_instruction or self._settings.system_instruction
        adapter: AWSBedrockLLMAdapter = self.get_llm_adapter()
        params: AWSBedrockLLMInvocationParams = adapter.get_llm_invocation_params(
            context, system_instruction=effective_instruction
        )
        messages = params["messages"]
        system = params["system"]  # [{"text": "system message"}] or None

        # Prepare request parameters using the same method as streaming
        inference_config = self._build_inference_config()

        # Override maxTokens if provided
        if max_tokens is not None:
            inference_config["maxTokens"] = max_tokens

        request_params = {
            "modelId": self._settings.model,
            "messages": messages,
            "additionalModelRequestFields": self._settings.additional_model_request_fields,
        }

        if inference_config:
            request_params["inferenceConfig"] = inference_config

        if system:
            request_params["system"] = system

        async with self._aws_session.client(
            service_name="bedrock-runtime", **self._aws_params
        ) as client:
            # Call Bedrock without streaming
            response = await client.converse(**request_params)

            # Extract the response text
            if (
                "output" in response
                and "message" in response["output"]
                and "content" in response["output"]["message"]
            ):
                content = response["output"]["message"]["content"]
                if isinstance(content, list):
                    for item in content:
                        if item.get("text"):
                            return item["text"]
                elif isinstance(content, str):
                    return content

            return None

    async def _create_converse_stream(self, client, request_params):
        """Create converse stream with optional timeout and retry.

        Args:
            client: The AWS Bedrock client instance.
            request_params: Parameters for the converse_stream call.

        Returns:
            Async stream of response events.
        """
        if self._retry_on_timeout:
            try:
                response = await asyncio.wait_for(
                    client.converse_stream(**request_params), timeout=self._retry_timeout_secs
                )
                return response
            except (TimeoutError, ReadTimeoutError) as e:
                # Retry, this time without a timeout so we get a response
                logger.debug(f"{self}: Retrying converse_stream due to timeout")
                response = await client.converse_stream(**request_params)
                return response
        else:
            response = await client.converse_stream(**request_params)
            return response

    def _create_no_op_tool(self):
        """Create a no-operation tool for AWS Bedrock when tool content exists but no tools are defined.

        This is required because AWS Bedrock doesn't allow empty tool configurations after tools were
        previously set. Other LLM vendors allow NOT_GIVEN or empty tool configurations,
        but AWS Bedrock requires at least one tool to be defined.
        """
        return {
            "toolSpec": {
                "name": "no_operation",
                "description": "Internal placeholder function. Do not call this function.",
                "inputSchema": {"json": {"type": "object", "properties": {}, "required": []}},
            }
        }

    def _get_llm_invocation_params(self, context: LLMContext) -> AWSBedrockLLMInvocationParams:
        adapter: AWSBedrockLLMAdapter = self.get_llm_adapter()
        params: AWSBedrockLLMInvocationParams = adapter.get_llm_invocation_params(
            context, system_instruction=self._settings.system_instruction
        )
        return params

    @traced_llm
    async def _process_context(self, context: LLMContext):
        # Usage tracking
        prompt_tokens = 0
        completion_tokens = 0
        completion_tokens_estimate = 0
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0
        use_completion_tokens_estimate = False

        using_noop_tool = False

        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()

            await self.start_ttfb_metrics()

            params_from_context = self._get_llm_invocation_params(context)
            messages = params_from_context["messages"]
            system = params_from_context["system"]
            tools = params_from_context["tools"]
            tool_choice = params_from_context["tool_choice"]

            # Set up inference config - only include parameters that are set
            inference_config = self._build_inference_config()

            # Prepare request parameters
            request_params = {
                "modelId": self._settings.model,
                "messages": messages,
                "additionalModelRequestFields": self._settings.additional_model_request_fields,
            }

            # Only add inference config if it has parameters
            if inference_config:
                request_params["inferenceConfig"] = inference_config

            # Add system message
            if system:
                request_params["system"] = system

            # Check if messages contain tool use or tool result content blocks
            has_tool_content = False
            for message in messages:
                if isinstance(message.get("content"), list):
                    for content_item in message["content"]:
                        if "toolUse" in content_item or "toolResult" in content_item:
                            has_tool_content = True
                            break
                if has_tool_content:
                    break

            # Handle tools: use current tools, or no-op if tool content exists but no current tools
            if has_tool_content and not tools:
                tools = [self._create_no_op_tool()]
                using_noop_tool = True

            if tools:
                tool_config = {"tools": tools}

                # Only add tool_choice if we have real tools (not just no-op)
                if not using_noop_tool and tool_choice:
                    if tool_choice == "auto":
                        tool_config["toolChoice"] = {"auto": {}}
                    elif tool_choice == "none":
                        # Skip adding toolChoice for "none"
                        pass
                    elif isinstance(tool_choice, dict) and "function" in tool_choice:
                        tool_config["toolChoice"] = {
                            "tool": {"name": tool_choice["function"]["name"]}
                        }

                request_params["toolConfig"] = tool_config

            # Add performance config if latency is specified
            if self._settings.latency in ["standard", "optimized"]:
                request_params["performanceConfig"] = {"latency": self._settings.latency}

            # Add cache checkpoints to system prompts and tool definitions.
            # This enables prompt caching for providers that support it (e.g.
            # Anthropic Claude on Bedrock), reducing TTFT by up to 85% on
            # multi-turn conversations where the system prompt stays constant.
            if self._settings.enable_prompt_caching:
                if "system" in request_params and request_params["system"]:
                    system_list = request_params["system"]
                    if not any("cachePoint" in item for item in system_list):
                        system_list.append({"cachePoint": {"type": "default"}})
                if (
                    "toolConfig" in request_params
                    and "tools" in request_params["toolConfig"]
                    and request_params["toolConfig"]["tools"]
                ):
                    tools_list = request_params["toolConfig"]["tools"]
                    if not any("cachePoint" in t for t in tools_list):
                        tools_list.append({"cachePoint": {"type": "default"}})

            # Log request params with messages redacted for logging
            adapter = self.get_llm_adapter()
            messages_for_logging = adapter.get_messages_for_logging(context)
            logger.debug(
                f"{self}: Generating chat from context [{system}] | {messages_for_logging}"
            )

            async with self._aws_session.client(
                service_name="bedrock-runtime", **self._aws_params
            ) as client:
                # Call AWS Bedrock with streaming
                response = await self._create_converse_stream(client, request_params)

                await self.stop_ttfb_metrics()

                # Process the streaming response
                tool_use_block = None
                json_accumulator = ""

                function_calls = []

                async for event in response["stream"]:
                    # Handle text content
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"]["delta"]
                        if "text" in delta:
                            await self._push_llm_text(delta["text"])
                            completion_tokens_estimate += self._estimate_tokens(delta["text"])
                        elif "toolUse" in delta and "input" in delta["toolUse"]:
                            # Handle partial JSON for tool use
                            json_accumulator += delta["toolUse"]["input"]
                            completion_tokens_estimate += self._estimate_tokens(
                                delta["toolUse"]["input"]
                            )

                    # Handle tool use start
                    elif "contentBlockStart" in event:
                        content_block_start = event["contentBlockStart"]["start"]
                        if "toolUse" in content_block_start:
                            tool_use_block = {
                                "id": content_block_start["toolUse"].get("toolUseId", ""),
                                "name": content_block_start["toolUse"].get("name", ""),
                            }
                            json_accumulator = ""

                    # Handle message completion with tool use
                    elif "messageStop" in event and "stopReason" in event["messageStop"]:
                        if event["messageStop"]["stopReason"] == "tool_use" and tool_use_block:
                            try:
                                arguments = json.loads(json_accumulator) if json_accumulator else {}

                                # Only call function if it's not the no_operation tool
                                if not using_noop_tool:
                                    function_calls.append(
                                        FunctionCallFromLLM(
                                            context=context,
                                            tool_call_id=tool_use_block["id"],
                                            function_name=tool_use_block["name"],
                                            arguments=arguments,
                                        )
                                    )
                                else:
                                    logger.debug("Ignoring no_operation tool call")
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse tool arguments: {json_accumulator}")

                    # Handle usage metrics if available
                    if "metadata" in event and "usage" in event["metadata"]:
                        usage = event["metadata"]["usage"]
                        prompt_tokens += usage.get("inputTokens", 0)
                        completion_tokens += usage.get("outputTokens", 0)
                        cache_read_input_tokens += usage.get("cacheReadInputTokens", 0)
                        cache_creation_input_tokens += usage.get("cacheWriteInputTokens", 0)

            await self.run_function_calls(function_calls)
        except asyncio.CancelledError:
            # If we're interrupted, we won't get a complete usage report. So set our flag to use the
            # token estimate. The reraise the exception so all the processors running in this task
            # also get cancelled.
            use_completion_tokens_estimate = True
            raise
        except (TimeoutError, ReadTimeoutError):
            await self._call_event_handler("on_completion_timeout")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())
            comp_tokens = (
                completion_tokens
                if not use_completion_tokens_estimate
                else completion_tokens_estimate
            )
            await self._report_usage_metrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=comp_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle LLM-specific frame types.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            await self._process_context(frame.context)
        elif isinstance(frame, LLMEnablePromptCachingFrame):
            logger.debug(f"Setting enable prompt caching to: [{frame.enable}]")
            self._settings.enable_prompt_caching = frame.enable
        else:
            await self.push_frame(frame, direction)

    def _estimate_tokens(self, text: str) -> int:
        return int(len(re.split(r"[^\w]+", text)) * 1.3)

    async def _report_usage_metrics(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_input_tokens: int,
        cache_creation_input_tokens: int,
    ):
        if prompt_tokens or completion_tokens:
            tokens = LLMTokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
            )
            await self.start_llm_usage_metrics(tokens)
