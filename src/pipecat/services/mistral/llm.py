import asyncio
import json
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

import httpx
from loguru import logger
from mistralai import (
    ConversationUsageInfo,
    FunctionCallEvent,
    MessageOutputEvent,
    ResponseDoneEvent,
    ResponseStartedEvent,
)
from openai.types.chat import (
    ChatCompletionMessageParam,
)
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    CancelFrame,
    Frame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import (
    OpenAIAssistantContextAggregator,
    OpenAIContextAggregatorPair,
    OpenAIUserContextAggregator,
)
from pipecat.utils.tracing.service_decorators import traced_llm

try:
    from mistralai import Mistral
    from mistralai.models import (
        CompletionArgs,
        ConversationEvents,
        ConversationInputs,
        FunctionResultEntry,
        MessageInputEntry,
        MessageOutputEntry,
    )
    from mistralai.utils import BackoffStrategy, RetryConfig, eventstreaming
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Mistral, you need to `pip install pipecat-ai[mistralai]`.")
    raise Exception(f"Missing module: {e}")


class ToolChoiceEnum(str, Enum):
    """Enum for tool choice."""

    auto = "auto"
    none = "none"
    any = "any"
    required = "required"


class MistralLLMService(LLMService):
    """LLM service for Mistral's conversation API."""

    class InputParams(BaseModel):
        """Input parameters for Mistral conversation inference."""

        max_tokens: Optional[int] = Field(default_factory=lambda: 4096, ge=1)
        temperature: Optional[float] = Field(default_factory=lambda: 0.7, ge=0.0, le=1.0)
        top_p: Optional[float] = Field(default_factory=lambda: 1.0, ge=0.0, le=1.0)
        random_seed: Optional[int] = Field(default_factory=lambda: None)
        safe_prompt: Optional[bool] = Field(default_factory=lambda: False)
        handoff_execution: Optional[str] = Field(default_factory=lambda: "server")
        store: Optional[bool] = Field(default_factory=lambda: True)
        tool_choice: Optional[ToolChoiceEnum] = Field(default_factory=lambda: ToolChoiceEnum.auto)
        extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "mistral-small-latest",
        params: Optional[InputParams] = None,
        client=None,
        **kwargs,
    ):
        """Initialize the Mistral LLM service.

        Args:
            api_key: Mistral API key for authentication.
            model: Model name to use. Defaults to mistral-small-latest".
            params: Optional model parameters for inference.
            client: Optional custom Mistral client instance.
            **kwargs: Additional arguments passed to parent LLMService.
        """
        super().__init__(model=model, params=params, **kwargs)
        params = params or MistralLLMService.InputParams()
        self._client = client or Mistral(api_key=api_key)
        self.set_model_name(model)
        self._settings = {
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "random_seed": params.random_seed,
            "safe_prompt": params.safe_prompt,
            "handoff_execution": params.handoff_execution,
            "store": params.store,
            "tool_choice": params.tool_choice,
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }
        self._current_conversation_id = None
        self._stop_streaming = False

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> OpenAIContextAggregatorPair:
        """Create OpenAI-specific context aggregators.

        Creates a pair of context aggregators optimized for OpenAI's message format,
        including support for function calls, tool usage, and image handling.

        Args:
            context: The LLM context to create aggregators for.
            user_params: Parameters for user message aggregation.
            assistant_params: Parameters for assistant message aggregation.

        Returns:
            OpenAIContextAggregatorPair: A pair of context aggregators, one for
            the user and one for the assistant, encapsulated in an
            OpenAIContextAggregatorPair.

        """
        context.set_llm_adapter(self.get_llm_adapter())
        user = OpenAIUserContextAggregator(context, params=user_params)
        assistant = OpenAIAssistantContextAggregator(context, params=assistant_params)
        return OpenAIContextAggregatorPair(_user=user, _assistant=assistant)

    def _convert_to_mistral_input(self, messages: List[Dict]) -> List[Dict]:
        """Convert messages to Mistral's conversation input format."""
        inputs = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if isinstance(content, list):
                # Handle complex content blocks
                text_content = ""
                for block in content:
                    if block["type"] == "text":
                        text_content += block["text"]
                    elif block["type"] == "image":
                        text_content += "[Image content]"
                content = text_content

            # Convert to Mistral conversation input format
            if role == "user":
                inputs.append(
                    {
                        "object": "entry",
                        "type": "message.input",
                        "role": "user",
                        "content": content,
                        "prefix": False,
                    }
                )
            elif role == "assistant":
                inputs.append(
                    {
                        "object": "entry",
                        "type": "message.output",
                        "role": "assistant",
                        "content": content,
                        "prefix": False,
                    }
                )
            elif role == "function":
                # For function/tool responses
                inputs.append(
                    {
                        "object": "entry",
                        "type": "function.result",
                        "content": content,
                        "tool_call_id": msg.get("tool_call_id", ""),
                        "result": content if isinstance(content, str) else json.dumps(content),
                    }
                )
        return inputs

    def _create_conversation_inputs(
        self, messages: List[ChatCompletionMessageParam]
    ) -> ConversationInputs:
        """Create ConversationInputs object for Mistral API."""
        inputs = []
        function_calls_pending = {}  # Track pending function calls by tool_call_id

        last_message = messages[-1]
        role = last_message.get("role")
        content = last_message.get("content", "")

        if role == "user":
            entry = MessageInputEntry(role="user", content=str(content), prefix=False)
            inputs.append(entry)

        elif role == "tool" or role == "function":
            entry = FunctionResultEntry(
                tool_call_id=last_message.get("tool_call_id", ""), result=str(content)
            )
            inputs.append(entry)

        elif role == "assistant":
            if content and str(content).strip():
                entry = MessageOutputEntry(
                    role="assistant",
                    content=str(content),
                )
                inputs.append(entry)

        return inputs

    async def _process_conversation_stream(
        self, context: OpenAILLMContext, stream: eventstreaming.EventStreamAsync[ConversationEvents]
    ):
        """Process the conversation event stream from Mistral."""
        try:
            full_response = ""
            function_calls = []
            prompt_tokens = 0
            completion_tokens = 0

            # Dictionaries to accumulate argument fragments and track function calls
            function_args_accumulators = {}  # {tool_call_id: accumulated_args}
            function_metadata = {}  # {tool_call_id: {'name': function_name, 'complete': False}}

            async for event in stream:
                if self._stop_streaming:
                    break

                await self.stop_ttfb_metrics()

                data = event.data

                # Management of different types of events
                if isinstance(data, ResponseStartedEvent):
                    # Store the conversation ID for future calls
                    if hasattr(data, "conversation_id"):
                        self._current_conversation_id = data.conversation_id

                elif isinstance(data, MessageOutputEvent):
                    if data.content:
                        await self.push_frame(LLMTextFrame(str(data.content)))
                        full_response += str(data.content)

                elif (
                    isinstance(data, FunctionCallEvent)
                    and getattr(data, "type", "") == "function.call.delta"
                ):
                    # Argument fragment for a function call
                    tool_call_id = data.tool_call_id
                    delta_arguments = (
                        data.arguments
                        if isinstance(data.arguments, str)
                        else json.dumps(data.arguments or {})
                    )
                    function_name = data.name if hasattr(data, "name") else "unknown_function"

                    # If this is a new function call (first fragment)
                    if tool_call_id not in function_args_accumulators:
                        function_args_accumulators[tool_call_id] = delta_arguments
                        function_metadata[tool_call_id] = {"name": function_name, "complete": False}
                    else:
                        function_args_accumulators[tool_call_id] += delta_arguments

                    # Try to parse the accumulated JSON to see if it is complete
                    try:
                        accumulated_args = function_args_accumulators[tool_call_id]

                        # Clean up the accumulated arguments (may contain multiple JSON fragments)
                        # It is assumed that the fragments form a complete JSON when concatenated
                        parsed_args = json.loads(accumulated_args)

                        # If we reach here, the JSON is complete
                        if not function_metadata[tool_call_id]["complete"]:
                            function_metadata[tool_call_id]["complete"] = True

                            # Create the FunctionCallFromLLM object and add it to the list
                            function_call = FunctionCallFromLLM(
                                context=context,
                                tool_call_id=tool_call_id,
                                function_name=function_metadata[tool_call_id]["name"],
                                arguments=parsed_args,
                            )
                            function_calls.append(function_call)

                            # Notify the pipeline that a function call is in progress
                            await self.push_frame(
                                FunctionCallInProgressFrame(
                                    function_name=function_metadata[tool_call_id]["name"],
                                    tool_call_id=tool_call_id,
                                    arguments=parsed_args,
                                )
                            )

                    except json.JSONDecodeError:
                        # The JSON is not yet complete, continue accumulating
                        pass

                elif isinstance(data, ResponseDoneEvent):
                    # End of the response - retrieve usage information
                    if isinstance(data.usage, ConversationUsageInfo):
                        prompt_tokens = data.usage.prompt_tokens
                        completion_tokens = data.usage.completion_tokens

            # Once the stream is finished, execute the function calls if necessary
            if function_calls:
                await self.run_function_calls(function_calls)

            # If we have token metrics, report them
            if prompt_tokens and completion_tokens:
                await self._report_usage_metrics(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"Error processing conversation stream: {e}")

    @traced_llm
    async def _process_context(self, context: OpenAILLMContext):
        """Process the conversation context with Mistral's API."""
        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()

            # Convert messages to Mistral's conversation format
            inputs = self._create_conversation_inputs(context.messages)

            # Prepare the completion arguments
            completion_args = CompletionArgs(
                max_tokens=self._settings["max_tokens"],
                temperature=self._settings["temperature"],
                top_p=self._settings["top_p"],
                tool_choice=self._settings["tool_choice"],
            )

            # If we have a conversation ID, we're continuing an existing conversation
            if self._current_conversation_id:
                # Use append_stream to add to existing conversation
                stream = await self._client.beta.conversations.append_stream_async(
                    conversation_id=self._current_conversation_id,
                    inputs=inputs,
                    stream=True,
                    store=self._settings["store"],
                    completion_args=completion_args,
                    retries=RetryConfig(
                        strategy="backoff",
                        backoff=BackoffStrategy(
                            initial_interval=500,
                            max_interval=1000,
                            exponent=1.1,
                            max_elapsed_time=5000,
                        ),
                        retry_connection_errors=True,
                    ),
                )
            else:
                # Start a new conversation
                stream = await self._client.beta.conversations.start_stream_async(
                    inputs=inputs,
                    stream=True,
                    tools=context.tools or [],
                    store=self._settings["store"],
                    completion_args=completion_args,
                    model=self.model_name,
                    retries=RetryConfig(
                        strategy="backoff",
                        backoff=BackoffStrategy(
                            initial_interval=500,
                            max_interval=1000,
                            exponent=1.1,
                            max_elapsed_time=5000,
                        ),
                        retry_connection_errors=True,
                    ),
                )

            await self.start_ttfb_metrics()
            await self._process_conversation_stream(context, stream)

        except asyncio.CancelledError:
            raise
        except httpx.TimeoutException:
            await self._call_event_handler("on_completion_timeout")
        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

    async def cancel(self, frame: CancelFrame):
        """Cancel any ongoing requests."""
        await super().cancel(frame)
        self._stop_streaming = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and route them appropriately.

        Handles various frame types including context frames, message frames,
        vision frames, and settings updates.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)
        context = None

        if isinstance(frame, OpenAILLMContextFrame):
            context: OpenAILLMContext = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)

    async def _report_usage_metrics(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cache_creation_input_tokens: int,
        cache_read_input_tokens: int,
    ):
        if any([prompt_tokens, completion_tokens]):
            tokens = LLMTokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            await self.start_llm_usage_metrics(tokens)

    async def _update_settings(self, settings: Mapping[str, Any]):
        """Update service settings."""
        for key, value in settings.items():
            if key in self._settings:
                logger.info(f"Updating LLM setting {key} to: [{value}]")
                self._settings[key] = value
            elif key == "model":
                logger.info(f"Updating LLM setting {key} to: [{value}]")
                self.set_model_name(value)
            else:
                logger.warning(f"Unknown setting for {self.name} service: {key}")
