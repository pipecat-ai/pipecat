import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

import httpx
from loguru import logger
from mistralai import (
    ConversationUsageInfo,
    FunctionCallEvent,
    MessageOutputEvent,
    ResponseDoneEvent,
    ResponseStartedEvent,
)
from openai import NOT_GIVEN, NotGiven
from pydantic import BaseModel, Field

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    Frame,
    FunctionCallCancelFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMAssistantContextAggregator,
    LLMUserAggregatorParams,
    LLMUserContextAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.utils.tracing.service_decorators import traced_llm

if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletionAssistantMessageParam,
        ChatCompletionMessageParam,
        ChatCompletionToolChoiceOptionParam,
        ChatCompletionToolParam,
        ChatCompletionUserMessageParam,
    )

try:
    from mistralai import Mistral
    from mistralai.models import (
        CompletionArgs,
        ConversationEvents,
        ConversationInputs,
    )
    from mistralai.utils import BackoffStrategy, RetryConfig, eventstreaming
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Mistral, you need to `pip install pipecat-ai[mistralai]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class MistralContextAggregatorPair:
    """Pair of context aggregators for Mistral conversations."""

    _user: "MistralUserContextAggregator"
    _assistant: "MistralAssistantContextAggregator"

    def user(self) -> "MistralUserContextAggregator":
        """Get the user context aggregator.

        Returns:
            The user context aggregator instance.
        """
        return self._user

    def assistant(self) -> "MistralAssistantContextAggregator":
        """Get the assistant context aggregator.

        Returns:
            The assistant context aggregator instance.
        """
        return self._assistant


class MistralLLMContext(OpenAILLMContext):
    """LLM context specialized for Mistral's conversation format."""

    def __init__(
        self,
        messages: Optional[List[ChatCompletionMessageParam]] = None,
        tools: List[ChatCompletionToolParam] | NotGiven | ToolsSchema = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
    ):
        """Initialize the Mistral LLM context.

        Args:
            messages: Initial list of conversation messages.
            tools: Available function calling tools.
            tool_choice: Tool selection preference.
        """
        super().__init__(messages=messages, tools=tools, tool_choice=tool_choice)

    @staticmethod
    def upgrade_to_mistral(obj: OpenAILLMContext) -> "MistralLLMContext":
        """Upgrade an OpenAI context to Mistral format.

        Converts message format and restructures content for Mistral compatibility.

        Args:
            obj: The OpenAI context to upgrade.

        Returns:
            The upgraded Mistral context.
        """
        if isinstance(obj, OpenAILLMContext) and not isinstance(obj, MistralLLMContext):
            new_obj = MistralLLMContext(messages=obj.messages, tools=obj.tools, tool_choice=obj.tool_choice)
            new_obj.__dict__.update(obj.__dict__)
            return new_obj
        return obj

    @staticmethod
    def from_messages(messages: List[ChatCompletionMessageParam]) -> "MistralLLMContext":
        """Create context from a list of messages.

        Args:
            messages: List of conversation messages.

        Returns:
            New Anthropic context with the provided messages.
        """
        return MistralLLMContext(messages=messages)


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
        super().__init__(**kwargs)
        params = params or MistralLLMService.InputParams()
        self._mistral_client = client or Mistral(api_key=api_key)
        self.set_model_name(model)
        self._settings = {
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "random_seed": params.random_seed,
            "safe_prompt": params.safe_prompt,
            "handoff_execution": params.handoff_execution,
            "store": params.store,
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }
        self._current_conversation_id = None
        self._current_stream_task = None
        self._stop_streaming = False

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate usage metrics.

        Returns:
            True, as Anthropic provides detailed token usage metrics.
        """
        return True

    @property
    def model_name(self) -> str:
        """Return model_name.

        Returns:
            str: String of model name
        """
        return self._model_name

    def set_model_name(self, model: str):
        """Set model_name.

        Args:
            model (str): String of model name
        """
        super().set_model_name(model)

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> MistralContextAggregatorPair:
        """Create contact aggregator from OpenAILLMCFontext.

        Args:
            context: The LLM context.
            user_params: User aggregator parameters.
            assistant_params: Assistant aggregator parameters.

        Returns:
            A pair of context aggregators, one for the user and one for the assistant,
            encapsulated in an MistralContextAggregatorPair.            
        """
        context = MistralLLMContext.upgrade_to_mistral(context)
        user = MistralUserContextAggregator(context, params=user_params)
        assistant = MistralAssistantContextAggregator(context, params=assistant_params)
        return MistralContextAggregatorPair(_user=user, _assistant=assistant)

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
                    {"object": "entry", "type": "message.input", "role": "user", "content": content, "prefix": False}
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

    def _create_conversation_inputs(self, messages: List[ChatCompletionMessageParam]) -> ConversationInputs:
        """Create ConversationInputs object for Mistral API."""
        inputs = []
        for msg in messages:
            if msg.get("role") == "user":
                inputs.append(
                    {
                        "object": "entry",
                        "type": "message.input",
                        "role": "user",
                        "content": msg.get("content"),
                        "prefix": False,
                    }
                )
            elif msg.get("role") == "assistant" and "tool_calls" in msg:
                tool_calls = msg.get("tool_calls")
                for tool_call in tool_calls if tool_calls else []:
                    inputs.append(
                        {
                            "object": "entry",
                            "type": tool_call.get("type"),
                            "tool_call_id": tool_call.get("id"),
                            "function": tool_call.get("function"),
                        }
                    )
            elif msg.get("role") == "function":
                inputs.append(
                    {
                        "object": "entry",
                        "type": "function.result",
                        "tool_call_id": msg.get("tool_call_id", ""),
                        "result": msg.get("content"),
                    }
                )
            else:
                inputs.append(
                    {
                        "object": "entry",
                        "type": "message.output",
                        "role": msg.get("role"),
                        "content": msg.get("content"),
                        "prefix": False,
                    }
                )
        return inputs

    async def _stop_current_stream(self):
        """Stop the current streaming task if it exists."""
        if self._current_stream_task:
            self._stop_streaming = True
            try:
                await self.cancel_task(self._current_stream_task)
            except Exception as e:
                logger.warning(f"Error stopping stream task: {e}")
            finally:
                self._current_stream_task = None
                self._stop_streaming = False

    async def _process_conversation_stream(
        self, context: MistralLLMContext, stream: eventstreaming.EventStreamAsync[ConversationEvents]
    ):
        """Process the conversation event stream from Mistral."""
        try:
            full_response = ""
            function_calls = []
            prompt_tokens = 0
            completion_tokens = 0

            async for event in stream:
                if self._stop_streaming:
                    break
                logger.debug(event)

                data = event.data

                # Gestion des différents types d'événements
                if isinstance(data, ResponseStartedEvent):
                    # Stocker l'ID de conversation pour les futurs appels
                    if hasattr(data, "conversation_id"):
                        self._current_conversation_id = data.conversation_id

                elif isinstance(data, MessageOutputEvent):
                    if data.content:
                        await self.push_frame(LLMTextFrame(str(data.content)))
                        full_response += str(data.content)

                elif isinstance(data, (FunctionCallEvent, dict)) and getattr(event, "type", "") == "function.call":
                    # Appel de fonction détecté
                    tool_call_id = data.tool_call_id
                    function_name = data.name
                    arguments = data.arguments

                    # Créer et stocker l'appel de fonction
                    function_call = FunctionCallFromLLM(
                        context=context,
                        tool_call_id=tool_call_id,
                        function_name=function_name,
                        arguments=arguments if isinstance(arguments, dict) else json.loads(arguments or "{}"),
                    )
                    function_calls.append(function_call)

                    # Notifier le pipeline qu'un appel de fonction est en cours
                    await self.push_frame(
                        FunctionCallInProgressFrame(
                            function_name=function_name, tool_call_id=tool_call_id, arguments=arguments
                        )
                    )

                elif isinstance(data, ResponseDoneEvent):
                    # Fin de la réponse - récupérer les infos d'usage
                    if isinstance(data.usage, ConversationUsageInfo):
                        prompt_tokens = data.usage.prompt_tokens
                        completion_tokens = data.usage.completion_tokens

            # Une fois le stream terminé, on exécute les appels de fonction si nécessaire
            if function_calls:
                await self.run_function_calls(function_calls)

            # Si on a des métriques de tokens, on les rapporte
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
    async def _process_context(self, context: MistralLLMContext):
        """Process the conversation context with Mistral's API."""
        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()

            # Convert messages to Mistral's conversation format
            inputs = self._create_conversation_inputs(context.messages)
            logger.debug(inputs)

            # Prepare the completion arguments
            completion_args = CompletionArgs(
                max_tokens=self._settings["max_tokens"],
                temperature=self._settings["temperature"],
                top_p=self._settings["top_p"],
            )

            # If we have a conversation ID, we're continuing an existing conversation
            if self._current_conversation_id:
                # Use append_stream to add to existing conversation
                stream = await self._mistral_client.beta.conversations.append_stream_async(
                    conversation_id=self._current_conversation_id,
                    inputs=inputs,
                    stream=True,
                    store=self._settings["store"],
                    completion_args=completion_args,
                    retries=RetryConfig(
                        strategy="backoff",
                        backoff=BackoffStrategy(
                            initial_interval=500, max_interval=1000, exponent=1.1, max_elapsed_time=5000
                        ),
                        retry_connection_errors=True,
                    ),
                )
            else:
                # Start a new conversation
                stream = await self._mistral_client.beta.conversations.start_stream_async(
                    inputs=inputs,
                    stream=True,
                    store=self._settings["store"],
                    completion_args=completion_args,
                    model=self.model_name,
                    retries=RetryConfig(
                        strategy="backoff",
                        backoff=BackoffStrategy(
                            initial_interval=500, max_interval=1000, exponent=1.1, max_elapsed_time=5000
                        ),
                        retry_connection_errors=True,
                    ),
                )

            await self.start_ttfb_metrics()
            await self._process_conversation_stream(context, stream)
            await self.stop_ttfb_metrics()

        except asyncio.CancelledError:
            raise
        except httpx.TimeoutException:
            await self._call_event_handler("on_completion_timeout")
        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

    async def cancel(self, frame=None):
        """Cancel any ongoing requests."""
        await super().cancel(frame)
        await self._stop_current_stream()

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
            context = MistralLLMContext.upgrade_to_mistral(frame.context)
        elif isinstance(frame, LLMMessagesFrame):
            context = MistralLLMContext.from_messages(frame.messages)
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


class MistralUserContextAggregator(LLMUserContextAggregator):
    """Mistral-specific user context aggregator."""

    pass


class MistralAssistantContextAggregator(LLMAssistantContextAggregator):
    """Context aggregator for assistant messages in Mistral conversations."""

    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        """Handle a function call that is starting."""
        assistant_message = ChatCompletionAssistantMessageParam(
            role="assistant", content=f"Calling function {frame.function_name} with arguments: {frame.arguments}"
        )
        self._context.add_message(assistant_message)
        self._context.add_message(
            ChatCompletionUserMessageParam(role="user", content=f"Function call {frame.tool_call_id} in progress")
        )

    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        """Handle the result of a completed function call."""
        result_str = json.dumps(frame.result) if frame.result else "COMPLETED"
        await self._update_function_call_result(frame.function_name, frame.tool_call_id, result_str)

    async def handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        """Handle cancellation of a function call."""
        await self._update_function_call_result(frame.function_name, frame.tool_call_id, "CANCELLED")

    async def _update_function_call_result(self, function_name: str, tool_call_id: str, result: Any):
        """Update the context with function call results."""
        # Find and update the appropriate message
        for i, message in enumerate(self._context.messages):
            if (
                message["role"] == "user"
                and isinstance(message["content"], str)
                and f"Function call {tool_call_id}" in message["content"]
            ):
                self._context.messages[i]["content"] = f"Function call {tool_call_id} result: {result}"
                break
