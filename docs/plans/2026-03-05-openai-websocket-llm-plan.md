# OpenAI WebSocket LLM Service Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an LLM service that connects to OpenAI's Responses API via WebSocket for persistent-connection, lower-latency text streaming and function calling, compatible with both OpenAI cloud and local services.

**Architecture:** `OpenAIWebSocketLLMService` extends both `LLMService` (frame processing, function calling) and `WebsocketService` (connection management, auto-reconnect). A new `OpenAIWebSocketLLMAdapter` converts universal `LLMContext` to the Responses API input format. Reuses event models from `pipecat.services.openai.realtime.events`.

**Tech Stack:** Python 3.10+, websockets library, Pydantic BaseModel for config, pipecat frame-based pipeline

---

### Task 1: Create the Adapter

Convert universal `LLMContext` messages to OpenAI Responses API `input` format.

**Files:**
- Create: `src/pipecat/adapters/services/open_ai_websocket_adapter.py`
- Test: `tests/test_openai_websocket_adapter.py`
- Reference: `src/pipecat/adapters/services/open_ai_adapter.py` (OpenAI HTTP adapter)
- Reference: `src/pipecat/adapters/services/open_ai_realtime_adapter.py` (Realtime adapter)
- Reference: `src/pipecat/adapters/base_llm_adapter.py` (base class)

**Step 1: Write the failing test**

```python
# tests/test_openai_websocket_adapter.py
import pytest

from pipecat.adapters.services.open_ai_websocket_adapter import OpenAIWebSocketLLMAdapter
from pipecat.processors.aggregators.llm_context import LLMContext


class TestOpenAIWebSocketLLMAdapter:
    def setup_method(self):
        self.adapter = OpenAIWebSocketLLMAdapter()

    def test_simple_user_message(self):
        context = LLMContext(
            messages=[
                {"role": "user", "content": "Hello"},
            ]
        )
        params = self.adapter.get_llm_invocation_params(context)
        assert params["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ]
        assert params["system_instruction"] is None

    def test_system_message_extracted(self):
        context = LLMContext(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"},
            ]
        )
        params = self.adapter.get_llm_invocation_params(context)
        assert params["system_instruction"] == "You are helpful"
        assert len(params["input"]) == 1
        assert params["input"][0]["role"] == "user"

    def test_assistant_message(self):
        context = LLMContext(
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        )
        params = self.adapter.get_llm_invocation_params(context)
        assert len(params["input"]) == 2
        assert params["input"][1]["role"] == "assistant"
        assert params["input"][1]["content"] == [{"type": "output_text", "text": "Hello!"}]

    def test_tool_calls_conversion(self):
        from pipecat.adapters.schemas.function_schema import FunctionSchema
        from pipecat.adapters.schemas.tools_schema import ToolsSchema

        tools = ToolsSchema(
            standard_tools=[
                FunctionSchema(
                    name="get_weather",
                    description="Get weather",
                    properties={"location": {"type": "string"}},
                    required=["location"],
                )
            ]
        )
        context = LLMContext(messages=[{"role": "user", "content": "Hi"}], tools=tools)
        params = self.adapter.get_llm_invocation_params(context)
        assert len(params["tools"]) == 1
        assert params["tools"][0]["name"] == "get_weather"
        assert params["tools"][0]["type"] == "function"

    def test_empty_messages(self):
        context = LLMContext(messages=[])
        params = self.adapter.get_llm_invocation_params(context)
        assert params["input"] == []

    def test_multipart_user_content(self):
        context = LLMContext(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    ],
                },
            ]
        )
        params = self.adapter.get_llm_invocation_params(context)
        assert len(params["input"]) == 1
        content = params["input"][0]["content"]
        assert content[0] == {"type": "input_text", "text": "Describe this"}

    def test_tool_call_result_message(self):
        context = LLMContext(
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "function": {"name": "get_weather", "arguments": '{"location":"SF"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_123", "content": "72F sunny"},
            ]
        )
        params = self.adapter.get_llm_invocation_params(context)
        # Should produce function_call + function_call_output items
        assert any(item.get("type") == "function_call" for item in params["input"])
        assert any(item.get("type") == "function_call_output" for item in params["input"])
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_websocket_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the adapter implementation**

```python
# src/pipecat/adapters/services/open_ai_websocket_adapter.py
#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI WebSocket LLM adapter for Pipecat Responses API."""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

from loguru import logger

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage


class OpenAIWebSocketLLMInvocationParams(TypedDict):
    system_instruction: Optional[str]
    input: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]


class OpenAIWebSocketLLMAdapter(BaseLLMAdapter):
    """LLM adapter converting LLMContext to OpenAI Responses API input format."""

    @property
    def id_for_llm_specific_messages(self) -> str:
        return "openai-websocket"

    def get_llm_invocation_params(
        self, context: LLMContext
    ) -> OpenAIWebSocketLLMInvocationParams:
        messages = self.get_messages(context)
        converted = self._convert_messages(messages)
        return {
            "system_instruction": converted.system_instruction,
            "input": converted.input_items,
            "tools": self.from_standard_tools(context.tools) or [],
        }

    def get_messages_for_logging(self, context) -> List[Dict[str, Any]]:
        msgs = []
        for message in self.get_messages(context):
            msg = copy.deepcopy(message)
            if "content" in msg and isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "image_url":
                        if item["image_url"]["url"].startswith("data:image/"):
                            item["image_url"]["url"] = "data:image/..."
            msgs.append(msg)
        return msgs

    @dataclass
    class ConvertedMessages:
        input_items: List[Dict[str, Any]]
        system_instruction: Optional[str] = None

    def _convert_messages(
        self, messages: List[LLMContextMessage]
    ) -> ConvertedMessages:
        if not messages:
            return self.ConvertedMessages(input_items=[])

        messages = copy.deepcopy(messages)
        system_instruction = None

        # Extract system message
        if messages and messages[0].get("role") == "system":
            system = messages.pop(0)
            content = system.get("content")
            if isinstance(content, str):
                system_instruction = content
            elif isinstance(content, list):
                system_instruction = content[0].get("text")

        input_items = []
        for msg in messages:
            converted = self._convert_message(msg)
            if converted:
                if isinstance(converted, list):
                    input_items.extend(converted)
                else:
                    input_items.append(converted)

        return self.ConvertedMessages(
            input_items=input_items, system_instruction=system_instruction
        )

    def _convert_message(
        self, message: LLMContextMessage
    ) -> Optional[Dict[str, Any] | List[Dict[str, Any]]]:
        role = message.get("role")

        if role == "user":
            return self._convert_user_message(message)
        elif role == "assistant":
            if message.get("tool_calls"):
                return self._convert_assistant_tool_call(message)
            return self._convert_assistant_message(message)
        elif role == "tool":
            return self._convert_tool_result(message)
        else:
            logger.warning(f"Unhandled message role: {role}")
            return None

    def _convert_user_message(self, message: LLMContextMessage) -> Dict[str, Any]:
        content = message.get("content")
        if isinstance(content, str):
            return {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": content}],
            }
        elif isinstance(content, list):
            converted_content = []
            for part in content:
                if part.get("type") == "text":
                    converted_content.append(
                        {"type": "input_text", "text": part["text"]}
                    )
                elif part.get("type") == "image_url":
                    converted_content.append(
                        {
                            "type": "input_image",
                            "image_url": part["image_url"]["url"],
                        }
                    )
                else:
                    logger.warning(
                        f"Unhandled user content type: {part.get('type')}"
                    )
            return {
                "type": "message",
                "role": "user",
                "content": converted_content,
            }
        return {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": str(content)}],
        }

    def _convert_assistant_message(
        self, message: LLMContextMessage
    ) -> Dict[str, Any]:
        content = message.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )
        else:
            text = str(content)
        return {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }

    def _convert_assistant_tool_call(
        self, message: LLMContextMessage
    ) -> List[Dict[str, Any]]:
        items = []
        for tc in message.get("tool_calls", []):
            items.append(
                {
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                }
            )
        return items

    def _convert_tool_result(self, message: LLMContextMessage) -> Dict[str, Any]:
        return {
            "type": "function_call_output",
            "call_id": message.get("tool_call_id"),
            "output": str(message.get("content", "")),
        }

    @staticmethod
    def _to_function_format(function: FunctionSchema) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": function.name,
            "description": function.description,
            "parameters": {
                "type": "object",
                "properties": function.properties,
                "required": function.required,
            },
        }

    def to_provider_tools_format(
        self, tools_schema: ToolsSchema
    ) -> List[Dict[str, Any]]:
        return [
            self._to_function_format(func)
            for func in tools_schema.standard_tools
        ]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_openai_websocket_adapter.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/pipecat/adapters/services/open_ai_websocket_adapter.py tests/test_openai_websocket_adapter.py
git commit -m "feat: add OpenAI WebSocket LLM adapter for Responses API"
```

---

### Task 2: Create the WebSocket LLM Service - Connection Lifecycle

Implement the service class with WebSocket connection management.

**Files:**
- Create: `src/pipecat/services/openai/websocket_llm.py`
- Reference: `src/pipecat/services/websocket_service.py` (WebsocketService base)
- Reference: `src/pipecat/services/openai/realtime/llm.py` (Realtime connection pattern)
- Reference: `src/pipecat/services/llm_service.py` (LLMService base)

**Step 1: Write the failing test**

```python
# tests/test_openai_websocket_llm.py
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.frames.frames import EndFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame, StartFrame
from pipecat.services.openai.websocket_llm import OpenAIWebSocketLLMService


class TestOpenAIWebSocketLLMServiceConnection:
    """Test WebSocket connection lifecycle."""

    def test_init_defaults(self):
        service = OpenAIWebSocketLLMService(model="gpt-4o")
        assert service._model == "gpt-4o"
        assert service._base_url == "wss://api.openai.com/v1/responses"
        assert service._store is False
        assert service._use_previous_response_id is True

    def test_init_custom_base_url(self):
        service = OpenAIWebSocketLLMService(
            model="local-model",
            base_url="ws://localhost:8080/v1/responses",
        )
        assert service._base_url == "ws://localhost:8080/v1/responses"

    def test_init_with_api_key(self):
        service = OpenAIWebSocketLLMService(
            model="gpt-4o",
            api_key="sk-test123",
        )
        assert service._api_key == "sk-test123"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_websocket_llm.py::TestOpenAIWebSocketLLMServiceConnection -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the service class skeleton with connection lifecycle**

```python
# src/pipecat/services/openai/websocket_llm.py
#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI WebSocket LLM service using the Responses API."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.adapters.services.open_ai_websocket_adapter import OpenAIWebSocketLLMAdapter
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallFromLLM,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    StartFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.services.websocket_service import WebsocketService

try:
    from websockets.asyncio.client import connect as websocket_connect
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use OpenAI WebSocket, you need to `pip install pipecat-ai[openai]`.")
    raise Exception(f"Missing module: {e}")

from pipecat.services.openai.realtime import events


class InputParams(BaseModel):
    """User-configurable parameters for the OpenAI WebSocket LLM service.

    Parameters:
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        top_p: Nucleus sampling parameter.
        frequency_penalty: Frequency penalty.
        presence_penalty: Presence penalty.
    """

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


@dataclass
class OpenAIWebSocketLLMSettings(LLMSettings):
    """Settings for OpenAI WebSocket LLM service.

    Parameters:
        store: Whether to persist responses server-side for previous_response_id.
        use_previous_response_id: Whether to use previous_response_id for context continuity.
    """

    store: bool = False
    use_previous_response_id: bool = True


class OpenAIWebSocketLLMService(LLMService, WebsocketService):
    """OpenAI LLM service using the Responses API over WebSocket.

    Provides persistent-connection, lower-latency text streaming and function
    calling via `wss://api.openai.com/v1/responses`. Compatible with both
    OpenAI cloud and local/self-hosted services implementing the same protocol.
    """

    _settings: OpenAIWebSocketLLMSettings

    adapter_class = OpenAIWebSocketLLMAdapter

    def __init__(
        self,
        *,
        model: str,
        api_key: str = "",
        base_url: str = "wss://api.openai.com/v1/responses",
        store: bool = False,
        use_previous_response_id: bool = True,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        """Initialize the OpenAI WebSocket LLM service.

        Args:
            model: Model name (e.g., "gpt-4o", "gpt-5.2").
            api_key: API key for authentication. Optional for local services.
            base_url: WebSocket endpoint URL. Defaults to OpenAI cloud.
            store: Whether to persist responses for previous_response_id reuse.
            use_previous_response_id: Use previous_response_id for context continuity.
            params: Model parameters (temperature, max_tokens, etc.).
            **kwargs: Additional arguments passed to LLMService.
        """
        super().__init__(
            base_url=base_url,
            settings=OpenAIWebSocketLLMSettings(
                model=model,
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                top_p=params.top_p,
                frequency_penalty=params.frequency_penalty,
                presence_penalty=params.presence_penalty,
                seed=None,
                top_k=None,
                store=store,
                use_previous_response_id=use_previous_response_id,
            ),
            **kwargs,
        )
        WebsocketService.__init__(self, reconnect_on_error=True)

        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._store = store
        self._use_previous_response_id = use_previous_response_id

        self._receive_task = None
        self._context: Optional[LLMContext] = None
        self._previous_response_id: Optional[str] = None

        # Function call tracking
        self._pending_function_calls: Dict[str, Any] = {}

    def can_generate_metrics(self) -> bool:
        return True

    # --- Connection lifecycle (WebsocketService) ---

    async def _connect_websocket(self):
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._websocket = await websocket_connect(
            uri=self._base_url,
            additional_headers=headers,
        )

    async def _disconnect_websocket(self):
        if self._websocket:
            await self._websocket.close()
            self._websocket = None

    async def _receive_messages(self):
        async for message in self._websocket:
            await self._on_message(message)

    # --- Frame processor lifecycle ---

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()
        self._receive_task = self.create_task(
            self._receive_task_handler(self._report_error), "ws_receive"
        )

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._teardown()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._teardown()

    async def _teardown(self):
        await self._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task, timeout=1.0)
            self._receive_task = None

    async def _report_error(self, error_frame):
        await self.push_error(error_msg=str(error_frame))

    # --- Message handling ---

    async def _on_message(self, raw_message: str):
        try:
            data = json.loads(raw_message)
            event_type = data.get("type", "")
        except json.JSONDecodeError as e:
            logger.warning(f"{self} failed to parse message: {e}")
            return

        if event_type == "response.created":
            await self._handle_response_created(data)
        elif event_type == "response.output_text.delta":
            await self._handle_text_delta(data)
        elif event_type == "response.output_text.done":
            pass  # Text is already streamed via deltas
        elif event_type == "response.function_call_arguments.delta":
            self._handle_function_call_delta(data)
        elif event_type == "response.function_call_arguments.done":
            await self._handle_function_call_done(data)
        elif event_type == "response.output_item.added":
            self._handle_output_item_added(data)
        elif event_type == "response.output_item.done":
            pass  # Handled via response.done
        elif event_type == "response.content_part.added":
            pass
        elif event_type == "response.content_part.done":
            pass
        elif event_type == "response.done":
            await self._handle_response_done(data)
        elif event_type == "error":
            await self._handle_error(data)
        else:
            logger.debug(f"{self} unhandled event type: {event_type}")

    async def _handle_response_created(self, data: Dict[str, Any]):
        await self.stop_ttfb_metrics()

    async def _handle_text_delta(self, data: Dict[str, Any]):
        delta = data.get("delta", "")
        if delta:
            await self._push_llm_text(delta)

    def _handle_output_item_added(self, data: Dict[str, Any]):
        item = data.get("item", {})
        if item.get("type") == "function_call":
            call_id = item.get("call_id", "")
            if call_id:
                self._pending_function_calls[call_id] = {
                    "name": item.get("name", ""),
                    "arguments": "",
                }

    def _handle_function_call_delta(self, data: Dict[str, Any]):
        call_id = data.get("call_id", "")
        delta = data.get("delta", "")
        if call_id in self._pending_function_calls:
            self._pending_function_calls[call_id]["arguments"] += delta

    async def _handle_function_call_done(self, data: Dict[str, Any]):
        call_id = data.get("call_id", "")
        arguments_str = data.get("arguments", "")
        fc = self._pending_function_calls.pop(call_id, None)
        if not fc:
            logger.warning(f"{self} no pending function call for call_id: {call_id}")
            return
        try:
            args = json.loads(arguments_str)
        except json.JSONDecodeError:
            logger.error(f"{self} failed to parse function arguments: {arguments_str}")
            return

        function_calls = [
            FunctionCallFromLLM(
                context=self._context,
                tool_call_id=call_id,
                function_name=fc["name"],
                arguments=args,
            )
        ]
        await self.run_function_calls(function_calls)

    async def _handle_response_done(self, data: Dict[str, Any]):
        response = data.get("response", {})

        # Store previous_response_id
        response_id = response.get("id")
        if response_id and self._use_previous_response_id:
            self._previous_response_id = response_id

        # Usage metrics
        usage = response.get("usage")
        if usage:
            tokens = LLMTokenUsage(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
            await self.start_llm_usage_metrics(tokens)

        await self.stop_processing_metrics()
        await self.push_frame(LLMFullResponseEndFrame())

        # Check for errors in response
        if response.get("status") == "failed":
            status_details = response.get("status_details", {})
            error = status_details.get("error", {})
            await self.push_error(error_msg=error.get("message", "Response failed"))

    async def _handle_error(self, data: Dict[str, Any]):
        error = data.get("error", {})
        code = error.get("code", "")
        message = error.get("message", "Unknown error")

        if code == "previous_response_not_found":
            logger.warning(f"{self} previous_response_id not found, falling back to full context")
            self._previous_response_id = None
            # Retry with full context if we have one
            if self._context:
                await self._send_response_create(self._context)
            return

        await self.push_error(error_msg=f"WebSocket error [{code}]: {message}")

    # --- Frame processing ---

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            self._context = frame.context
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()
            await self._send_response_create(frame.context)
        elif isinstance(frame, InterruptionFrame):
            # Cancel in-flight response not directly supported over this API
            # but we stop tracking
            self._pending_function_calls.clear()
        elif isinstance(frame, LLMSetToolsFrame):
            pass  # Tools are sent with each response.create

        await self.push_frame(frame, direction)

    # --- Send request ---

    async def _send_response_create(self, context: LLMContext):
        adapter: OpenAIWebSocketLLMAdapter = self.get_llm_adapter()
        params = adapter.get_llm_invocation_params(context)

        request: Dict[str, Any] = {
            "type": "response.create",
            "model": self._model,
            "input": params["input"],
            "store": self._store,
        }

        # Add system instructions
        if params.get("system_instruction"):
            request["instructions"] = params["system_instruction"]

        # Add tools
        if params.get("tools"):
            request["tools"] = params["tools"]

        # Add previous_response_id
        if self._use_previous_response_id and self._previous_response_id:
            request["previous_response_id"] = self._previous_response_id
            # When using previous_response_id, only send new messages
            # For now, send full input - optimization can come later

        # Add model parameters
        if self._settings.temperature is not None:
            request["temperature"] = self._settings.temperature
        if self._settings.max_tokens is not None:
            request["max_output_tokens"] = self._settings.max_tokens
        if self._settings.top_p is not None:
            request["top_p"] = self._settings.top_p

        try:
            await self._websocket.send(json.dumps(request))
        except Exception as e:
            await self.push_error(error_msg=f"Error sending request: {e}", exception=e)

    # --- Context aggregator (backward compatibility) ---

    def create_context_aggregator(self, context, *, user_params=None, assistant_params=None):
        """Create context aggregators.

        NOTE: this method exists only for backward compatibility. New code
        should instead do::

            context = LLMContext(...)
            context_aggregator = LLMContextAggregatorPair(context)
        """
        from pipecat.processors.aggregators.llm_response import (
            LLMAssistantAggregatorParams,
            LLMUserAggregatorParams,
        )
        from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
        from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

        if isinstance(context, OpenAILLMContext):
            context = LLMContext.from_openai_context(context)
        return LLMContextAggregatorPair(
            context,
            user_params=user_params or LLMUserAggregatorParams(),
            assistant_params=assistant_params or LLMAssistantAggregatorParams(),
        )

    # --- Inference (for context summarization) ---

    async def run_inference(self, context, max_tokens=None):
        raise NotImplementedError(
            "run_inference is not yet supported for OpenAIWebSocketLLMService"
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_openai_websocket_llm.py::TestOpenAIWebSocketLLMServiceConnection -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/pipecat/services/openai/websocket_llm.py tests/test_openai_websocket_llm.py
git commit -m "feat: add OpenAI WebSocket LLM service with connection lifecycle"
```

---

### Task 3: Test Text Streaming End-to-End

Test the full flow from `LLMContextFrame` through WebSocket to `LLMTextFrame` output.

**Files:**
- Modify: `tests/test_openai_websocket_llm.py`

**Step 1: Add streaming test**

```python
# Add to tests/test_openai_websocket_llm.py

class MockWebSocket:
    """Mock WebSocket that sends predefined responses."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._sent: list[str] = []
        self.state = "OPEN"

    async def send(self, data: str):
        self._sent.append(data)

    async def close(self):
        self.state = "CLOSED"

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._responses:
            raise StopAsyncIteration
        return self._responses.pop(0)


class TestOpenAIWebSocketLLMServiceStreaming:
    """Test text streaming."""

    @pytest.mark.asyncio
    async def test_text_streaming(self):
        """Test that text deltas are pushed as LLMTextFrames."""
        from pipecat.frames.frames import LLMContextFrame
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.tests.utils import run_test

        responses = [
            json.dumps({"type": "response.created", "response": {"id": "resp_1", "status": "in_progress", "output": []}}),
            json.dumps({"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "role": "assistant", "id": "item_1"}}),
            json.dumps({"type": "response.content_part.added", "item_id": "item_1", "output_index": 0, "content_index": 0, "part": {"type": "text", "text": ""}}),
            json.dumps({"type": "response.output_text.delta", "delta": "Hello"}),
            json.dumps({"type": "response.output_text.delta", "delta": " world"}),
            json.dumps({"type": "response.output_text.done", "text": "Hello world"}),
            json.dumps({"type": "response.content_part.done", "part": {"type": "text", "text": "Hello world"}}),
            json.dumps({"type": "response.output_item.done", "output_index": 0, "item": {"type": "message", "role": "assistant"}}),
            json.dumps({
                "type": "response.done",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                },
            }),
        ]

        mock_ws = MockWebSocket(responses)

        service = OpenAIWebSocketLLMService(model="gpt-4o", api_key="test")

        with patch(
            "pipecat.services.openai.websocket_llm.websocket_connect",
            return_value=mock_ws,
        ):
            context = LLMContext(messages=[{"role": "user", "content": "Hi"}])
            down_frames, _ = await run_test(
                service,
                frames_to_send=[LLMContextFrame(context=context)],
                expected_down_frames=[
                    LLMFullResponseStartFrame,
                    LLMTextFrame,
                    LLMTextFrame,
                    LLMFullResponseEndFrame,
                ],
            )

        # Verify text content
        text_frames = [f for f in down_frames if isinstance(f, LLMTextFrame)]
        assert text_frames[0].text == "Hello"
        assert text_frames[1].text == " world"

        # Verify request was sent
        assert len(mock_ws._sent) == 1
        sent = json.loads(mock_ws._sent[0])
        assert sent["type"] == "response.create"
        assert sent["model"] == "gpt-4o"
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_openai_websocket_llm.py::TestOpenAIWebSocketLLMServiceStreaming -v`
Expected: PASS (may need adjustments based on run_test behavior with async receive)

**Step 3: Debug and fix any timing issues**

The mock WebSocket and `run_test` may need a small delay (`SleepFrame`) for the async receive task to process messages. Adjust as needed.

**Step 4: Commit**

```bash
git add tests/test_openai_websocket_llm.py
git commit -m "test: add text streaming test for OpenAI WebSocket LLM service"
```

---

### Task 4: Test Function Calling

Test function call accumulation and execution.

**Files:**
- Modify: `tests/test_openai_websocket_llm.py`

**Step 1: Add function calling test**

```python
# Add to tests/test_openai_websocket_llm.py

class TestOpenAIWebSocketLLMServiceFunctionCalling:
    """Test function calling."""

    @pytest.mark.asyncio
    async def test_function_call(self):
        """Test function call arguments are accumulated and executed."""
        from pipecat.frames.frames import LLMContextFrame
        from pipecat.processors.aggregators.llm_context import LLMContext

        responses = [
            json.dumps({"type": "response.created", "response": {"id": "resp_1", "status": "in_progress", "output": []}}),
            json.dumps({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_abc",
                    "name": "get_weather",
                    "arguments": "",
                },
            }),
            json.dumps({"type": "response.function_call_arguments.delta", "call_id": "call_abc", "delta": '{"loc'}),
            json.dumps({"type": "response.function_call_arguments.delta", "call_id": "call_abc", "delta": 'ation":"SF"}'}),
            json.dumps({"type": "response.function_call_arguments.done", "call_id": "call_abc", "arguments": '{"location":"SF"}'}),
            json.dumps({
                "type": "response.done",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 10, "output_tokens": 15, "total_tokens": 25},
                },
            }),
        ]

        mock_ws = MockWebSocket(responses)
        service = OpenAIWebSocketLLMService(model="gpt-4o", api_key="test")

        # Track function calls
        function_called = {}

        async def handle_get_weather(params):
            function_called["name"] = params.function_name
            function_called["args"] = params.arguments
            await params.result_callback({"temp": "72F"})

        service.register_function("get_weather", handle_get_weather)

        with patch(
            "pipecat.services.openai.websocket_llm.websocket_connect",
            return_value=mock_ws,
        ):
            context = LLMContext(messages=[{"role": "user", "content": "Weather?"}])
            await run_test(
                service,
                frames_to_send=[LLMContextFrame(context=context)],
                expected_down_frames=[
                    LLMFullResponseStartFrame,
                    LLMFullResponseEndFrame,
                ],
            )

        assert function_called.get("name") == "get_weather"
        assert function_called.get("args") == {"location": "SF"}
```

**Step 2: Run test**

Run: `uv run pytest tests/test_openai_websocket_llm.py::TestOpenAIWebSocketLLMServiceFunctionCalling -v`

**Step 3: Fix any issues and commit**

```bash
git add tests/test_openai_websocket_llm.py
git commit -m "test: add function calling test for OpenAI WebSocket LLM service"
```

---

### Task 5: Test Error Handling and previous_response_id

**Files:**
- Modify: `tests/test_openai_websocket_llm.py`

**Step 1: Add error handling tests**

```python
# Add to tests/test_openai_websocket_llm.py

class TestOpenAIWebSocketLLMServiceErrors:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_previous_response_not_found_fallback(self):
        """Test fallback to full context when previous_response_id is invalid."""
        service = OpenAIWebSocketLLMService(model="gpt-4o", api_key="test")
        service._previous_response_id = "resp_old"
        service._context = LLMContext(messages=[{"role": "user", "content": "Hi"}])

        # Simulate error
        error_data = {
            "type": "error",
            "error": {
                "code": "previous_response_not_found",
                "message": "Response not found",
            },
        }

        mock_ws = MockWebSocket([])
        service._websocket = mock_ws

        await service._handle_error(error_data)

        # Should clear previous_response_id
        assert service._previous_response_id is None

    def test_previous_response_id_stored(self):
        """Test that response ID is stored after successful response."""
        service = OpenAIWebSocketLLMService(model="gpt-4o", api_key="test")
        assert service._previous_response_id is None
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_openai_websocket_llm.py::TestOpenAIWebSocketLLMServiceErrors -v`

**Step 3: Commit**

```bash
git add tests/test_openai_websocket_llm.py
git commit -m "test: add error handling tests for OpenAI WebSocket LLM service"
```

---

### Task 6: Export from Package and Lint

**Files:**
- Modify: `src/pipecat/services/openai/__init__.py`

**Step 1: Read current init file**

Check `src/pipecat/services/openai/__init__.py` for the current exports pattern.

**Step 2: Add websocket_llm export**

Add `from .websocket_llm import *` to the imports in `__init__.py`:

```python
# src/pipecat/services/openai/__init__.py
import sys

from pipecat.services import DeprecatedModuleProxy

from .image import *
from .llm import *
from .realtime import *
from .stt import *
from .tts import *
from .websocket_llm import *

sys.modules[__name__] = DeprecatedModuleProxy(globals(), "openai", "openai.[image,llm,stt,tts,websocket_llm]")
```

**Step 3: Run linter**

Run: `uv run ruff check src/pipecat/services/openai/websocket_llm.py src/pipecat/adapters/services/open_ai_websocket_adapter.py`
Run: `uv run ruff format src/pipecat/services/openai/websocket_llm.py src/pipecat/adapters/services/open_ai_websocket_adapter.py`

Fix any issues.

**Step 4: Run all tests**

Run: `uv run pytest tests/test_openai_websocket_adapter.py tests/test_openai_websocket_llm.py -v`

**Step 5: Commit**

```bash
git add src/pipecat/services/openai/__init__.py
git commit -m "feat: export OpenAI WebSocket LLM service from openai package"
```

---

### Task 7: Final Review and PR Preparation

**Step 1: Run full test suite to check for regressions**

Run: `uv run pytest -x --timeout=60`

**Step 2: Run linter on all new files**

Run: `uv run ruff check && uv run ruff format --check`

**Step 3: Review all changes**

Run: `git diff main...HEAD --stat` and `git log --oneline main..HEAD`

**Step 4: Create PR**

Push branch and create PR targeting the upstream pipecat repository. PR should include:
- Summary of what the service does
- How to use it (code example)
- What was tested
- Note about local service compatibility
