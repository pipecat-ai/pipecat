# OpenAI WebSocket LLM Service Design

## Overview

Add a new LLM service that connects to OpenAI's Responses API via WebSocket (`wss://api.openai.com/v1/responses`). This provides persistent-connection, lower-latency text streaming and function calling compared to the existing HTTP-based `OpenAILLMService`.

The service supports both OpenAI cloud and local/self-hosted services that implement the same WebSocket Responses API protocol.

## Architecture

### Class Hierarchy

```
LLMService + WebsocketService
    └── OpenAIWebSocketLLMService
```

- **`LLMService`** — frame processing, function calling, context aggregation, turn management
- **`WebsocketService`** — WebSocket connection management with auto-reconnect and exponential backoff

### New Files

| File | Purpose |
|------|---------|
| `src/pipecat/services/openai/websocket_llm.py` | Main service class |
| `src/pipecat/adapters/services/open_ai_websocket_adapter.py` | LLMContext → Responses API adapter |
| `tests/test_openai_websocket_llm.py` | Unit tests |

### Modified Files

| File | Change |
|------|--------|
| `src/pipecat/services/openai/__init__.py` | Export new service |

## Connection

**Endpoint:** Configurable `base_url`, default `wss://api.openai.com/v1/responses`

**Authentication:** `Authorization: Bearer {api_key}` header. `api_key` is optional for local services.

**Connection limit:** OpenAI enforces a 60-minute limit. The service auto-reconnects via `WebsocketService`.

## Message Flow

### Sending (request)

1. Pipeline pushes `LLMContextFrame` with universal `LLMContext`
2. `OpenAIWebSocketLLMAdapter` converts context → Responses API `input` array
3. Service sends `response.create` event over WebSocket:
   ```json
   {
     "type": "response.create",
     "model": "gpt-5.2",
     "input": [...],
     "tools": [...],
     "previous_response_id": "resp_xxx",
     "store": false
   }
   ```

### Receiving (streaming response)

Reuses event models from `pipecat.services.openai.realtime.events`:

| Server Event | Pipecat Frame |
|-------------|---------------|
| `response.created` | `LLMFullResponseStartFrame` |
| `response.output_text.delta` | `LLMTextFrame(delta)` |
| `response.function_call_arguments.delta` | Accumulate arguments |
| `response.function_call_arguments.done` | `run_function_calls()` |
| `response.done` | `LLMFullResponseEndFrame`, store `previous_response_id` |
| `error` | `push_error()` |

### Context Strategy with `previous_response_id`

- **First request:** Send full context as `input`
- **Subsequent requests:** Send only new messages + `previous_response_id`
- **On reconnect or `previous_response_not_found`:** Fall back to full context
- **Configurable:** `use_previous_response_id` flag (default `True`)

## Configuration

```python
class OpenAIWebSocketLLMService(LLMService, WebsocketService):
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
```

### InputParams (Pydantic BaseModel)

- `temperature: Optional[float]`
- `max_tokens: Optional[int]`
- `top_p: Optional[float]`
- `frequency_penalty: Optional[float]`
- `presence_penalty: Optional[float]`

## Frame Handling

| Input Frame | Action |
|-------------|--------|
| `LLMContextFrame` | Convert context, send `response.create` |
| `LLMUpdateSettingsFrame` | Update local settings |
| `LLMSetToolsFrame` | Update tools list |
| `InterruptionFrame` | Cancel in-flight response |
| `StartFrame` | Connect WebSocket |
| `EndFrame` / `CancelFrame` | Disconnect WebSocket |

## Error Handling

- **WebSocket errors:** Auto-reconnect via `WebsocketService` with exponential backoff
- **`previous_response_not_found`:** Fall back to full context send
- **`websocket_connection_limit_reached`:** Auto-reconnect
- **API errors:** `push_error()` upstream (non-fatal by default)
- **Parse errors:** Log warning, skip event

## Local Service Compatibility

- Configurable `base_url` (supports `ws://` for local services)
- Optional `api_key`
- `previous_response_id` gracefully degrades if unsupported
- No hardcoded OpenAI-specific assumptions

## Testing

- Unit tests using `run_test()` from `pipecat/tests/utils.py`
- Mock WebSocket connection
- Test scenarios: text streaming, function calling, reconnection, error handling
- Separate adapter tests for context conversion
