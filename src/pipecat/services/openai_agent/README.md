# OpenAI Agents SDK Integration

This service integrates the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) with Pipecat, enabling powerful agentic workflows with features like:

- **Agent loops** with tool calling and response streaming
- **Handoffs** between specialized agents  
- **Guardrails** for input/output validation
- **Sessions** with automatic conversation history
- **Built-in tracing** and monitoring

## Installation

Install the OpenAI Agents SDK dependency:

```bash
pip install "pipecat-ai[openai-agent]"
# or
uv add "pipecat-ai[openai-agent]"
```

## Basic Usage

```python
from pipecat.services.openai_agent import OpenAIAgentService

# Create a simple agent
agent_service = OpenAIAgentService(
    name="Assistant",
    instructions="You are a helpful assistant.",
    api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True,
)

# Use in a pipeline
pipeline = Pipeline([
    transport.input(),
    stt,
    agent_service,
    tts,
    transport.output(),
])
```

## Features

### Tool Integration

```python
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: sunny, 22°C"

agent_service = OpenAIAgentService(
    name="Weather Assistant",
    instructions="Help users with weather information.",
    tools=[get_weather],
    api_key=os.getenv("OPENAI_API_KEY"),
)
```

### Agent Handoffs

```python
# Create specialized agents
weather_agent = OpenAIAgentService(
    name="Weather Specialist",
    instructions="Provide weather information and forecasts.",
    tools=[get_weather, get_forecast],
)

trivia_agent = OpenAIAgentService(
    name="Trivia Master", 
    instructions="Share interesting facts and trivia.",
    tools=[get_random_fact],
)

# Create coordinator that can hand off to specialists
coordinator = OpenAIAgentService(
    name="Coordinator",
    instructions="Route users to the right specialist.",
    handoffs=[weather_agent.agent, trivia_agent.agent],
)
```

### Guardrails

```python
from agents import InputGuardrail, GuardrailFunctionOutput

async def content_filter(ctx, agent, input_data):
    # Check input for appropriate content
    if is_inappropriate(input_data):
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info="Content not allowed"
        )
    return GuardrailFunctionOutput(tripwire_triggered=False)

agent_service = OpenAIAgentService(
    name="Safe Assistant",
    instructions="You are a helpful and safe assistant.",
    input_guardrails=[InputGuardrail(guardrail_function=content_filter)],
)
```

### Session Management

```python
agent_service = OpenAIAgentService(
    name="Personal Assistant",
    instructions="Remember user preferences and context.",
    session_config={
        "user_id": "user_123",
        "memory_enabled": True,
    }
)

# Update session context dynamically
agent_service.update_session_context({
    "user_preferences": {"language": "en", "style": "formal"}
})
```

## Configuration Options

### Basic Parameters

- `name`: Agent identifier for handoffs and tracing
- `instructions`: System prompt defining agent behavior  
- `api_key`: OpenAI API key (or use `OPENAI_API_KEY` env var)
- `streaming`: Enable real-time token streaming (default: True)

### Advanced Configuration

- `tools`: List of callable functions for the agent to use
- `handoffs`: List of other agents this agent can transfer to
- `input_guardrails`: Input validation and filtering
- `output_guardrails`: Output validation and filtering  
- `model_config`: Model settings (model, temperature, etc.)
- `session_config`: Session and memory configuration

### Model Configuration

```python
agent_service = OpenAIAgentService(
    name="Precise Assistant",
    instructions="Provide accurate, concise responses.",
    model_config={
        "model": "gpt-4o",
        "temperature": 0.1,
        "max_tokens": 150,
    }
)
```

## Examples

See the foundational examples:

- [`45-openai-agent-basic.py`](../examples/foundational/45-openai-agent-basic.py) - Basic agent with tools
- [`46-openai-agent-handoffs.py`](../examples/foundational/46-openai-agent-handoffs.py) - Multi-agent system with handoffs

## Methods

### Core Methods

- `update_agent_config()` - Update instructions and model settings
- `add_tool()` - Add new tools dynamically
- `add_handoff_agent()` - Add handoff destinations
- `get_session_context()` - Get current session state
- `update_session_context()` - Update session variables

### Lifecycle Methods

Inherited from `AIService`:
- `start()` - Initialize the agent
- `stop()` - Clean up resources
- `cancel()` - Cancel ongoing operations

## Integration with Pipecat

The service processes `TextFrame` inputs and generates:
- `LLMFullResponseStartFrame` - Response beginning
- `LLMTextFrame` - Streaming text tokens (if streaming enabled)
- `LLMFullResponseEndFrame` - Response completion

This integrates seamlessly with Pipecat's conversation pipeline and context aggregators.

## Error Handling

The service includes robust error handling for:
- Missing API keys or SDK installation
- Agent processing failures  
- Network connectivity issues
- Malformed tool responses

Errors are emitted as `ErrorFrame` objects in the pipeline.

## Requirements

- OpenAI API key
- `openai-agents` package
- Python 3.10+

## Limitations

- Currently supports OpenAI models only (via Agents SDK)
- Handoffs work within individual requests (no cross-request state)
- Real-time voice features require additional setup