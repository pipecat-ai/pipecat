# OpenTelemetry Tracing for Pipecat

This demo showcases OpenTelemetry tracing integration for Pipecat services, allowing you to visualize service calls, performance metrics, and dependencies in a Jaeger dashboard.

## Features

- **Hierarchical Tracing**: Track entire conversations, turns, and service calls
- **Service Tracing**: Detailed spans for TTS, STT, and LLM services with rich context
- **TTFB Metrics**: Capture Time To First Byte metrics for latency analysis
- **Usage Statistics**: Track character counts for TTS and token usage for LLMs
- **Flexible Exporters**: Use Jaeger, Zipkin, or any OpenTelemetry-compatible backend

## Trace Structure

Traces are organized hierarchically:

```
Conversation (conversation-uuid)
├── Turn 1
│   ├── deepgram_transcription
│   ├── process_context
│   ├── get_chat_completion
│   └── cartesia_tts
└── Turn 2
    ├── deepgram_transcription
    ├── process_context
    ├── get_chat_completion
    └── cartesia_tts
```

This organization helps you track conversation-to-conversation and turn-to-turn.

## Setup Instructions

### 1. Start the Jaeger Container

Run Jaeger in Docker to collect and visualize traces:

```bash
docker run -d --name jaeger \
  -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

### 2. Environment Configuration

Create a `.env` file with your API keys and enable tracing:

```
ENABLE_TRACING=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317  # Point to your preferred backend
# OTEL_CONSOLE_EXPORT=true  # Set to any value for debug output to console

# Service API keys
DEEPGRAM_API_KEY=your_key_here
CARTESIA_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### 3. Configure Your Pipeline Task

Enable tracing in your Pipecat application:

```python
# Initialize OpenTelemetry with your chosen exporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",  # Jaeger OTLP endpoint
    insecure=True,
)

setup_tracing(
    service_name="pipecat-demo",
    exporter=exporter,
    console_export=os.getenv("OTEL_CONSOLE_EXPORT", "false").lower() == "true",
)

# Enable tracing in your PipelineTask
task = PipelineTask(
    pipeline,
    params=PipelineParams(
        allow_interruptions=True,
        enable_metrics=True,  # Required for some service metrics
    ),
    enable_tracing=True,  # Enables both turn and conversation tracing
    conversation_id="customer-123",  # Optional - will auto-generate if not provided
)
```

### 4. Exporter Options

While this demo uses Jaeger, you can configure any OpenTelemetry-compatible exporter:

#### Jaeger (Default for the demo)

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",  # Jaeger OTLP endpoint
    insecure=True,
)
```

#### Cloud Providers

Many cloud providers offer OpenTelemetry-compatible observability services:

- AWS X-Ray
- Google Cloud Trace
- Azure Monitor
- Datadog APM

See the OpenTelemetry documentation for specific exporter configurations:
https://opentelemetry.io/ecosystem/vendors/

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Demo

```bash
python bot.py
```

### 7. View Traces in Jaeger

Open your browser to [http://localhost:16686](http://localhost:16686) and select the "pipecat-demo" service to view traces.

## Understanding the Traces

- **Conversation Spans**: The top-level span representing an entire conversation
- **Turn Spans**: Child spans of conversations that represent each turn in the dialog
- **Service Spans**: Detailed service operations nested under turns
- **Service Attributes**: Each service includes rich context about its operation:
  - **TTS**: Voice ID, character count, service type
  - **STT**: Transcription text, language, model
  - **LLM**: Messages, tokens used, model, service configuration
- **Metrics**: Performance data like `metrics.ttfb_ms` and processing durations

## How It Works

The tracing system consists of:

1. **TurnTrackingObserver**: Detects conversation turns
2. **TurnTraceObserver**: Creates spans for turns and conversations
3. **Service Decorators**: `@traced_tts`, `@traced_stt`, `@traced_llm` for service-specific tracing
4. **Context Providers**: Share context between different parts of the pipeline

## Troubleshooting

- **No Traces in Jaeger**: Ensure the Docker container is running and the OTLP endpoint is correct
- **Debugging Traces**: Set `OTEL_CONSOLE_EXPORT=true` to print traces to the console for debugging
- **Missing Metrics**: Check that `enable_metrics=True` in PipelineParams
- **Connection Errors**: Verify network connectivity to the Jaeger container
- **Exporter Issues**: Try the Console exporter (`OTEL_CONSOLE_EXPORT=true`) to verify tracing works
- **Other Backends**: If using a different backend, ensure you've configured the correct exporter and endpoint

## References

- [OpenTelemetry Python Documentation](https://opentelemetry-python.readthedocs.io/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/latest/)
