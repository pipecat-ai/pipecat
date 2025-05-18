# Langfuse Tracing for Pipecat via OpenTelemetry

This demo showcases [Langfuse](https://langfuse.com) tracing integration for Pipecat services via OpenTelemetry, allowing you to visualize service calls, performance metrics, and dependencies.

This is a fork of the [OpenTelemetry Tracing for Pipecat](../open-telemetry-tracing) demo, but uses Langfuse instead of Jaeger. In contrast to the original demo, this demo uses the `opentelemetry-exporter-otlp-proto-http` exporter as the `grpc` exporter is not supported by Langfuse.

Pipecat trace in Langfuse:

https://github.com/user-attachments/assets/13dd7431-bf5e-42e3-8d6d-2ed84c51195d

## Features

- **Hierarchical Tracing**: Track entire conversations, turns, and service calls
- **Service Tracing**: Detailed spans for TTS, STT, and LLM services with rich context
- **TTFB Metrics**: Capture Time To First Byte metrics for latency analysis
- **Usage Statistics**: Track character counts for TTS and token usage for LLMs

## Trace Structure

Traces are organized hierarchically:

```
Conversation (conversation-uuid)
├── turn-1
│   ├── stt_deepgramsttservice
│   ├── llm_openaillmservice
│   └── tts_cartesiattsservice
└── turn-2
    ├── stt_deepgramsttservice
    ├── llm_openaillmservice
    └── tts_cartesiattsservice
    turn-N
    └── ...
```

This organization helps you track conversation-to-conversation and turn-to-turn.

## Setup Instructions

### 1. Create a Langfuse Project and get API keys

[Self-host](https://langfuse.com/self-hosting) Langfuse or create a free [Langfuse Cloud](https://cloud.langfuse.com) account.
Create a new project and get the API keys.

### 2. Environment Configuration

Base64 encode your Langfuse public and secret key:

```bash
echo -n "pk-lf-1234567890:sk-lf-1234567890" | base64
```

Create a `.env` file with your API keys to enable tracing:

```
ENABLE_TRACING=true
# OTLP endpoint (defaults to localhost:4317 if not set)
OTEL_EXPORTER_OTLP_ENDPOINT=http://cloud.langfuse.com/api/public/otel
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20<base64_encoded_api_key>
# Set to any value to enable console output for debugging
# OTEL_CONSOLE_EXPORT=true
```

### 3. Configure Your Pipeline Task

Enable tracing in your Pipecat application:

```python
# Initialize OpenTelemetry with your chosen exporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Configured automatically from .env
exporter = OTLPSpanExporter()

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

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Demo

```bash
python bot.py
```

### 6. View Traces in Langfuse

Open your browser to [https://cloud.langfuse.com](https://cloud.langfuse.com) to view traces.

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

- **No Traces in Langfuse**: Ensure that your credentials are correct and follow this [troubleshooting guide](https://langfuse.com/faq/all/missing-traces)
- **Debugging Traces**: Set `OTEL_CONSOLE_EXPORT=true` to print traces to the console for debugging
- **Missing Metrics**: Check that `enable_metrics=True` in PipelineParams
- **Connection Errors**: Verify network connectivity to the Jaeger container
- **Exporter Issues**: Try the Console exporter (`OTEL_CONSOLE_EXPORT=true`) to verify tracing works

## References

- [OpenTelemetry Python Documentation](https://opentelemetry-python.readthedocs.io/)
- [Langfuse OpenTelemetry Documentation](https://langfuse.com/docs/opentelemetry/get-started)
