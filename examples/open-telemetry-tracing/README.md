# OpenTelemetry Tracing for Pipecat

This demo showcases OpenTelemetry tracing integration for Pipecat services, allowing you to visualize service calls, performance metrics, and dependencies in a Jaeger dashboard.

## Features

- **Service Tracing**: Track requests through TTS, STT, and LLM services
- **TTFB Metrics**: Capture Time To First Byte metrics for latency analysis
- **Usage Statistics**: Track character counts for TTS and token usage for LLMs
- **Flexible Exporters**: Use Jaeger, Zipkin, or any OpenTelemetry-compatible backend

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
OTEL_CONSOLE_EXPORT=false  # Set to true for debug output to console

# Service API keys
DEEPGRAM_API_KEY=your_key_here
CARTESIA_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### 3. Exporter Options

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

See the OpenTelemetry documentation for specific exporter configurations.

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Demo

```bash
python bot.py
```

### 6. View Traces in Jaeger

Open your browser to [http://localhost:16686](http://localhost:16686) and select the "pipecat-demo" service to view traces.

## Understanding the Traces

- **Service Spans**: Each service operation creates a span in the trace
- **TTS Metrics**: Look for `metrics.ttfb_ms` and `metrics.tts.character_count` attributes
- **Service Settings**: Each service's configuration is captured in the trace

## How It Works

The tracing system consists of:

1. **Traceable Base Class**: Provides basic tracing capabilities
2. **TraceMetricsCollector**: Captures and formats metrics for spans
3. **Decorators**: `@traceable` for classes, `@traced` for methods
4. **Context Manager**: `traced_operation` for easy metrics collection

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
