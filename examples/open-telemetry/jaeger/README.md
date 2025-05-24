# Jaeger Tracing for Pipecat

This demo showcases OpenTelemetry tracing integration for Pipecat services using Jaeger, allowing you to visualize service calls, performance metrics, and dependencies.

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
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317  # Point to your Jaeger backend
# OTEL_CONSOLE_EXPORT=true  # Set to any value for debug output to console

# Service API keys
DEEPGRAM_API_KEY=your_key_here
CARTESIA_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> Install only the grpc exporter. If you have a conflict, uninstall the http exporter.

### 4. Run the Demo

```bash
python bot.py
```

### 5. View Traces in Jaeger

Open your browser to [http://localhost:16686](http://localhost:16686) and select the "pipecat-demo" service to view traces.

## Jaeger-Specific Configuration

In the `bot.py` file, note the GRPC exporter configuration:

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Create the exporter
otlp_exporter = OTLPSpanExporter(
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
    insecure=True,
)

# Set up tracing with the exporter
setup_tracing(
    service_name="pipecat-demo",
    exporter=otlp_exporter,
    console_export=bool(os.getenv("OTEL_CONSOLE_EXPORT")),
)
```

## Troubleshooting

- **No Traces in Jaeger**: Ensure the Docker container is running and the OTLP endpoint is correct
- **Connection Errors**: Verify network connectivity to the Jaeger container
- **Exporter Issues**: Try the Console exporter (`OTEL_CONSOLE_EXPORT=true`) to verify tracing works

## References

- [Jaeger Documentation](https://www.jaegertracing.io/docs/latest/)
