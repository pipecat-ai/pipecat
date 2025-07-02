# Langfuse Tracing for Pipecat

This demo showcases [Langfuse](https://langfuse.com) tracing integration for Pipecat services via OpenTelemetry, allowing you to visualize service calls, performance metrics, and dependencies with a focus on LLM observability.

Pipecat trace in Langfuse:

https://github.com/user-attachments/assets/13dd7431-bf5e-42e3-8d6d-2ed84c51195d

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
# OTLP endpoint for Langfuse
OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20<base64_encoded_api_key>
# Set to any value to enable console output for debugging
# OTEL_CONSOLE_EXPORT=true

# Service API keys
DEEPGRAM_API_KEY=your_key_here
CARTESIA_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> Install only the http exporter. If you have a conflict, uninstall the grpc exporter.

### 4. Run the Demo

```bash
python bot.py
```

### 5. View Traces in Langfuse

Open your browser to [https://cloud.langfuse.com](https://cloud.langfuse.com) to view traces.

## Langfuse-Specific Configuration

In the `bot.py` file, note the HTTP exporter configuration:

```python
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Create the exporter - configured from environment variables
otlp_exporter = OTLPSpanExporter()

# Set up tracing with the exporter
setup_tracing(
    service_name="pipecat-demo",
    exporter=otlp_exporter,
    console_export=bool(os.getenv("OTEL_CONSOLE_EXPORT")),
)
```

## Troubleshooting

- **No Traces in Langfuse**: Ensure that your credentials are correct and follow this [troubleshooting guide](https://langfuse.com/faq/all/missing-traces)
- **Connection Errors**: Verify network connectivity to Langfuse
- **Authorization Issues**: Check that your base64 encoding is correct and the API keys are valid

## References

- [Langfuse OpenTelemetry Documentation](https://langfuse.com/docs/opentelemetry/get-started)
