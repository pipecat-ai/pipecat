# OpenTelemetry Tracing with Pipecat

This repository demonstrates OpenTelemetry tracing integration for Pipecat services, with examples for different backends.

## Tracing Features in Pipecat

- **Hierarchical Tracing**: Track entire conversations, turns, and service calls
- **Service Tracing**: Detailed spans for TTS, STT, and LLM services with rich context
- **TTFB Metrics**: Capture Time To First Byte metrics for latency analysis
- **Usage Statistics**: Track character counts for TTS and token usage for LLMs

## Trace Structure

Traces are organized hierarchically:

```
Conversation (conversation)
├── turn
│   ├── stt_deepgramsttservice
│   ├── llm_openaillmservice
│   └── tts_cartesiattsservice
└── turn
    ├── stt_deepgramsttservice
    ├── llm_openaillmservice
    └── tts_cartesiattsservice
    turn
    └── ...
```

This organization helps you track conversation-to-conversation and turn-to-turn interactions.

## Available Demos

| Demo                            | Description                                                               |
| ------------------------------- | ------------------------------------------------------------------------- |
| [Jaeger Tracing](./jaeger/)     | Tracing with Jaeger, an open-source end-to-end distributed tracing system |
| [Langfuse Tracing](./langfuse/) | Tracing with Langfuse, a specialized platform for LLM observability       |

## Common Requirements

- Python 3.10+
- Pipecat and its dependencies
- API keys for the services used (Deepgram, Cartesia, OpenAI)
- The appropriate OpenTelemetry exporters

## How Tracing Works

The tracing system consists of:

1. **TurnTrackingObserver**: Detects conversation turns
2. **TurnTraceObserver**: Creates spans for turns and conversations
3. **Service Decorators**: `@traced_tts`, `@traced_stt`, `@traced_llm` for service-specific tracing
4. **Context Providers**: Share context between different parts of the pipeline

## Getting Started

1. Choose one of the demos from the table above
2. Follow the README instructions in the respective directory

## Common Troubleshooting

- **Debugging Traces**: Set `OTEL_CONSOLE_EXPORT=true` to print traces to the console for debugging
- **Missing Metrics**: Check that `enable_metrics=True` in PipelineParams
- **API Key Issues**: Verify your API keys are set correctly in the .env file

## References

- [OpenTelemetry Python Documentation](https://opentelemetry-python.readthedocs.io/)
- [Pipecat Documentation](https://docs.pipecat.ai/server/utilities/opentelemetry)
