# Murf AI Text-to-Speech Service

This module provides integration with [Murf AI](https://murf.ai/api)'s WebSocket-based text-to-speech API for real-time speech synthesis.

## Features

- Real-time WebSocket streaming
- Voice customization (voice_id, style, rate, pitch)
- Pronunciation dictionaries
- Variation control for generated audios
- Multiple languages and locale support
- Automatic reconnection on errors

## Installation

```bash
pip install pipecat-ai[murf]
```

## Configuration

You need a Murf AI API key to use this service. Create a `.env` file in your project directory and add:

```
MURF_API_KEY=
```

## Basic Usage

```python
from pipecat.services.murf.tts import MurfTTSService

# Basic setup
tts = MurfTTSService(
    api_key="your_murf_api_key",
    params=MurfTTSService.InputParams(
        voice_id="en-UK-ruby",
        style="Conversational",
        rate=0,
        pitch=0,
        variation=1,
    ),
)

# Use in a pipeline
pipeline = Pipeline([
    # ... other processors
    tts,
    # ... other processors
])
```

## Advanced Configuration

```python
from pipecat.services.murf.tts import MurfTTSService

# Advanced setup with all options
tts = MurfTTSService(
    api_key="your_murf_api_key",
    url="wss://api.murf.ai/v1/speech/stream-input",  # Optional: custom endpoint
    sample_rate=24000,  # Optional: audio sample rate
    aggregate_sentences=True,  # Optional: aggregate sentences before synthesis
    params=MurfTTSService.InputParams(
        voice_id="en-UK-ruby",
        style="Conversational",
        rate=0,  # Speech rate adjustment
        pitch=0,  # Pitch adjustment
        variation=3,  # Variation in pause, pitch, and speed (0-5)
        pronunciation_dictionary={
            "live": {"type": "IPA", "pronunciation": "laɪv"}
        },
    ),
)
```

## Voice Configuration Parameters

- **voice_id**: Voice identifier (default: "en-UK-ruby")
- **style**: Speech style (default: "Conversational")
- **rate**: Speech rate adjustment (integer, default: 0)
- **pitch**: Pitch adjustment (integer, default: 0)
- **variation**: Variation in pause, pitch, and speed (0-5, default: 1)
- **pronunciation_dictionary**: Custom pronunciation for specific words
- **multi_native_locale**: language locale code like `en-US`, `en-UK`, `es-ES`

## Error Handling

The service includes automatic error handling and reconnection:

```python
@tts.event_handler("on_connection_error")
async def on_connection_error(tts, error):
    print(f"TTS connection error: {error}")
```

## Example

See `examples/murf-tts-interruptible/bot.py` for a complete example that demonstrates:

- Real-time voice conversation with Murf TTS
- Deepgram STT integration
- OpenAI LLM integration
- Local audio device input/output
- Proper interruption handling

## Requirements

- Python 3.8+
- `websockets` library
- Valid Murf AI API key
- Internet connection for API access

## Context Management

The service uses Murf's context-based system for efficient synthesis:

- **Context IDs**: Each synthesis request gets a unique context ID
- **End Marker**: Each message is sent with `end: True` to indicate completion
- **Interruption Handling**: On interruption, the current context is cleared using Murf's clear context API
- **No Reconnection**: Interruptions don't require WebSocket reconnection - just context clearing

## Notes

- The service uses WebSocket streaming for real-time synthesis
- Context IDs are automatically managed for each synthesis request
- Each text input is sent as a single message with `end: True`
- The service supports interruption handling via context clearing (no reconnection needed)
- Audio is streamed as PCM format at the specified sample rate
