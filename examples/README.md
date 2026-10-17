# Pipecat Examples

This directory contains examples showing how to build voice and multimodal agents with Pipecat.

## Setup

1. Follow the [README](https://github.com/pipecat-ai/pipecat/blob/main/README.md#%EF%B8%8F-developing-pipecat) steps to get your local environment configured.

   > **Run from root directory**: Make sure you are running the steps from the root directory.

   > **Using local audio?**: The `LocalAudioTransport` requires a system dependency for `portaudio`. Install the dependency to use the transport.

2. Copy the [`env.example`](../env.example) file and add API keys for services you plan to use:

   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

3. Run any example:

   ```bash
   uv run python getting-started/01-say-one-thing.py
   ```

4. Open the web interface at http://localhost:7860/client/ and click "Connect"

## OpenAI-compatible backends

[`voice/voice-openai-compatible.py`](voice/voice-openai-compatible.py) uses the
standard OpenAI services with independently configurable endpoints, models, and
credentials. No provider-specific Pipecat service or SDK is needed.

The backends must implement these protocols, not just chat completions:

| Stage | Configuration | Required protocol |
| --- | --- | --- |
| STT | `STT_BASE_URL`, `STT_MODEL` | Full `ws://` or `wss://` URL to `/v1/realtime`; transcription sessions with 24 kHz mono PCM16, transcript delta/completed events, and explicit audio-buffer commits |
| LLM | `LLM_BASE_URL`, `LLM_MODEL` | HTTP base URL ending in `/v1`; streaming `/chat/completions` |
| TTS | `TTS_BASE_URL`, `TTS_MODEL`, `TTS_VOICE` | HTTP base URL ending in `/v1`; `/audio/speech` with `response_format="pcm"`, returning raw 24 kHz mono PCM16 chunks |

Set `STT_API_KEY`, `LLM_API_KEY`, and `TTS_API_KEY` when authentication is required.
Each defaults to `not-needed` for unauthenticated local servers. The example uses
local Silero VAD and disables server-side turn detection. It does not use the
multipart `/audio/transcriptions` endpoint or a speech-to-speech realtime model.

For example, after deploying the [Dynamo Nemotron speech pipeline](https://github.com/ai-dynamo/dynamo/pull/12567)
(currently proposed), forward its frontend to port 8000 and configure:

```bash
export STT_BASE_URL=ws://localhost:8000/v1/realtime
export STT_MODEL=nemotron-asr-streaming
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_MODEL=nvidia/nemotron-3-nano
export TTS_BASE_URL=http://localhost:8000/v1
export TTS_MODEL=nvidia/magpie-tts-multilingual
export TTS_VOICE=Magpie-Multilingual.EN-US.Aria

# From the Pipecat repository root:
uv sync --extra runner --extra webrtc
uv run --no-sync python examples/voice/voice-openai-compatible.py -t webrtc
```

Open http://localhost:7860/client/ and connect. The service URLs are accessed
by the Pipecat server, not the browser. They may point to one frontend or separate
providers. To carry browser audio through an SSH TCP tunnel, use `-t websocket`
instead and select the WebSocket transport in the client.

## Running examples with other transports

Most examples support running with other transports, like Twilio or Daily.

### Daily

You need to create a Daily account at https://dashboard.daily.co/u/signup. Once signed up, you can create your own room from the dashboard and set the environment variables `DAILY_ROOM_URL` and `DAILY_API_KEY`. Alternatively, you can let the example create a room for you (still needs `DAILY_API_KEY` environment variable). Then, start any example with `-t daily`:

```bash
uv run getting-started/06-voice-agent.py -t daily
```

### Twilio

It is also possible to run the example through a Twilio phone number. You will need to setup a few things:

1. Install and run [ngrok](https://ngrok.com/download).

```bash
ngrok http 7860
```

2. Configure your Twilio phone number. One way is to setup a TwiML app and set the request URL to the ngrok URL from step (1). Then, set your phone number to use the new TwiML app.

Then, run the example with:

```bash
uv run getting-started/06-voice-agent.py -t twilio -x NGROK_HOST_NAME
```

## Directory Structure

### [`getting-started/`](./getting-started/)

Progressive introduction to Pipecat, from minimal TTS to a full voice agent with function calling.

### [`flows/`](./flows/)

Structured conversations with [Pipecat Flows](../src/pipecat/flows): predefined and dynamic conversation paths with state management, across multiple LLM providers.

### [`voice/`](./voice/)

Full STT + LLM + TTS voice agent pipelines showcasing different speech service providers (Deepgram, ElevenLabs, Cartesia, etc.)

### [`function-calling/`](./function-calling/)

Function calling with different LLM providers (OpenAI, Anthropic, Google, etc.)

### [`transcription/`](./transcription/)

Speech-to-text examples with various STT providers.

### [`vision/`](./vision/)

Image description and vision capabilities with different multimodal LLMs.

### [`image-generation/`](./image-generation/)

Generating an image from a text prompt with different image generation services (fal, Google, OpenAI).

### [`realtime/`](./realtime/)

Realtime and multimodal live APIs (OpenAI Realtime, Gemini Live, AWS Nova Sonic, Ultravox, Grok).

### [`persistent-context/`](./persistent-context/)

Maintaining conversation context across sessions with different providers.

### [`context-summarization/`](./context-summarization/)

Summarizing conversation context to manage token limits.

### [`update-settings/`](./update-settings/)

Changing service settings at runtime, organized by service type:

- **[`stt/`](./update-settings/stt/)** — Speech-to-text settings
- **[`tts/`](./update-settings/tts/)** — Text-to-speech settings
- **[`llm/`](./update-settings/llm/)** — LLM settings

### [`turn-management/`](./turn-management/)

Turn detection, interruption handling, and user input management.

### [`thinking/`](./thinking/)

LLM thinking/reasoning modes.

### [`mcp/`](./mcp/)

MCP (Model Context Protocol) tool server integration.

### [`transports/`](./transports/)

Transport layer examples (WebRTC, Daily, LiveKit).

### [`video-avatar/`](./video-avatar/)

Video avatar integrations (Tavus, HeyGen, Simli, LemonSlice).

### [`video-processing/`](./video-processing/)

Video processing, mirroring, GStreamer, and custom video tracks.

### [`audio/`](./audio/)

Audio recording, background sounds, and sound effects.

### [`observability/`](./observability/)

Pipeline monitoring: observers, heartbeats, and Sentry metrics.

### [`rag/`](./rag/)

Retrieval-augmented generation, grounding, and long-term memory (Mem0, Gemini).

### [`features/`](./features/)

Miscellaneous features: wake phrases, live translation, service switching, voice switching, DTMF keypad menus, and more.

## Advanced Usage

### Customizing Network Settings

```bash
uv run python <example-name> --host 0.0.0.0 --port 8080
```

### Troubleshooting

- **No audio/video**: Check browser permissions for microphone and camera
- **Connection errors**: Verify API keys in `.env` file
- **Port conflicts**: Use `--port` to change the port

For more examples, visit the [pipecat-examples repository](https://github.com/pipecat-ai/pipecat-examples).
