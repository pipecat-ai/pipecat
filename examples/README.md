# Pipecat Examples

This directory contains examples showing how to build voice and multimodal agents with Pipecat.

## Setup

1. Follow the [README](https://github.com/pipecat-ai/pipecat/blob/main/README.md#%EF%B8%8F-contributing-to-the-framework) steps to get your local environment configured.

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

### [`voice/`](./voice/)

Full STT + LLM + TTS voice agent pipelines showcasing different speech service providers (Deepgram, ElevenLabs, Cartesia, etc.)

### [`function-calling/`](./function-calling/)

Function calling with different LLM providers (OpenAI, Anthropic, Google, etc.)

### [`transcription/`](./transcription/)

Speech-to-text examples with various STT providers.

### [`vision/`](./vision/)

Image description and vision capabilities with different multimodal LLMs.

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

### [`thinking-and-mcp/`](./thinking-and-mcp/)

LLM thinking/reasoning modes and MCP (Model Context Protocol) tool server integration.

### [`transports/`](./transports/)

Transport layer examples (WebRTC, Daily, LiveKit).

### [`video-avatar/`](./video-avatar/)

Video avatar integrations (Tavus, HeyGen, Simli, LemonSlice, Ojin).

### [`video-processing/`](./video-processing/)

Video processing, mirroring, GStreamer, and custom video tracks.

### [`audio/`](./audio/)

Audio recording, background sounds, and sound effects.

### [`observability/`](./observability/)

Pipeline monitoring: observers, heartbeats, and Sentry metrics.

### [`rag/`](./rag/)

Retrieval-augmented generation, grounding, and long-term memory (Mem0, Gemini).

### [`features/`](./features/)

Miscellaneous features: wake phrases, live translation, service switching, voice switching, and more.

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
