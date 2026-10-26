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

## Dynamo with incremental text prefill

[`voice/voice-dynamo.py`](voice/voice-dynamo.py) connects Dynamo's realtime STT,
text LLM, and streaming TTS endpoints. STT interim deltas are sent to the LLM
while the user speaks. The final Pipecat context remains authoritative, including
transcript corrections; response generation begins only when the user turn ends.

Use Dynamo [**1.6.0**](https://github.com/ai-dynamo/dynamo/releases/tag/v1.6.0)
for both the frontend and vLLM runtime. This example requires its **experimental text-input
Realtime API**, not just an OpenAI-compatible chat-completions endpoint. The LLM backend
must support `input_text.append`, `input_text.clear`, and `input_text.commit`,
alongside `conversation.item.create`, `response.create`, and streamed
`response.output_text.delta` events. Engine-side incremental prefill must also be
enabled for the latency optimization to take effect. Start the Dynamo vLLM
worker with `--realtime --enable-prefix-caching`; an ordinary text-generation
worker does not register this capability. These input events are Dynamo
extensions, not part of the OpenAI standard, and remain experimental.

The release images are `nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.6.0` and
`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0`.
Build the speech adapters from the
[matching release's example](https://github.com/ai-dynamo/dynamo/tree/v1.6.0/examples/nemotron_speech_cascaded_pipeline)
using the same frontend base image. ASR/TTS NIMs retain their own model-specific
image versions and access requirements.

The STT endpoint must support transcription sessions with 24 kHz PCM input,
client-side audio-buffer commits, and append-only transcription deltas. The TTS
endpoint must return mono 16-bit PCM at 24 kHz for `response_format="pcm"`.
Pipecat talks only to these endpoints; no Riva client dependency is needed.

From the repository root, configure the model and voice names registered by your
Dynamo deployment:

```bash
uv sync --group dev --extra runner --extra websocket
export DYNAMO_STT_MODEL="your-stt-model"
export DYNAMO_LLM_MODEL="your-text-model"
export DYNAMO_TTS_MODEL="your-tts-model"
export DYNAMO_TTS_VOICE="your-voice"
uv run python examples/voice/voice-dynamo.py -t websocket
```

Open `http://localhost:7860/client/`. By default, all services use the Dynamo
frontend on `localhost:8000`. Override `DYNAMO_STT_BASE_URL` (default
`ws://localhost:8000/v1/realtime`), `DYNAMO_LLM_BASE_URL`, and
`DYNAMO_TTS_BASE_URL` (both default `http://localhost:8000/v1`) for another
deployment. Set `DYNAMO_API_KEY` when the endpoints require authentication; use
`wss://` and `https://` for remote authenticated endpoints.

`DynamoStreamingPrefillProcessor` belongs between streaming STT and the user context
aggregator. It expects **append-only deltas**, as emitted by
`OpenAIRealtimeSTTService`, not cumulative interim hypotheses. The text LLM
service supports plain-text conversation messages, not tools or multimodal
input. Keep the same `LLMContext` in the prefill processor and aggregator pair.

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

### [`expressive-conversation/`](./expressive-conversation/)

Expressive conversation (alpha): the LLM writes expression tags into its replies, such as emotions and laughter, and the TTS performs them.

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
