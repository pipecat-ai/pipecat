# Pipecat Foundational Examples

This directory contains examples showing how to build voice and multimodal agents with Pipecat. Each example demonstrates specific features, progressing from basic to advanced concepts.

## Learning Paths

Depending on what you're trying to build, these learning paths will guide you through relevant examples:

- **New to Pipecat**: Start with examples [01](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/01-say-one-thing.py), [02](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/02-llm-say-one-thing.py), [07](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/07-interruptible.py)
- **Building conversational bots**: [07](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/07-interruptible.py), [10](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/10-wake-phrase.py), [38](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/38-smart-turn-fal.py)
- **Common add-on capabilities**: [17](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/17-detect-user-idle.py), [24](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/24-stt-mute-filter.py), [28](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/28-transcription-processor.py), [34](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/34-audio-recording.py)
- **Adding visual capabilities**: [03](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/03-still-frame.py), [12](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/12a-describe-video-gemini-flash.py), [26](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/26c-gemini-multimodal-live-video.py)
- **Advanced agent capabilities**: [14](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/14-function-calling.py), [20](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/20a-persistent-context-openai.py), [37](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/37-mem0.py)

## Quick Start

1. Set up a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys.

4. Run any example:

   ```bash
   python 01-say-one-thing.py
   ```

5. Open the web interface at http://localhost:7860 and click "Connect"

## Running examples with other transports

It is possible to run most of the examples with other transports such as Twilio or Daily.

### Daily

You need to create a Daily account at https://dashboard.daily.co/u/signup. Once signed up, you can create your own room from the dashboard and set the environment variables `DAILY_SAMPLE_ROOM_URL` and `DAILY_API_KEY`. Alternatively, you can let the example create a room for you (still needs `DAILY_API_KEY` environment variable). Then, start any example with `-t daily`:

```bash
python 07-interruptible.py -t daily
```

### Twilio

It is also possible to run the example through a Twilio phone number. You will
need to setup a few things:

1. Install and run [ngrok](https://ngrok.com/download).

  ```bash
  ngrok http 7860
  ```

2. Configure your Twilio phone number. One way is to setup a TwiML app and set the request URL to the ngrok URL from step (1). Then, set your phone number to use the new TwiML app.

Then, run the example with:

```bash
python 07-interruptible.py -t twilio -x NGROK_HOST_NAME (no protocol)
```

## Examples by Feature

### Basics

- **[01-say-one-thing.py](./01-say-one-thing.py)**: Most basic bot that says one phrase and exits (Transport, TTS, Event handlers)
- **[02-llm-say-one-thing.py](./02-llm-say-one-thing.py)**: Bot generates a response with an LLM (LLM initialization)
- **[03-still-frame.py](./03-still-frame.py)**: Displays a static image (Video transport, Image service)
- **[04-transport.py](./04-transport.py)**: Different transport options (WebRTC, Daily, Livekit)

### Conversational AI

- **[07-interruptible.py](./07-interruptible.py)**: Basic voice assistant bot (STT, TTS, LLM, Interruptible speech)
- **[10-wake-phrase.py](./10-wake-phrase.py)**: Bot activated by wake phrase (WakeCheckFilter)
- **[22-natural-conversation.py](./22-natural-conversation.py)**: Smart turn detection (Multiple LLMs, Turn management)
- **[38-smart-turn-fal.py](./38-smart-turn-fal.py)**: ML-based turn detection (Fal service, Local models)

### Common Utilities

- **[17-detect-user-idle.py](./17-detect-user-idle.py)**: Handle inactive users (UserIdleProcessor)
- **[24-stt-mute-filter.py](./24-stt-mute-filter.py)**: Selectively mute user input (STTMuteFilter)
- **[28-transcription-processor.py](./28-transcription-processor.py)**: Record conversation text (TranscriptProcessor)
- **[30-observer.py](./30-observer.py)**: Access frame data (Custom observers)
- **[31-heartbeats.py](./31-heartbeats.py)**: Detect idle pipelines (Pipeline monitoring)
- **[34-audio-recording.py](./34-audio-recording.py)**: Record conversation audio (Composite and track-level recording)

### Advanced LLM Features

- **[14-function-calling.py](./14-function-calling.py)**: Bot with tool usage (Function schemas, Tool registration)
- **[20a-persistent-context-openai.py](./20a-persistent-context-openai.py)**: Persistent conversation context (Memory management)
- **[32-gemini-grounding-metadata.py](./32-gemini-grounding-metadata.py)**: Web search capabilities (Google search integration)
- **[33-gemini-rag.py](./33-gemini-rag.py)**: Retrieval-augmented generation (Data sources, Grounding)
- **[37-mem0.py](./37-mem0.py)**: Long-term agent memory (Mem0 service integration)

### Media Handling

- **[05-sync-speech-and-images.py](./05-sync-speech-and-images.py)**: Synchronized narration with images (Custom processors, SyncParallelPipeline)
- **[06a-image-sync.py](./06a-image-sync.py)**: Dynamic image updates while speaking (Synchronized A/V pipelines)
- **[09-mirror.py](./09-mirror.py)**: Mirror user's audio and video (Custom frame processors)
- **[11-sound-effects.py](./11-sound-effects.py)**: Add sounds when bot speaks (Sound playback, Event synchronization)
- **[23-bot-background-sound.py](./23-bot-background-sound.py)**: Play background audio (SoundfileMixer)

### Vision & Multimodal

- **[12a-describe-video-gemini-flash.py](./12a-describe-video-gemini-flash.py)**: Bot describes user's video (Video input, Multimodal LLMs)
- **[26c-gemini-multimodal-live-video.py](./26c-gemini-multimodal-live-video.py)**: Gemini with video input (Streaming video, Function calls)

### Voice & Language

- **[13-transcription.py](./13-transcription.py)**: Speech transcription demo (STT providers, Real-time transcription)
- **[15-switch-voices.py](./15-switch-voices.py)**: Dynamic voice/language changing (ParallelPipelines, FunctionFilters)
- **[25-google-audio-in.py](./25-google-audio-in.py)**: Gemini for speech recognition (Alternative transcription)
- **[35-pattern-pair-voice-switching.py](./35-pattern-pair-voice-switching.py)**: Dynamic TTS voice switching (XML parsing, PatternPairAggregator)
- **[36-user-email-gathering.py](./36-user-email-gathering.py)**: Spelling mode for TTS (Confirmation patterns, XML tags)

### Integration Examples

- **[18-gstreamer-filesrc.py](./18-gstreamer-filesrc.py)**: GStreamer video streaming (Video processing)
- **[19-openai-realtime-beta.py](./19-openai-realtime-beta.py)**: OpenAI Speech-to-Speech (Direct S2S, Function calls)
- **[21-tavus-layer-tavus-transport.py](./21-tavus-layer-tavus-transport.py)**: Tavus digital twin (Avatar integration)
- **[27-simli-layer.py](./27-simli-layer.py)**: Simli avatar integration (Video synchronization)

### Performance & Optimization

- **[16-gpu-container-local-bot.py](./16-gpu-container-local-bot.py)**: GPU-accelerated local bot (Performance measurement)

### Utilities

## Advanced Usage

### Customizing Network Settings

```bash
python <example-name> --host 0.0.0.0 --port 8080
```

### Troubleshooting

- **No audio/video**: Check browser permissions for microphone and camera
- **Connection errors**: Verify API keys in `.env` file
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **Port conflicts**: Use `--port` to change the port

For more examples, visit our [GitHub repository](https://github.com/pipecat-ai/pipecat/tree/main/examples).
