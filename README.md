<h1><div align="center">
 <img alt="pipecat" width="300px" height="auto" src="https://raw.githubusercontent.com/pipecat-ai/pipecat/main/pipecat.png">
</div></h1>

[![PyPI](https://img.shields.io/pypi/v/pipecat-ai)](https://pypi.org/project/pipecat-ai) ![Tests](https://github.com/pipecat-ai/pipecat/actions/workflows/tests.yaml/badge.svg) [![codecov](https://codecov.io/gh/pipecat-ai/pipecat/graph/badge.svg?token=LNVUIVO4Y9)](https://codecov.io/gh/pipecat-ai/pipecat) [![Docs](https://img.shields.io/badge/Documentation-blue)](https://docs.pipecat.ai) [![Discord](https://img.shields.io/discord/1239284677165056021)](https://discord.gg/pipecat) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/pipecat-ai/pipecat)
[![](https://getmanta.ai/api/badges?text=Manta%20Graph&link=manta)](https://getmanta.ai/pipecat)

# 🎙️ Pipecat: Real-Time Voice & Multimodal AI Agents

**Pipecat** is an open-source Python framework for building real-time voice and multimodal conversational agents. Orchestrate audio and video, AI services, different transports, and conversation pipelines effortlessly—so you can focus on what makes your agent unique.

> Want to dive right in? Try the [quickstart](https://docs.pipecat.ai/getting-started/quickstart).

## 🚀 What You Can Build

- **Voice Assistants** – natural, streaming conversations with AI
- **AI Companions** – coaches, meeting assistants, characters
- **Multimodal Interfaces** – voice, video, images, and more
- **Interactive Storytelling** – creative tools with generative media
- **Business Agents** – customer intake, support bots, guided flows
- **Complex Dialog Systems** – design logic with structured conversations

## 🧠 Why Pipecat?

- **Voice-first**: Integrates speech recognition, text-to-speech, and conversation handling
- **Pluggable**: Supports many AI services and tools
- **Composable Pipelines**: Build complex behavior from modular components
- **Real-Time**: Ultra-low latency interaction with different transports (e.g. WebSockets or WebRTC)

## 🌐 Pipecat Ecosystem

### 📱 Client SDKs

Building client applications? You can connect to Pipecat from any platform using our official SDKs:

<a href="https://docs.pipecat.ai/client/js/introduction">JavaScript</a> | <a href="https://docs.pipecat.ai/client/react/introduction">React</a> | <a href="https://docs.pipecat.ai/client/react-native/introduction">React Native</a> |
<a href="https://docs.pipecat.ai/client/ios/introduction">Swift</a> | <a href="https://docs.pipecat.ai/client/android/introduction">Kotlin</a> | <a href="https://docs.pipecat.ai/client/c++/introduction">C++</a> | <a href="https://github.com/pipecat-ai/pipecat-esp32">ESP32</a>

### 🧭 Structured conversations

Looking to build structured conversations? Check out [Pipecat Flows](https://github.com/pipecat-ai/pipecat-flows) for managing complex conversational states and transitions.

### 🪄 Beautiful UIs

Want to build beautiful and engaging experiences? Checkout the [Voice UI Kit](https://github.com/pipecat-ai/voice-ui-kit), a collection of components, hooks and templates for building voice AI applications quickly.

### 🔍 Debugging

Looking for help debugging your pipeline and processors? Check out [Whisker](https://github.com/pipecat-ai/whisker), a real-time Pipecat debugger.

### 🖥️ Terminal

Love terminal applications? Check out [Tail](https://github.com/pipecat-ai/tail), a terminal dashboard for Pipecat.

### 📺️ Pipecat TV Channel

Catch new features, interviews, and how-tos on our [Pipecat TV](https://www.youtube.com/playlist?list=PLzU2zoMTQIHjqC3v4q2XVSR3hGSzwKFwH) channel.

## 🎬 See it in action

<p float="left">
    <a href="https://github.com/pipecat-ai/pipecat-examples/tree/main/simple-chatbot"><img src="https://raw.githubusercontent.com/pipecat-ai/pipecat-examples/main/simple-chatbot/image.png" width="400" /></a>&nbsp;
    <a href="https://github.com/pipecat-ai/pipecat-examples/tree/main/storytelling-chatbot"><img src="https://raw.githubusercontent.com/pipecat-ai/pipecat-examples/main/storytelling-chatbot/image.png" width="400" /></a>
    <br/>
    <a href="https://github.com/pipecat-ai/pipecat-examples/tree/main/translation-chatbot"><img src="https://raw.githubusercontent.com/pipecat-ai/pipecat-examples/main/translation-chatbot/image.png" width="400" /></a>&nbsp;
    <a href="https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/12-describe-video.py"><img src="https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/assets/moondream.png" width="400" /></a>
</p>

## 🧩 Available services

| Category            | Services                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Speech-to-Text      | [AssemblyAI](https://docs.pipecat.ai/server/services/stt/assemblyai), [AWS](https://docs.pipecat.ai/server/services/stt/aws), [Azure](https://docs.pipecat.ai/server/services/stt/azure), [Cartesia](https://docs.pipecat.ai/server/services/stt/cartesia), [Deepgram](https://docs.pipecat.ai/server/services/stt/deepgram), [ElevenLabs](https://docs.pipecat.ai/server/services/stt/elevenlabs), [Fal Wizper](https://docs.pipecat.ai/server/services/stt/fal), [Gladia](https://docs.pipecat.ai/server/services/stt/gladia), [Google](https://docs.pipecat.ai/server/services/stt/google), [Groq (Whisper)](https://docs.pipecat.ai/server/services/stt/groq), [NVIDIA Riva](https://docs.pipecat.ai/server/services/stt/riva), [OpenAI (Whisper)](https://docs.pipecat.ai/server/services/stt/openai), [SambaNova (Whisper)](https://docs.pipecat.ai/server/services/stt/sambanova), [Soniox](https://docs.pipecat.ai/server/services/stt/soniox), [Speechmatics](https://docs.pipecat.ai/server/services/stt/speechmatics), [Ultravox](https://docs.pipecat.ai/server/services/stt/ultravox), [Whisper](https://docs.pipecat.ai/server/services/stt/whisper)                                                                                                                                                                              |
| LLMs                | [Anthropic](https://docs.pipecat.ai/server/services/llm/anthropic), [AWS](https://docs.pipecat.ai/server/services/llm/aws), [Azure](https://docs.pipecat.ai/server/services/llm/azure), [Cerebras](https://docs.pipecat.ai/server/services/llm/cerebras), [DeepSeek](https://docs.pipecat.ai/server/services/llm/deepseek), [Fireworks AI](https://docs.pipecat.ai/server/services/llm/fireworks), [Gemini](https://docs.pipecat.ai/server/services/llm/gemini), [Grok](https://docs.pipecat.ai/server/services/llm/grok), [Groq](https://docs.pipecat.ai/server/services/llm/groq), [Mistral](https://docs.pipecat.ai/server/services/llm/mistral), [NVIDIA NIM](https://docs.pipecat.ai/server/services/llm/nim), [Ollama](https://docs.pipecat.ai/server/services/llm/ollama), [OpenAI](https://docs.pipecat.ai/server/services/llm/openai), [OpenRouter](https://docs.pipecat.ai/server/services/llm/openrouter), [Perplexity](https://docs.pipecat.ai/server/services/llm/perplexity), [Qwen](https://docs.pipecat.ai/server/services/llm/qwen), [SambaNova](https://docs.pipecat.ai/server/services/llm/sambanova) [Together AI](https://docs.pipecat.ai/server/services/llm/together)                                                                                                                                                    |
| Text-to-Speech      | [Async](https://docs.pipecat.ai/server/services/tts/asyncai), [AWS](https://docs.pipecat.ai/server/services/tts/aws), [Azure](https://docs.pipecat.ai/server/services/tts/azure), [Cartesia](https://docs.pipecat.ai/server/services/tts/cartesia), [Deepgram](https://docs.pipecat.ai/server/services/tts/deepgram), [ElevenLabs](https://docs.pipecat.ai/server/services/tts/elevenlabs), [Fish](https://docs.pipecat.ai/server/services/tts/fish), [Google](https://docs.pipecat.ai/server/services/tts/google), [Groq](https://docs.pipecat.ai/server/services/tts/groq), [Hume](https://docs.pipecat.ai/server/services/tts/hume), [Inworld](https://docs.pipecat.ai/server/services/tts/inworld), [LMNT](https://docs.pipecat.ai/server/services/tts/lmnt), [MiniMax](https://docs.pipecat.ai/server/services/tts/minimax), [Neuphonic](https://docs.pipecat.ai/server/services/tts/neuphonic), [NVIDIA Riva](https://docs.pipecat.ai/server/services/tts/riva), [OpenAI](https://docs.pipecat.ai/server/services/tts/openai), [Piper](https://docs.pipecat.ai/server/services/tts/piper), [PlayHT](https://docs.pipecat.ai/server/services/tts/playht), [Rime](https://docs.pipecat.ai/server/services/tts/rime), [Sarvam](https://docs.pipecat.ai/server/services/tts/sarvam), [XTTS](https://docs.pipecat.ai/server/services/tts/xtts) |
| Speech-to-Speech    | [AWS Nova Sonic](https://docs.pipecat.ai/server/services/s2s/aws), [Gemini Multimodal Live](https://docs.pipecat.ai/server/services/s2s/gemini), [OpenAI Realtime](https://docs.pipecat.ai/server/services/s2s/openai)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Transport           | [Daily (WebRTC)](https://docs.pipecat.ai/server/services/transport/daily), [FastAPI Websocket](https://docs.pipecat.ai/server/services/transport/fastapi-websocket), [SmallWebRTCTransport](https://docs.pipecat.ai/server/services/transport/small-webrtc), [WebSocket Server](https://docs.pipecat.ai/server/services/transport/websocket-server), Local                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Serializers         | [Plivo](https://docs.pipecat.ai/server/utilities/serializers/plivo), [Twilio](https://docs.pipecat.ai/server/utilities/serializers/twilio), [Telnyx](https://docs.pipecat.ai/server/utilities/serializers/telnyx)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Video               | [HeyGen](https://docs.pipecat.ai/server/services/video/heygen), [Tavus](https://docs.pipecat.ai/server/services/video/tavus), [Simli](https://docs.pipecat.ai/server/services/video/simli)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Memory              | [mem0](https://docs.pipecat.ai/server/services/memory/mem0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Vision & Image      | [fal](https://docs.pipecat.ai/server/services/image-generation/fal), [Google Imagen](https://docs.pipecat.ai/server/services/image-generation/fal), [Moondream](https://docs.pipecat.ai/server/services/vision/moondream)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Audio Processing    | [Silero VAD](https://docs.pipecat.ai/server/utilities/audio/silero-vad-analyzer), [Krisp](https://docs.pipecat.ai/server/utilities/audio/krisp-filter), [Koala](https://docs.pipecat.ai/server/utilities/audio/koala-filter), [ai-coustics](https://docs.pipecat.ai/server/utilities/audio/aic-filter)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Analytics & Metrics | [OpenTelemetry](https://docs.pipecat.ai/server/utilities/opentelemetry), [Sentry](https://docs.pipecat.ai/server/services/analytics/sentry)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

📚 [View full services documentation →](https://docs.pipecat.ai/server/services/supported-services)

## ⚡ Getting started

You can get started with Pipecat running on your local machine, then move your agent processes to the cloud when you're ready.

1. Install uv

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   > **Need help?** Refer to the [uv install documentation](https://docs.astral.sh/uv/getting-started/installation/).

2. Install the module

   ```bash
   # For new projects
   uv init my-pipecat-app
   cd my-pipecat-app
   uv add pipecat-ai

   # Or for existing projects
   uv add pipecat-ai
   ```

3. Set up your environment

   ```bash
   cp env.example .env
   ```

4. To keep things lightweight, only the core framework is included by default. If you need support for third-party AI services, you can add the necessary dependencies with:

   ```bash
   uv add "pipecat-ai[option,...]"
   ```

> **Using pip?** You can still use `pip install pipecat-ai` and `pip install "pipecat-ai[option,...]"` to get set up.

## 🧪 Code examples

- [Foundational](https://github.com/pipecat-ai/pipecat/tree/main/examples/foundational) — small snippets that build on each other, introducing one or two concepts at a time
- [Example apps](https://github.com/pipecat-ai/pipecat-examples) — complete applications that you can use as starting points for development

## 🛠️ Contributing to the framework

### Prerequisites

**Minimum Python Version:** 3.10
**Recommended Python Version:** 3.12

### Setup Steps

1. Clone the repository and navigate to it:

   ```bash
   git clone https://github.com/pipecat-ai/pipecat.git
   cd pipecat
   ```

2. Install development and testing dependencies:

   ```bash
   uv sync --group dev --all-extras \
     --no-extra gstreamer \
     --no-extra krisp \
     --no-extra local \
     --no-extra ultravox # (ultravox not fully supported on macOS)
   ```

3. Install the git pre-commit hooks:

   ```bash
   uv run pre-commit install
   ```

> **Note**: Some extras (local, gstreamer) require system dependencies. See documentation if you encounter build errors.

### Running tests

To run all tests, from the root directory:

```bash
uv run pytest
```

Run a specific test suite:

```bash
uv run pytest tests/test_name.py
```

## 🤝 Contributing

We welcome contributions from the community! Whether you're fixing bugs, improving documentation, or adding new features, here's how you can help:

- **Found a bug?** Open an [issue](https://github.com/pipecat-ai/pipecat/issues)
- **Have a feature idea?** Start a [discussion](https://discord.gg/pipecat)
- **Want to contribute code?** Check our [CONTRIBUTING.md](CONTRIBUTING.md) guide
- **Documentation improvements?** [Docs](https://github.com/pipecat-ai/docs) PRs are always welcome

Before submitting a pull request, please check existing issues and PRs to avoid duplicates.

We aim to review all contributions promptly and provide constructive feedback to help get your changes merged.

## 🛟 Getting help

➡️ [Join our Discord](https://discord.gg/pipecat)

➡️ [Read the docs](https://docs.pipecat.ai)

➡️ [Reach us on X](https://x.com/pipecat_ai)
