<h1><div align="center">
 <img alt="pipecat" width="300px" height="auto" src="https://raw.githubusercontent.com/pipecat-ai/pipecat/main/pipecat.png">
</div></h1>

[![PyPI](https://img.shields.io/pypi/v/pipecat-ai)](https://pypi.org/project/pipecat-ai) ![Tests](https://github.com/pipecat-ai/pipecat/actions/workflows/tests.yaml/badge.svg) [![codecov](https://codecov.io/gh/pipecat-ai/pipecat/graph/badge.svg?token=LNVUIVO4Y9)](https://codecov.io/gh/pipecat-ai/pipecat) [![Docs](https://img.shields.io/badge/Documentation-blue)](https://docs.pipecat.ai) [![Discord](https://img.shields.io/discord/1239284677165056021)](https://discord.gg/pipecat) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/pipecat-ai/pipecat)

# 🎙️ Pipecat: Real-Time Voice & Multimodal AI Agents

**Pipecat** is an open-source Python framework for building real-time voice and multimodal conversational agents. Orchestrate audio and video, AI services, different transports, and conversation pipelines effortlessly—so you can focus on what makes your agent unique.

> Want to dive right in? Run `pipecat init quickstart` or follow the [quickstart guide](https://docs.pipecat.ai/getting-started/quickstart).

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

### 🧩 Multi-agent systems

Need multiple AI agents working together? [Pipecat Subagents](https://github.com/pipecat-ai/pipecat-subagents) lets you build distributed multi-agent systems where each agent runs its own pipeline and communicates through a shared message bus. Hand off conversations between specialists, dispatch background tasks, and scale agents across processes or machines.

### 📱 Client SDKs

Building client applications? You can connect to Pipecat from any platform using our official SDKs:

<a href="https://docs.pipecat.ai/client/js/introduction">JavaScript</a> | <a href="https://docs.pipecat.ai/client/react/introduction">React</a> | <a href="https://docs.pipecat.ai/client/react-native/introduction">React Native</a> |
<a href="https://docs.pipecat.ai/client/ios/introduction">Swift</a> | <a href="https://docs.pipecat.ai/client/android/introduction">Kotlin</a> | <a href="https://docs.pipecat.ai/client/c++/introduction">C++</a> | <a href="https://github.com/pipecat-ai/pipecat-esp32">ESP32</a>

### 🧭 Structured conversations

Looking to build structured conversations? Check out [Pipecat Flows](https://github.com/pipecat-ai/pipecat-flows) for managing complex conversational states and transitions.

### 🪄 Beautiful UIs

Want to build beautiful and engaging experiences? Checkout the [Voice UI Kit](https://github.com/pipecat-ai/voice-ui-kit), a collection of components, hooks and templates for building voice AI applications quickly.

### 🛠️ Create and deploy projects

Create a new project in under a minute with the [Pipecat CLI](https://github.com/pipecat-ai/pipecat-cli). Then use the CLI to monitor and deploy your agent to production.

### 🔍 Debugging

Looking for help debugging your pipeline and processors? Check out [Whisker](https://github.com/pipecat-ai/whisker), a real-time Pipecat debugger.

### 🖥️ Terminal

Love terminal applications? Check out [Tail](https://github.com/pipecat-ai/tail), a terminal dashboard for Pipecat.

### 🤖 Claude Code Skills

Use [Pipecat Skills](https://github.com/pipecat-ai/skills) with [Claude Code](https://claude.ai/code) to scaffold projects, deploy to Pipecat Cloud, and more. Install the marketplace with:

```
claude plugin marketplace add pipecat-ai/skills
```

and install any of the available plugins.

### 🧩 Community Integrations

Build and share your own Pipecat service integrations! Browse existing [community integrations](https://docs.pipecat.ai/api-reference/server/services/community-integrations) or check out our [guide](COMMUNITY_INTEGRATIONS.md) to create your own.

### 📺️ Pipecat TV Channel

Catch new features, interviews, and how-tos on our [Pipecat TV](https://www.youtube.com/playlist?list=PLzU2zoMTQIHjqC3v4q2XVSR3hGSzwKFwH) channel.

## 🎬 See it in action

<p float="left">
    <a href="https://github.com/pipecat-ai/pipecat-examples/tree/main/simple-chatbot"><img src="https://raw.githubusercontent.com/pipecat-ai/pipecat-examples/main/simple-chatbot/image.png" width="400" /></a>&nbsp;
    <a href="https://github.com/pipecat-ai/pipecat-examples/tree/main/storytelling-chatbot"><img src="https://raw.githubusercontent.com/pipecat-ai/pipecat-examples/main/storytelling-chatbot/image.png" width="400" /></a>
    <br/>
    <a href="https://github.com/pipecat-ai/pipecat-examples/tree/main/daily-multi-translation"><img src="https://raw.githubusercontent.com/pipecat-ai/pipecat-examples/main/daily-multi-translation/image.png" width="400" /></a>&nbsp;
    <a href="https://github.com/pipecat-ai/pipecat/blob/main/examples/vision/vision-moondream.py"><img src="https://github.com/pipecat-ai/pipecat/blob/main/examples/assets/moondream.png" width="400" /></a>
</p>

## 🧩 Available services

| Category            | Services                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Speech-to-Text      | [AssemblyAI](https://docs.pipecat.ai/api-reference/server/services/stt/assemblyai), [AWS](https://docs.pipecat.ai/api-reference/server/services/stt/aws), [Azure](https://docs.pipecat.ai/api-reference/server/services/stt/azure), [Cartesia](https://docs.pipecat.ai/api-reference/server/services/stt/cartesia), [Deepgram](https://docs.pipecat.ai/api-reference/server/services/stt/deepgram), [ElevenLabs](https://docs.pipecat.ai/api-reference/server/services/stt/elevenlabs), [Fal Wizper](https://docs.pipecat.ai/api-reference/server/services/stt/fal), [Gladia](https://docs.pipecat.ai/api-reference/server/services/stt/gladia), [Google](https://docs.pipecat.ai/api-reference/server/services/stt/google), [Gradium](https://docs.pipecat.ai/api-reference/server/services/stt/gradium), [Groq (Whisper)](https://docs.pipecat.ai/api-reference/server/services/stt/groq), [Mistral](https://docs.pipecat.ai/api-reference/server/services/stt/mistral), [NVIDIA Riva](https://docs.pipecat.ai/api-reference/server/services/stt/riva), [OpenAI (Whisper)](https://docs.pipecat.ai/api-reference/server/services/stt/openai), [Sarvam](https://docs.pipecat.ai/api-reference/server/services/stt/sarvam), [Soniox](https://docs.pipecat.ai/api-reference/server/services/stt/soniox), [Speechmatics](https://docs.pipecat.ai/api-reference/server/services/stt/speechmatics), [Whisper](https://docs.pipecat.ai/api-reference/server/services/stt/whisper), [xAI](https://docs.pipecat.ai/api-reference/server/services/stt/xai)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| LLMs                | [Anthropic](https://docs.pipecat.ai/api-reference/server/services/llm/anthropic), [AWS](https://docs.pipecat.ai/api-reference/server/services/llm/aws), [Azure](https://docs.pipecat.ai/api-reference/server/services/llm/azure), [Cerebras](https://docs.pipecat.ai/api-reference/server/services/llm/cerebras), [DeepSeek](https://docs.pipecat.ai/api-reference/server/services/llm/deepseek), [Fireworks AI](https://docs.pipecat.ai/api-reference/server/services/llm/fireworks), [Gemini](https://docs.pipecat.ai/api-reference/server/services/llm/gemini), [Grok](https://docs.pipecat.ai/api-reference/server/services/llm/grok), [Groq](https://docs.pipecat.ai/api-reference/server/services/llm/groq), [Mistral](https://docs.pipecat.ai/api-reference/server/services/llm/mistral), [Nebius](https://docs.pipecat.ai/api-reference/server/services/llm/nebius), [Novita](https://docs.pipecat.ai/api-reference/server/services/llm/novita), [NVIDIA NIM](https://docs.pipecat.ai/api-reference/server/services/llm/nvidia), [Ollama](https://docs.pipecat.ai/api-reference/server/services/llm/ollama), [OpenAI](https://docs.pipecat.ai/api-reference/server/services/llm/openai), [OpenAI Responses](https://docs.pipecat.ai/api-reference/server/services/llm/openai-responses), [OpenRouter](https://docs.pipecat.ai/api-reference/server/services/llm/openrouter), [Perplexity](https://docs.pipecat.ai/api-reference/server/services/llm/perplexity), [Qwen](https://docs.pipecat.ai/api-reference/server/services/llm/qwen), [SambaNova](https://docs.pipecat.ai/api-reference/server/services/llm/sambanova), [Sarvam](https://docs.pipecat.ai/api-reference/server/services/llm/sarvam), [Together AI](https://docs.pipecat.ai/api-reference/server/services/llm/together)                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Text-to-Speech      | [Async](https://docs.pipecat.ai/api-reference/server/services/tts/asyncai), [AWS](https://docs.pipecat.ai/api-reference/server/services/tts/aws), [Azure](https://docs.pipecat.ai/api-reference/server/services/tts/azure), [Camb AI](https://docs.pipecat.ai/api-reference/server/services/tts/camb), [Cartesia](https://docs.pipecat.ai/api-reference/server/services/tts/cartesia), [Deepgram](https://docs.pipecat.ai/api-reference/server/services/tts/deepgram), [ElevenLabs](https://docs.pipecat.ai/api-reference/server/services/tts/elevenlabs), [Fish](https://docs.pipecat.ai/api-reference/server/services/tts/fish), [Google](https://docs.pipecat.ai/api-reference/server/services/tts/google), [Gradium](https://docs.pipecat.ai/api-reference/server/services/tts/gradium), [Groq](https://docs.pipecat.ai/api-reference/server/services/tts/groq), [Hume](https://docs.pipecat.ai/api-reference/server/services/tts/hume), [Inworld](https://docs.pipecat.ai/api-reference/server/services/tts/inworld), [Kokoro](https://docs.pipecat.ai/api-reference/server/services/tts/kokoro), [LMNT](https://docs.pipecat.ai/api-reference/server/services/tts/lmnt), [MiniMax](https://docs.pipecat.ai/api-reference/server/services/tts/minimax), [Mistral](https://docs.pipecat.ai/api-reference/server/services/tts/mistral), [Neuphonic](https://docs.pipecat.ai/api-reference/server/services/tts/neuphonic), [NVIDIA Riva](https://docs.pipecat.ai/api-reference/server/services/tts/riva), [OpenAI](https://docs.pipecat.ai/api-reference/server/services/tts/openai), [Piper](https://docs.pipecat.ai/api-reference/server/services/tts/piper), [Resemble](https://docs.pipecat.ai/api-reference/server/services/tts/resemble), [Rime](https://docs.pipecat.ai/api-reference/server/services/tts/rime), [Sarvam](https://docs.pipecat.ai/api-reference/server/services/tts/sarvam), [Smallest](https://docs.pipecat.ai/api-reference/server/services/tts/smallest), [Speechmatics](https://docs.pipecat.ai/api-reference/server/services/tts/speechmatics), [xAI](https://docs.pipecat.ai/api-reference/server/services/tts/xai), [XTTS](https://docs.pipecat.ai/api-reference/server/services/tts/xtts) |
| Speech-to-Speech    | [AWS Nova Sonic](https://docs.pipecat.ai/api-reference/server/services/s2s/aws), [Gemini Multimodal Live](https://docs.pipecat.ai/api-reference/server/services/s2s/gemini), [Grok Voice Agent](https://docs.pipecat.ai/api-reference/server/services/s2s/grok), [OpenAI Realtime](https://docs.pipecat.ai/api-reference/server/services/s2s/openai), [Ultravox](https://docs.pipecat.ai/api-reference/server/services/s2s/ultravox),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Transport           | [Daily (WebRTC)](https://docs.pipecat.ai/api-reference/server/services/transport/daily), [FastAPI Websocket](https://docs.pipecat.ai/api-reference/server/services/transport/fastapi-websocket), [LiveKit (WebRTC)](https://docs.pipecat.ai/api-reference/server/services/transport/livekit), [SmallWebRTCTransport](https://docs.pipecat.ai/api-reference/server/services/transport/small-webrtc), [WebSocket Server](https://docs.pipecat.ai/api-reference/server/services/transport/websocket-server), [WhatsApp](https://docs.pipecat.ai/api-reference/server/services/transport/whatsapp), Local                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Serializers         | [Exotel](https://docs.pipecat.ai/api-reference/server/services/serializers/exotel), [Genesys](https://docs.pipecat.ai/api-reference/server/services/serializers/genesys), [Plivo](https://docs.pipecat.ai/api-reference/server/services/serializers/plivo), [Twilio](https://docs.pipecat.ai/api-reference/server/services/serializers/twilio), [Telnyx](https://docs.pipecat.ai/api-reference/server/services/serializers/telnyx), [Vonage](https://docs.pipecat.ai/api-reference/server/services/serializers/vonage)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Video               | [HeyGen](https://docs.pipecat.ai/api-reference/server/services/video/heygen), [LemonSlice](https://docs.pipecat.ai/api-reference/server/services/transport/lemonslice), [Tavus](https://docs.pipecat.ai/api-reference/server/services/video/tavus), [Simli](https://docs.pipecat.ai/api-reference/server/services/video/simli)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Memory              | [mem0](https://docs.pipecat.ai/api-reference/server/services/memory/mem0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Vision & Image      | [fal](https://docs.pipecat.ai/api-reference/server/services/image-generation/fal), [Google Imagen](https://docs.pipecat.ai/api-reference/server/services/image-generation/google-imagen), [Moondream](https://docs.pipecat.ai/api-reference/server/services/vision/moondream)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Audio Processing    | [Silero VAD](https://docs.pipecat.ai/api-reference/server/utilities/audio/silero-vad-analyzer), [Krisp Viva](https://docs.pipecat.ai/guides/features/krisp-viva), [Koala](https://docs.pipecat.ai/api-reference/server/utilities/audio/koala-filter), [ai-coustics](https://docs.pipecat.ai/api-reference/server/utilities/audio/aic-filter), [RNNoise](https://docs.pipecat.ai/api-reference/server/utilities/audio/rnnoise-filter)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Analytics & Metrics | [OpenTelemetry](https://docs.pipecat.ai/api-reference/server/utilities/opentelemetry), [Sentry](https://docs.pipecat.ai/api-reference/server/services/analytics/sentry)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Community           | [Browse community integrations →](https://docs.pipecat.ai/api-reference/server/services/community-integrations)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

📚 [View full services documentation →](https://docs.pipecat.ai/api-reference/server/services/supported-services)

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

- [Foundational](https://github.com/pipecat-ai/pipecat/tree/main/examples) — small snippets that build on each other, introducing one or two concepts at a time
- [Example apps](https://github.com/pipecat-ai/pipecat-examples) — complete applications that you can use as starting points for development

## 🛠️ Contributing to the framework

### Prerequisites

**Minimum Python Version:** 3.11
**Recommended Python Version:** >= 3.12

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
     --no-extra local \
   ```

3. Install the git pre-commit hooks:

   ```bash
   uv run pre-commit install
   ```

> **Note**: Some extras (local, gstreamer) require system dependencies. See documentation if you encounter build errors.

### Claude Code Skills

Install development workflow skills for contributing to Pipecat with [Claude Code](https://claude.ai/code):

```
claude plugin marketplace add pipecat-ai/pipecat
claude plugin install pipecat-dev@pipecat-dev-skills
```

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
