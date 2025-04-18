<h1><div align="center">
 <img alt="pipecat" width="300px" height="auto" src="https://raw.githubusercontent.com/pipecat-ai/pipecat/main/pipecat.png">
</div></h1>

[![PyPI](https://img.shields.io/pypi/v/pipecat-ai)](https://pypi.org/project/pipecat-ai) ![Tests](https://github.com/pipecat-ai/pipecat/actions/workflows/tests.yaml/badge.svg) [![codecov](https://codecov.io/gh/pipecat-ai/pipecat/graph/badge.svg?token=LNVUIVO4Y9)](https://codecov.io/gh/pipecat-ai/pipecat) [![Docs](https://img.shields.io/badge/Documentation-blue)](https://docs.pipecat.ai) [![Discord](https://img.shields.io/discord/1239284677165056021)](https://discord.gg/pipecat)

# 🎙️ Pipecat: Real-Time Voice & Multimodal AI Agents

**Pipecat** is an open-source Python framework for building real-time voice and multimodal conversational agents. Orchestrate audio and video, AI services, different transports, and conversation pipelines effortlessly—so you can focus on what makes your agent unique.

## 🚀 What You Can Build

- **Voice Assistants** – natural, streaming conversations with AI
- **AI Companions** – coaches, meeting assistants, characters
- **Multimodal Interfaces** – voice, video, images, and more
- **Interactive Storytelling** – creative tools with generative media
- **Business Agents** – customer intake, support bots, guided flows
- **Complex Dialog Systems** – design logic with structured conversations

🧭 Looking to build structured conversations? Check out [Pipecat Flows](https://github.com/pipecat-ai/pipecat-flows) for managing complex conversational states and transitions.

## 🧠 Why Pipecat?

- 🗣 **Voice-first**: Integrates speech recognition, text-to-speech, and conversation handling
- 🔌 **Pluggable**: Supports many AI services and tools
- 🔄 **Composable Pipelines**: Build complex behavior from modular components
- ⚡ **Real-Time**: Ultra-low latency interaction with different transports (e.g. WebSockets or WebRTC)

## 🎬 See it in action

<p float="left">
    <a href="https://github.com/pipecat-ai/pipecat/tree/main/examples/simple-chatbot"><img src="https://raw.githubusercontent.com/pipecat-ai/pipecat/main/examples/simple-chatbot/image.png" width="280" /></a>&nbsp;
    <a href="https://github.com/pipecat-ai/pipecat/tree/main/examples/storytelling-chatbot"><img src="https://raw.githubusercontent.com/pipecat-ai/pipecat/main/examples/storytelling-chatbot/image.png" width="280" /></a>
    <br/>
    <a href="https://github.com/pipecat-ai/pipecat/tree/main/examples/translation-chatbot"><img src="https://raw.githubusercontent.com/pipecat-ai/pipecat/main/examples/translation-chatbot/image.png" width="280" /></a>&nbsp;
    <a href="https://github.com/pipecat-ai/pipecat/tree/main/examples/moondream-chatbot"><img src="https://raw.githubusercontent.com/pipecat-ai/pipecat/main/examples/moondream-chatbot/image.png" width="280" /></a>
</p>

## 📱 Client SDKs

You can connect to Pipecat from any platform using our official SDKs:

| Platform                     | SDK Repo                                                                                                             | Description                              |
|------------------------------|----------------------------------------------------------------------------------------------------------------------|------------------------------------------|
| Web                          | [pipecat-client-web](https://github.com/pipecat-ai/pipecat-client-web)                                               | Web client SDK                           |
| Web transports               | [pipecat-client-web-transports](https://github.com/pipecat-ai/pipecat-client-web-transports)                         | Web client transports                    |
| iOS                          | [pipecat-client-ios](https://github.com/pipecat-ai/pipecat-client-ios)                                               | Swift package for iOS                    |
| iOS (P2P WebRTC)             | [pipecat-client-ios-small-webrtc](https://github.com/pipecat-ai/pipecat-client-ios-small-webrtc)                     | iOS transport for P2P WebRTC             |
| iOS (Daily)                  | [pipecat-client-ios-daily](https://github.com/pipecat-ai/pipecat-client-ios-daily)                                   | iOS transport for Daily                  |
| iOS (OpenAI Realtime)        | [pipecat-client-ios-openai-realtime-webrtc](https://github.com/pipecat-ai/pipecat-client-ios-openai-realtime-webrtc) | iOS transport for OpenAI Realtime        |
| iOS (Gemini Multimodal Live) | [pipecat-client-ios-gemini-live-websocket](https://github.com/pipecat-ai/pipecat-client-ios-gemini-live-websocket)   | iOS transport for Gemini Multimodal Live |
| Android                      | [pipecat-client-android](https://github.com/pipecat-ai/pipecat-client-android)                                       | Kotlin SDK for Android                   |
| Android transports           | [pipecat-client-android-transports](https://github.com/pipecat-ai/pipecat-client-android-transports)                 | Android client transports                |
| C++                          | [pipecat-client-cxx](https://github.com/pipecat-ai/pipecat-client-cxx)                                               | C++ client SDK                           |
| C++ (Daily)                  | [pipecat-client-cxx-daily](https://github.com/pipecat-ai/pipecat-client-cxx-daily)                                   | C++ transport for Daily                  |

## 🧩 Available services

| Category            | Services                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Install Command Example                 |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| Speech-to-Text      | [AssemblyAI](https://docs.pipecat.ai/server/services/stt/assemblyai), [Azure](https://docs.pipecat.ai/server/services/stt/azure), [Deepgram](https://docs.pipecat.ai/server/services/stt/deepgram), [Fal Wizper](https://docs.pipecat.ai/server/services/stt/fal), [Gladia](https://docs.pipecat.ai/server/services/stt/gladia), [Google](https://docs.pipecat.ai/server/services/stt/google), [Groq (Whisper)](https://docs.pipecat.ai/server/services/stt/groq), [OpenAI (Whisper)](https://docs.pipecat.ai/server/services/stt/openai), [Parakeet (NVIDIA)](https://docs.pipecat.ai/server/services/stt/parakeet), [Ultravox](https://docs.pipecat.ai/server/services/stt/ultravox), [Whisper](https://docs.pipecat.ai/server/services/stt/whisper)                                                                                                                                                                                                                                            | `pip install "pipecat-ai[deepgram]"`    |
| LLMs                | [Anthropic](https://docs.pipecat.ai/server/services/llm/anthropic), [Azure](https://docs.pipecat.ai/server/services/llm/azure), [Cerebras](https://docs.pipecat.ai/server/services/llm/cerebras), [DeepSeek](https://docs.pipecat.ai/server/services/llm/deepseek), [Fireworks AI](https://docs.pipecat.ai/server/services/llm/fireworks), [Gemini](https://docs.pipecat.ai/server/services/llm/gemini), [Grok](https://docs.pipecat.ai/server/services/llm/grok), [Groq](https://docs.pipecat.ai/server/services/llm/groq), [NVIDIA NIM](https://docs.pipecat.ai/server/services/llm/nim), [Ollama](https://docs.pipecat.ai/server/services/llm/ollama), [OpenAI](https://docs.pipecat.ai/server/services/llm/openai), [OpenRouter](https://docs.pipecat.ai/server/services/llm/openrouter), [Perplexity](https://docs.pipecat.ai/server/services/llm/perplexity), [Qwen](https://docs.pipecat.ai/server/services/llm/qwen), [Together AI](https://docs.pipecat.ai/server/services/llm/together) | `pip install "pipecat-ai[openai]"`      |
| Text-to-Speech      | [AWS](https://docs.pipecat.ai/server/services/tts/aws), [Azure](https://docs.pipecat.ai/server/services/tts/azure), [Cartesia](https://docs.pipecat.ai/server/services/tts/cartesia), [Deepgram](https://docs.pipecat.ai/server/services/tts/deepgram), [ElevenLabs](https://docs.pipecat.ai/server/services/tts/elevenlabs), [FastPitch (NVIDIA)](https://docs.pipecat.ai/server/services/tts/fastpitch), [Fish](https://docs.pipecat.ai/server/services/tts/fish), [Google](https://docs.pipecat.ai/server/services/tts/google), [LMNT](https://docs.pipecat.ai/server/services/tts/lmnt), [Neuphonic](https://docs.pipecat.ai/server/services/tts/neuphonic), [OpenAI](https://docs.pipecat.ai/server/services/tts/openai), [Piper](https://docs.pipecat.ai/server/services/tts/piper), [PlayHT](https://docs.pipecat.ai/server/services/tts/playht), [Rime](https://docs.pipecat.ai/server/services/tts/rime), [XTTS](https://docs.pipecat.ai/server/services/tts/xtts)                       | `pip install "pipecat-ai[cartesia]"`    |
| Speech-to-Speech    | [Gemini Multimodal Live](https://docs.pipecat.ai/server/services/s2s/gemini), [OpenAI Realtime](https://docs.pipecat.ai/server/services/s2s/openai)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | `pip install "pipecat-ai[google]"`      |
| Transport           | [Daily (WebRTC)](https://docs.pipecat.ai/server/services/transport/daily), [FastAPI Websocket](https://docs.pipecat.ai/server/services/transport/fastapi-websocket), [SmallWebRTCTransport](https://docs.pipecat.ai/server/services/transport/small-webrtc), [WebSocket Server](https://docs.pipecat.ai/server/services/transport/websocket-server), Local                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | `pip install "pipecat-ai[daily]"`       |
| Video               | [Tavus](https://docs.pipecat.ai/server/services/video/tavus), [Simli](https://docs.pipecat.ai/server/services/video/simli)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | `pip install "pipecat-ai[tavus,simli]"` |
| Memory              | [mem0](https://docs.pipecat.ai/server/services/memory/mem0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `pip install "pipecat-ai[mem0]"`        |
| Vision & Image      | [fal](https://docs.pipecat.ai/server/services/image-generation/fal), [Google Imagen](https://docs.pipecat.ai/server/services/image-generation/fal), [Moondream](https://docs.pipecat.ai/server/services/vision/moondream)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | `pip install "pipecat-ai[moondream]"`   |
| Audio Processing    | [Silero VAD](https://docs.pipecat.ai/server/utilities/audio/silero-vad-analyzer), [Krisp](https://docs.pipecat.ai/server/utilities/audio/krisp-filter), [Koala](https://docs.pipecat.ai/server/utilities/audio/koala-filter), [Noisereduce](https://docs.pipecat.ai/server/utilities/audio/noisereduce-filter)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | `pip install "pipecat-ai[silero]"`      |
| Analytics & Metrics | [Canonical AI](https://docs.pipecat.ai/server/services/analytics/canonical), [Sentry](https://docs.pipecat.ai/server/services/analytics/sentry)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | `pip install "pipecat-ai[canonical]"`   |

📚 [View full services documentation →](https://docs.pipecat.ai/server/services/supported-services)

## 🚀 Getting started

You can get started with Pipecat running on your local machine, then move your agent processes to the cloud when you’re ready. You can also add a 📞 telephone number, 🖼️ image output, 📺 video input, use different LLMs, and more.

```shell
# Install the module
pip install pipecat-ai

# Set up your environment
cp dot-env.template .env
```

To keep things lightweight, only the core framework is included by default. If you need support for third-party AI services, you can add the necessary dependencies with:

```shell
pip install "pipecat-ai[option,...]"
```

## 🧪 Code examples

- [Foundational](https://github.com/pipecat-ai/pipecat/tree/main/examples/foundational) — small snippets that build on each other, introducing one or two concepts at a time
- [Example apps](https://github.com/pipecat-ai/pipecat/tree/main/examples/) — complete applications that you can use as starting points for development

## 🛠️ Hacking on the framework itself

_Note: You may need to set up a virtual environment before following these instructions. From the root of the repo:_

```shell
python3 -m venv venv
source venv/bin/activate
```

Install the development dependencies:

```shell
pip install -r dev-requirements.txt
```

Install the git pre-commit hooks (these help ensure your code follows project rules):

```shell
pre-commit install
```

Install the `pipecat-ai` package locally in editable mode:

```shell
pip install -e .
```

The `-e` or `--editable` option allows you to modify the code without reinstalling.

To include optional dependencies, add them to the install command. For example:

```shell
pip install -e ".[daily,deepgram,cartesia,openai,silero]"     # Updated for the services you're using
```

If you want to use this package from another directory:

```shell
pip install "path_to_this_repo[option,...]"
```

### Running tests

From the root directory, run:

```shell
pytest
```

### Setting up your editor

This project uses strict [PEP 8](https://peps.python.org/pep-0008/) formatting via [Ruff](https://github.com/astral-sh/ruff).

#### Emacs

You can use [use-package](https://github.com/jwiegley/use-package) to install [emacs-lazy-ruff](https://github.com/christophermadsen/emacs-lazy-ruff) package and configure `ruff` arguments:

```elisp
(use-package lazy-ruff
  :ensure t
  :hook ((python-mode . lazy-ruff-mode))
  :config
  (setq lazy-ruff-format-command "ruff format")
  (setq lazy-ruff-check-command "ruff check --select I"))
```

`ruff` was installed in the `venv` environment described before, so you should be able to use [pyvenv-auto](https://github.com/ryotaro612/pyvenv-auto) to automatically load that environment inside Emacs.

```elisp
(use-package pyvenv-auto
  :ensure t
  :defer t
  :hook ((python-mode . pyvenv-auto-run)))
```

#### Visual Studio Code

Install the
[Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) extension. Then edit the user settings (_Ctrl-Shift-P_ `Open User Settings (JSON)`) and set it as the default Python formatter, and enable formatting on save:

```json
"[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true
}
```

#### PyCharm

`ruff` was installed in the `venv` environment described before, now to enable autoformatting on save, go to `File` -> `Settings` -> `Tools` -> `File Watchers` and add a new watcher with the following settings:

1. **Name**: `Ruff formatter`
2. **File type**: `Python`
3. **Working directory**: `$ContentRoot$`
4. **Arguments**: `format $FilePath$`
5. **Program**: `$PyInterpreterDirectory$/ruff`

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
