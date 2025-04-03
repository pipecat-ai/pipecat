<h1><div align="center">
 <img alt="pipecat" width="300px" height="auto" src="https://raw.githubusercontent.com/pipecat-ai/pipecat/main/pipecat.png">
</div></h1>

[![PyPI](https://img.shields.io/pypi/v/pipecat-ai)](https://pypi.org/project/pipecat-ai) ![Tests](https://github.com/pipecat-ai/pipecat/actions/workflows/tests.yaml/badge.svg) [![codecov](https://codecov.io/gh/pipecat-ai/pipecat/graph/badge.svg?token=LNVUIVO4Y9)](https://codecov.io/gh/pipecat-ai/pipecat) [![Docs](https://img.shields.io/badge/Documentation-blue)](https://docs.pipecat.ai) [![Discord](https://img.shields.io/discord/1239284677165056021)](https://discord.gg/pipecat)

Pipecat is an open source Python framework for building voice and multimodal conversational agents. It handles the complex orchestration of AI services, network transport, audio processing, and multimodal interactions, letting you focus on creating engaging experiences.

## What you can build

- **Voice Assistants**: [Natural, real-time conversations with AI](https://demo.dailybots.ai/)
- **Interactive Agents**: Personal coaches and meeting assistants
- **Multimodal Apps**: Combine voice, video, images, and text
- **Creative Tools**: [Story-telling experiences](https://storytelling-chatbot.fly.dev/) and social companions
- **Business Solutions**: [Customer intake flows](https://www.youtube.com/watch?v=lDevgsp9vn0) and support bots
- **Complex conversational flows**: [Refer to Pipecat Flows](https://github.com/pipecat-ai/pipecat-flows) to learn more

## See it in action

<p float="left">
    <a href="https://github.com/pipecat-ai/pipecat/tree/main/examples/simple-chatbot"><img src="https://raw.githubusercontent.com/pipecat-ai/pipecat/main/examples/simple-chatbot/image.png" width="280" /></a>&nbsp;
    <a href="https://github.com/pipecat-ai/pipecat/tree/main/examples/storytelling-chatbot"><img src="https://raw.githubusercontent.com/pipecat-ai/pipecat/main/examples/storytelling-chatbot/image.png" width="280" /></a>
    <br/>
    <a href="https://github.com/pipecat-ai/pipecat/tree/main/examples/translation-chatbot"><img src="https://raw.githubusercontent.com/pipecat-ai/pipecat/main/examples/translation-chatbot/image.png" width="280" /></a>&nbsp;
    <a href="https://github.com/pipecat-ai/pipecat/tree/main/examples/moondream-chatbot"><img src="https://raw.githubusercontent.com/pipecat-ai/pipecat/main/examples/moondream-chatbot/image.png" width="280" /></a>
</p>

## Key features

- **Voice-first Design**: Built-in speech recognition, TTS, and conversation handling
- **Flexible Integration**: Works with popular AI services (OpenAI, ElevenLabs, etc.)
- **Pipeline Architecture**: Build complex apps from simple, reusable components
- **Real-time Processing**: Frame-based pipeline architecture for fluid interactions
- **Production Ready**: Enterprise-grade WebRTC and Websocket support

💡 Looking to build structured conversations? Check out [Pipecat Flows](https://github.com/pipecat-ai/pipecat-flows) for managing complex conversational states and transitions.

## Getting started

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

### Available services

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

## Code examples

- [Foundational](https://github.com/pipecat-ai/pipecat/tree/main/examples/foundational) — small snippets that build on each other, introducing one or two concepts at a time
- [Example apps](https://github.com/pipecat-ai/pipecat/tree/main/examples/) — complete applications that you can use as starting points for development

## A simple voice agent running locally

Here is a very basic Pipecat bot that greets a user when they join a real-time session. We'll use [Daily](https://daily.co) for real-time media transport, and [Cartesia](https://cartesia.ai/) for text-to-speech.

```python
import asyncio

from pipecat.frames.frames import TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport

async def main():
  # Use Daily as a real-time media transport (WebRTC)
  transport = DailyTransport(
    room_url=...,
    token="", # leave empty. Note: token is _not_ your api key
    bot_name="Bot Name",
    params=DailyParams(audio_out_enabled=True))

  # Use Cartesia for Text-to-Speech
  tts = CartesiaTTSService(
    api_key=...,
    voice_id=...
  )

  # Simple pipeline that will process text to speech and output the result
  pipeline = Pipeline([tts, transport.output()])

  # Create Pipecat processor that can run one or more pipelines tasks
  runner = PipelineRunner()

  # Assign the task callable to run the pipeline
  task = PipelineTask(pipeline)

  # Register an event handler to play audio when a
  # participant joins the transport WebRTC session
  @transport.event_handler("on_first_participant_joined")
  async def on_first_participant_joined(transport, participant):
    participant_name = participant.get("info", {}).get("userName", "")
    # Queue a TextFrame that will get spoken by the TTS service (Cartesia)
    await task.queue_frame(TextFrame(f"Hello there, {participant_name}!"))

  # Register an event handler to exit the application when the user leaves.
  @transport.event_handler("on_participant_left")
  async def on_participant_left(transport, participant, reason):
    await task.cancel()

  # Run the pipeline task
  await runner.run(task)

if __name__ == "__main__":
  asyncio.run(main())
```

Run it with:

```shell
python app.py
```

Daily provides a prebuilt WebRTC user interface. While the app is running, you can visit at `https://<yourdomain>.daily.co/<room_url>` and listen to the bot say hello!

## WebRTC for production use

WebSockets are fine for server-to-server communication or for initial development. But for production use, you’ll need client-server audio to use a protocol designed for real-time media transport. (For an explanation of the difference between WebSockets and WebRTC, see [this post.](https://www.daily.co/blog/how-to-talk-to-an-llm-with-your-voice/#webrtc))

One way to get up and running quickly with WebRTC is to sign up for a Daily developer account. Daily gives you SDKs and global infrastructure for audio (and video) routing. Every account gets 10,000 audio/video/transcription minutes free each month.

Sign up [here](https://dashboard.daily.co/u/signup) and [create a room](https://docs.daily.co/reference/rest-api/rooms) in the developer Dashboard.

## Hacking on the framework itself

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

## Setting up your editor

This project uses strict [PEP 8](https://peps.python.org/pep-0008/) formatting via [Ruff](https://github.com/astral-sh/ruff).

### Emacs

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

### Visual Studio Code

Install the
[Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) extension. Then edit the user settings (_Ctrl-Shift-P_ `Open User Settings (JSON)`) and set it as the default Python formatter, and enable formatting on save:

```json
"[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true
}
```

### PyCharm

`ruff` was installed in the `venv` environment described before, now to enable autoformatting on save, go to `File` -> `Settings` -> `Tools` -> `File Watchers` and add a new watcher with the following settings:

1. **Name**: `Ruff formatter`
2. **File type**: `Python`
3. **Working directory**: `$ContentRoot$`
4. **Arguments**: `format $FilePath$`
5. **Program**: `$PyInterpreterDirectory$/ruff`

## Contributing

We welcome contributions from the community! Whether you're fixing bugs, improving documentation, or adding new features, here's how you can help:

- **Found a bug?** Open an [issue](https://github.com/pipecat-ai/pipecat/issues)
- **Have a feature idea?** Start a [discussion](https://discord.gg/pipecat)
- **Want to contribute code?** Check our [CONTRIBUTING.md](CONTRIBUTING.md) guide
- **Documentation improvements?** [Docs](https://github.com/pipecat-ai/docs) PRs are always welcome

Before submitting a pull request, please check existing issues and PRs to avoid duplicates.

We aim to review all contributions promptly and provide constructive feedback to help get your changes merged.

## Getting help

➡️ [Join our Discord](https://discord.gg/pipecat)

➡️ [Read the docs](https://docs.pipecat.ai)

➡️ [Reach us on X](https://x.com/pipecat_ai)
