<div align="center">
 <img alt="pipecat" width="300px" height="auto" src="https://raw.githubusercontent.com/pipecat-ai/pipecat/main/pipecat.png">
</div>

# Pipecat

[![PyPI](https://img.shields.io/pypi/v/pipecat-ai)](https://pypi.org/project/pipecat-ai) [![Discord](https://img.shields.io/discord/1239284677165056021)](https://discord.gg/pipecat) <a href="https://app.commanddash.io/agent/github_pipecat-ai_pipecat"><img src="https://img.shields.io/badge/AI-Code%20Agent-EB9FDA"></a>

`pipecat` is a framework for building voice (and multimodal) conversational agents. Things like personal coaches, meeting assistants, [story-telling toys for kids](https://storytelling-chatbot.fly.dev/), customer support bots, [intake flows](https://www.youtube.com/watch?v=lDevgsp9vn0), and snarky social companions.

Take a look at some example apps:

<p float="left">
    <a href="https://github.com/pipecat-ai/pipecat/tree/main/examples/simple-chatbot"><img src="https://raw.githubusercontent.com/pipecat-ai/pipecat/main/examples/simple-chatbot/image.png" width="280" /></a>&nbsp;
    <a href="https://github.com/pipecat-ai/pipecat/tree/main/examples/storytelling-chatbot"><img src="https://raw.githubusercontent.com/pipecat-ai/pipecat/main/examples/storytelling-chatbot/image.png" width="280" /></a>
    <br/>
    <a href="https://github.com/pipecat-ai/pipecat/tree/main/examples/translation-chatbot"><img src="https://raw.githubusercontent.com/pipecat-ai/pipecat/main/examples/translation-chatbot/image.png" width="280" /></a>&nbsp;
    <a href="https://github.com/pipecat-ai/pipecat/tree/main/examples/moondream-chatbot"><img src="https://raw.githubusercontent.com/pipecat-ai/pipecat/main/examples/moondream-chatbot/image.png" width="280" /></a>
</p>

## Getting started with voice agents

You can get started with Pipecat running on your local machine, then move your agent processes to the cloud when you’re ready. You can also add a 📞 telephone number, 🖼️ image output, 📺 video input, use different LLMs, and more.

```shell
# install the module
pip install pipecat-ai

# set up an .env file with API keys
cp dot-env.template .env
```

By default, in order to minimize dependencies, only the basic framework functionality is available. Some third-party AI services require additional dependencies that you can install with:

```shell
pip install "pipecat-ai[option,...]"
```

Your project may or may not need these, so they're made available as optional requirements. Here is a list:

- **AI services**: `anthropic`, `assemblyai`, `aws`, `azure`, `deepgram`, `gladia`, `google`, `fal`, `lmnt`, `moondream`, `openai`, `openpipe`, `playht`, `silero`, `whisper`, `xtts`
- **Transports**: `local`, `websocket`, `daily`

## Code examples

- [foundational](https://github.com/pipecat-ai/pipecat/tree/main/examples/foundational) — small snippets that build on each other, introducing one or two concepts at a time
- [example apps](https://github.com/pipecat-ai/pipecat/tree/main/examples/) — complete applications that you can use as starting points for development

## A simple voice agent running locally

Here is a very basic Pipecat bot that greets a user when they join a real-time session. We'll use [Daily](https://daily.co) for real-time media transport, and [Cartesia](https://cartesia.ai/) for text-to-speech.

```python
import asyncio

from pipecat.frames.frames import EndFrame, TextFrame
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
    await task.queue_frame(EndFrame())

  # Run the pipeline task
  await runner.run(task)

if __name__ == "__main__":
  asyncio.run(main())
```

Run it with:

```shell
python app.py
```

Daily provides a prebuilt WebRTC user interface. Whilst the app is running, you can visit at `https://<yourdomain>.daily.co/<room_url>` and listen to the bot say hello!

## WebRTC for production use

WebSockets are fine for server-to-server communication or for initial development. But for production use, you’ll need client-server audio to use a protocol designed for real-time media transport. (For an explanation of the difference between WebSockets and WebRTC, see [this post.](https://www.daily.co/blog/how-to-talk-to-an-llm-with-your-voice/#webrtc))

One way to get up and running quickly with WebRTC is to sign up for a Daily developer account. Daily gives you SDKs and global infrastructure for audio (and video) routing. Every account gets 10,000 audio/video/transcription minutes free each month.

Sign up [here](https://dashboard.daily.co/u/signup) and [create a room](https://docs.daily.co/reference/rest-api/rooms) in the developer Dashboard.

## What is VAD?

Voice Activity Detection &mdash; very important for knowing when a user has finished speaking to your bot. If you are not using press-to-talk, and want Pipecat to detect when the user has finished talking, VAD is an essential component for a natural feeling conversation.

Pipecat makes use of WebRTC VAD by default when using a WebRTC transport layer. Optionally, you can use Silero VAD for improved accuracy at the cost of higher CPU usage.

```shell
pip install pipecat-ai[silero]
```

## Hacking on the framework itself

_Note that you may need to set up a virtual environment before following the instructions below. For instance, you might need to run the following from the root of the repo:_

```shell
python3 -m venv venv
source venv/bin/activate
```

From the root of this repo, run the following:

```shell
pip install -r dev-requirements.txt
python -m build
```

This builds the package. To use the package locally (e.g. to run sample files), run

```shell
pip install --editable ".[option,...]"
```

If you want to use this package from another directory, you can run:

```shell
pip install "path_to_this_repo[option,...]"
```

### Running tests

From the root directory, run:

```shell
pytest --doctest-modules --ignore-glob="*to_be_updated*" --ignore-glob=*pipeline_source* src tests
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
  (setq lazy-ruff-only-format-block t)
  (setq lazy-ruff-only-format-region t)
  (setq lazy-ruff-only-format-buffer t))
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

## Getting help

➡️ [Join our Discord](https://discord.gg/pipecat)

➡️ [Reach us on X](https://x.com/pipecat_ai)
