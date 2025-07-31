# Pipecat Quickstart

Run your first Pipecat bot in under 5 minutes. This example creates a voice AI bot that you can talk to in your browser.

## Prerequisites

### Python 3.10+

Pipecat requires Python 3.10 or newer. Check your version:

```bash
python --version
```

If you need to upgrade Python, we recommend using a version manager like `uv` or `pyenv`.

### AI Service API keys

Pipecat orchestrates different AI services in a pipeline, ensuring low latency communication. In this quickstart example, we'll use:

- [Deepgram](https://console.deepgram.com/signup) for Speech-to-Text transcriptions
- [OpenAI](https://auth.openai.com/create-account) for LLM inference
- [Cartesia](https://play.cartesia.ai/sign-up) for Text-to-Speech audio generation

Have your API keys ready. We'll add them to your `.env` shortly.

## Setup

1. Set up a virtual environment

From the `examples/quickstart` directory, run:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

> Using `uv`? Create your venv using: `uv venv && source .venv/bin/activate`.

2. Install dependencies

```bash
pip install -r requirements.txt
```

> Using `uv`? Install requirements using: `uv pip install -r requirements.txt`.

3. Configure environment variables

Create a `.env` file:

```bash
cp env.example .env
```

Then, add your API keys:

```
DEEPGRAM_API_KEY=your_deepgram_api_key
OPENAI_API_KEY=your_openai_api_key
CARTESIA_API_KEY=your_cartesia_api_key
```

4. Run the example

Run your bot using:

```bash
python bot.py
```

> Using `uv`? Run your bot using: `uv run bot.py`.

**Open http://localhost:7860 in your browser** and click `Connect` to start talking to your bot.

> 💡 First run note: The initial startup may take ~10 seconds as Pipecat downloads required models, like the Silero VAD model.

## Troubleshooting

- **Browser permissions**: Make sure to allow microphone access when prompted by your browser.
- **Connection issues**: If the WebRTC connection fails, first try a different browser. If that fails, make sure you don't have a VPN or firewall rules blocking traffic. WebRTC uses UDP to communicate.
- **Audio issues**: Check that your microphone and speakers are working and not muted.

## Next Steps

- **Read the docs**: Check out [Pipecat's docs](https://docs.pipecat.ai/) for guides and reference information.
- **Join Discord**: Join [Pipecat's Discord server](https://discord.gg/pipecat) to get help and learn about what others are building.
