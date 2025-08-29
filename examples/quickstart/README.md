# Pipecat Quickstart

Run your first Pipecat bot in under 5 minutes. This example creates a voice AI bot that you can talk to in your browser.

## Prerequisites

### Python 3.10+

Pipecat requires Python 3.10 or newer. Check your version:

```bash
python --version
```

If you need to upgrade Python, we recommend using a version manager like `uv` or `pyenv`.

## Setup

1. Install dependencies

From the root of the `pipecat` repo, run:

```bash
uv sync --extra webrtc --extra silero --extra deepgram --extra google --extra respeecher --extra runner
```

2. Configure environment variables

Create a `.env` file:

```bash
cp env.example .env
```

Then, add your API keys:

```
DEEPGRAM_API_KEY=your_deepgram_api_key
GOOGLE_API_KEY=your_google_api_key
RESPEECHER_API_KEY=your_respeecher_api_key
```

4. Run the example

Run your bot using:

```bash
uv run examples/quickstart/bot.py
```

**Open http://localhost:7860 in your browser** and click `Connect` to start talking to your bot.

> 💡 First run note: The initial startup may take ~10 seconds as Pipecat downloads required models, like the Silero VAD model.

## Troubleshooting

- **Browser permissions**: Make sure to allow microphone access when prompted by your browser.
- **Connection issues**: If the WebRTC connection fails, first try a different browser. If that fails, make sure you don't have a VPN or firewall rules blocking traffic. WebRTC uses UDP to communicate.
- **Audio issues**: Check that your microphone and speakers are working and not muted.

## Next Steps

- **Read the docs**: Check out [Pipecat's docs](https://docs.pipecat.ai/) for guides and reference information.
- **Join Discord**: Join [Pipecat's Discord server](https://discord.gg/pipecat) to get help and learn about what others are building.
