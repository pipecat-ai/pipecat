## Quickstart

### Setup

1. Set up a venv

2. Install packages

pip install "pipecat-ai[webrtc,deepgram,openai,cartesia,silero]" \
 "pipecat-ai-small-webrtc-prebuilt" \
 "python-dotenv"

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

```bash
python bot.py
```
