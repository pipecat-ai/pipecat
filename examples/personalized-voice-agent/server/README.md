# Personalized Voice Agent Server

A FastAPI server that manages bot instances and provides endpoints for both Daily Prebuilt and Pipecat client connections.

## Environment Variables

Copy `env.example` to `.env` and configure:

```ini
# Required API Keys
DAILY_API_KEY=           # Your Daily API key
MEM0_API_KEY=            # Your Mem0 API key
OPENAI_API_KEY=          # Your OpenAI API key (required for OpenAI bot)
ELEVENLABS_API_KEY=      # Your ElevenLabs API key

# Optional Configuration
DAILY_API_URL=           # Optional: Daily API URL (defaults to https://api.daily.co/v1)
DAILY_SAMPLE_ROOM_URL=   # Optional: Fixed room URL for development
HOST=                    # Optional: Host address (defaults to 0.0.0.0)
FAST_API_PORT=           # Optional: Port number (defaults to 7860)
```

## Available Bots

The server supports two bot implementations:

1. **OpenAI Bot** (Default)

   - Uses GPT-4 for conversation
   - Requires OPENAI_API_KEY

## Running the Server

Set up and activate your virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you want to use the local version of `pipecat` in this repo rather than the last published version, also run:

```bash
pip install --editable "../../../[daily,elevenlabs,openai,silero,mem0ai]"
```

Run the server:

```bash
python server.py
```
