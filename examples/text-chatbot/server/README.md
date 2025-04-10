# Simple Chatbot Server

A FastAPI server that manages bot instances and provides endpoints for both Daily Prebuilt and Pipecat client connections.

## Endpoints

- `GET /` - Direct browser access, redirects to a Daily Prebuilt room
- `POST /connect` - Pipecat client connection endpoint
- `GET /status/{pid}` - Get status of a specific bot process

## Environment Variables

Copy `env.example` to `.env` and configure:

```ini
# Required API Keys
DAILY_API_KEY=           # Your Daily API key
OPENAI_API_KEY=          # Your OpenAI API key (required for OpenAI bot)

# Optional Configuration
DAILY_API_URL=           # Optional: Daily API URL (defaults to https://api.daily.co/v1)
DAILY_SAMPLE_ROOM_URL=   # Optional: Fixed room URL for development
HOST=                    # Optional: Host address (defaults to 0.0.0.0)
FAST_API_PORT=           # Optional: Port number (defaults to 7860)
```

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
pip install --editable "../../../[daily,openai]"
```

Run the server:

```bash
python server.py
```
