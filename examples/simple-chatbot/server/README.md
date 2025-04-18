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
GEMINI_API_KEY=          # Your Gemini API key (required for Gemini bot)
ELEVENLABS_API_KEY=      # Your ElevenLabs API key

# Bot Selection
BOT_IMPLEMENTATION=      # Options: 'openai' or 'gemini'

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

2. **Gemini Bot**
   - Uses Google's Gemini model
   - Requires GEMINI_API_KEY

Select your preferred bot by setting `BOT_IMPLEMENTATION` in your `.env` file.

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
pip install --editable "../../../[daily,elevenlabs,openai,silero,google]"
```

Run the server:

```bash
python server.py
```

## Troubleshooting

If you encounred this error:

```bash
aiohttp.client_exceptions.ClientConnectorCertificateError: Cannot connect to host api.daily.co:443 ssl:True [SSLCertVerificationError: (1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)')]
```

It's because Python cannot verify the SSL certificate from https://api.daily.co when making a POST request to create a room or token.

This is a common issue when the system doesn't have the proper CA certificates.

Install SSL Certificates (macOS): `/Applications/Python\ 3.12/Install\ Certificates.command`
