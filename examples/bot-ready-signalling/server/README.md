# Bot ready signaling Server

A FastAPI server that manages bot instances and provide endpoint for Pipecat client connections.

## Endpoints

- `POST /connect` - Pipecat client connection endpoint

## Environment Variables

Copy `env.example` to `.env` and configure:

```ini
# Required API Keys
DAILY_API_KEY=           # Your Daily API key
CARTESIA_API_KEY=        # Your Cartesia API key

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
pip install --editable "../../../[daily,cartesia,openai]"
```

Run the server:

```bash
python server.py
```
