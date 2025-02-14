# Instant Voice

This demo showcases how to enable instant voice communication as soon as a user connects.
By leveraging optimizations on both the server and client sides, users can start speaking immediately after pressing the connect button.

## How It Works

### Server-Side Improvements:
- A **pool of Daily rooms** is managed to ensure quick connections.
- When a user connects, an existing room from the pool is assigned.
- A new room is created asynchronously to maintain the predefined pool size.

### Client-Side Improvements:
- Using the **DailyTransport** property `bufferLocalAudioUntilBotReady` set to enabled, users can start speaking immediately
  upon receiving the `AUDIO_BUFFERING_STARTED` event (typically within ~1s).
- This allows users to speak even before the bot is fully ready or the WebRTC connection is fully established.

## Quick Start

### 1. Start the Bot Server

1. Navigate to the server directory:
   ```bash
   cd server
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the `.env.example` file to `.env` and configure it:
    - Add your API keys.
5. Start the server:
   ```bash
   python src/server.py
   ```

### 2. Connect Using the Client App

For client-side setup, refer to the [JavaScript Guide](client/javascript/README.md).

## Important Notes
- The bot server **must** be running before using the client implementation.
- Ensure your environment variables are correctly set up.

## Requirements
- **Python 3.10+**
- **Node.js 16+** (for JavaScript/React client)
- **Daily API key**
- **Google API key**
- **Modern web browser with WebRTC support**
