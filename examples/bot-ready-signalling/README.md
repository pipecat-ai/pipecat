# Bot ready signaling

A simple Pipecat example demonstrating how to handle signaling between the client and the bot, 
ensuring that the bot starts sending audio only when the client is available, 
thereby avoiding the risk of cutting off the beginning of the audio.

## Quick Start

### First, start the bot server:

1. Navigate to the server directory:
   ```bash
   cd server
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy env.example to .env and configure:
   - Add your API keys
5. Start the server:
   ```bash
   python server.py
   ```

### Next, connect using the client app:

For client-side setup, refer to the [JavaScript Guide](client/javascript/README.md).

## Important Note

Ensure the bot server is running before using any client implementations.

## Requirements

- Python 3.10+
- Node.js 16+ (for JavaScript)
- Daily API key
- Cartesia API key
- Modern web browser with WebRTC support