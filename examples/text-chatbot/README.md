# Simple Text Chatbot

This repository demonstrates a simple AI chatbot with real-time audio/video interaction, implemented using different client and server options. The bot server supports multiple AI backends, and you can connect to it using five different client approaches.

## Bot

1. **OpenAI Bot**
   - Uses gpt-4o for conversation
   - Requires OpenAI API key

## Client

1. **JavaScript**
   - Basic implementation using [Pipecat JavaScript SDK](https://docs.pipecat.ai/client/js/introduction)
   - No framework dependencies

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

### start the client:

1. Navigate to the directory:
   ```bash
   cd client/javascript
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the client app:

   ```bash
   npm run dev
   ```

4. Visit http://localhost:5173 in your browser.

## Important Note

The bot server must be running for any of the client implementations to work. Start the server first before trying any of the client apps.

## Requirements

- Python 3.10+
- Node.js 16+ (for JavaScript implementations)
- Daily API key
- OpenAI API key (for OpenAI bot)
- Modern web browser with WebRTC support

## Project Structure

```
simple-chatbot/
├── server/              # Bot server implementation
│   ├── bot.py           # OpenAI bot implementation
│   ├── runner.py        # Server runner utilities
│   ├── server.py        # FastAPI server
│   └── requirements.txt
└── client/              # Client implementation
    ├── javascript/      # Daily JavaScript connection
```
