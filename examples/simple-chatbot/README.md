# Simple Chatbot

<img src="image.png" width="420px">

This repository demonstrates a simple AI chatbot with real-time audio/video interaction, implemented in three different ways. The bot server supports multiple AI backends, and you can connect to it using three different client approaches.

## Two Bot Options

1. **OpenAI Bot** (Default)

   - Uses gpt-4o for conversation
   - Requires OpenAI API key

2. **Gemini Bot**
   - Uses Google's Gemini Multimodal Live model
   - Requires Gemini API key

## Three Ways to Connect

1. **Daily Prebuilt** (Simplest)

   - Direct connection through a Daily Prebuilt room
   - For demo purposes only; handy for quick testing

2. **JavaScript**

   - Basic implementation using [Pipecat JavaScript SDK](https://docs.pipecat.ai/client/js/introduction)
   - No framework dependencies
   - Good for learning the fundamentals

3. **React**
   - Basic impelmentation using [Pipecat React SDK](https://docs.pipecat.ai/client/react/introduction)
   - Demonstrates the basic client principles with Pipecat React

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
   - Choose your bot implementation:
     ```ini
     BOT_IMPLEMENTATION=      # Options: 'openai' (default) or 'gemini'
     ```
5. Start the server:
   ```bash
   python server.py
   ```

### Next, connect using your preferred client app:

- [Daily Prebuilt](examples/prebuilt/README.md)
- [JavaScript Guide](examples/javascript/README.md)
- [React Guide](examples/react/README.md)

## Important Note

The bot server must be running for any of the client implementations to work. Start the server first before trying any of the client apps.

## Requirements

- Python 3.10+
- Node.js 16+ (for JavaScript and React implementations)
- Daily API key
- OpenAI API key (for OpenAI bot)
- Gemini API key (for Gemini bot)
- ElevenLabs API key
- Modern web browser with WebRTC support

## Project Structure

```
simple-chatbot/
├── server/              # Bot server implementation
│   ├── bot-openai.py    # OpenAI bot implementation
│   ├── bot-gemini.py    # Gemini bot implementation
│   ├── runner.py        # Server runner utilities
│   ├── server.py        # FastAPI server
│   └── requirements.txt
└── examples/            # Client implementations
    ├── prebuilt/        # Daily Prebuilt connection
    ├── javascript/      # Pipecat JavaScript client
    └── react/           # Pipecat React client
```
