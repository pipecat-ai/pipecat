# Simple Chatbot

This repository demonstrates a simple AI chatbot with real-time audio/video interaction, implemented in three different ways. The bot server remains the same, but you can connect to it using three different client approaches.

## Three Ways to Connect

1. **Daily Prebuilt** (Simplest)

   - Direct connection through a Daily Prebuilt room
   - For demo purposes only; handy for quick testing

2. **JavaScript**

   - Basic implementation using RTVI JavaScript SDK
   - No framework dependencies
   - Good for learning the fundamentals

3. **React**
   - Basic impelmentation using RTVI React SDK
   - Demonstrates the basic client principles with RTVI React

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
4. Copy env.example to .env and add your credentials

5. Start the server:
   ```bash
   python server.py
   ```

### Next, connect using your preferred client app:

- [Daily Prebuilt](examples/prebuilt/README.md)
- [Vanilla JavaScript Guide](examples/javascript/README.md)
- [React Guide](examples/react/README.md)

## Important Note

The bot server must be running for any of the client implementations to work. Start the server first before trying any of the client apps.

## Requirements

- Python 3.10+
- Node.js 16+ (for JavaScript and React implementations)
- Daily API key
- OpenAI API key
- Cartesia API key
- Modern web browser with WebRTC support

## Project Structure

```
simple-chatbot-full-stack/
├── server/             # Bot server implementation
│   ├── bot.py          # Bot logic and media handling
│   ├── runner.py       # Server runner utilities
│   ├── server.py       # FastAPI server
│   └── requirements.txt
└── examples/           # Client implementations
    ├── prebuilt/       # Daily Prebuilt connection
    ├── javascript/     # JavaScript RTVI client
    └── react/          # React RTVI client
```
