# News Chatbot

A simple AI-powered chatbot that leverages Gemini's real-time search capabilities in a voice AI application.

This example demonstrates Gemini's ability to query Google search in real time and return relevant responses, including links to the URLs that Gemini searched.

All the details about grounding with Google Search can be found [here](https://ai.google.dev/gemini-api/docs/grounding?lang=python).

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
- Node.js 16+ (for JavaScript and React implementations)
- Daily API key
- Gemini API key (for Gemini bot)
- Cartesia API key
- Modern web browser with WebRTC support