# Video transform

A Pipecat example demonstrating the simplest way to create a voice agent with SmallWebRTCTransport.

## Quick Start

### First, start the bot server:

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy env.example to .env and configure:
   - Add your API keys
4. Start the server:
   ```bash
   python server.py
   ```

### Next, connect using the client app:

Visit http://localhost:7860 in your browser.

## Requirements

- Python 3.10+
- Node.js 16+ (for JavaScript)
- Google API key
- Modern web browser with WebRTC support