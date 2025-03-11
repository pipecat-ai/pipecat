# Video transform

A Pipecat example demonstrating how to send and receive audio and video using SmallWebRTCTransport.
It also performs some image processing on the video frames using OpenCV.

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

For client-side setup, refer to the [JavaScript Guide](client/typescript/README.md).

## Important Note

Ensure the bot server is running before using any client implementations.

## Requirements

- Python 3.10+
- Node.js 16+ (for JavaScript)
- Google API key
- Modern web browser with WebRTC support