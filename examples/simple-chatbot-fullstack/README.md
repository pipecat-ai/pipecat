# Simple Chatbot Full Stack

A full-stack implementation of an AI chatbot with real-time audio/video interaction.

## Structure

- `server/` - Python-based bot server using FastAPI
- `client/` - JavaScript client using RTVI and Daily.co for WebRTC

## Setup

### Server Setup

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

### Client Setup

1. Navigate to the client directory:
   ```bash
   cd client
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Open the URL shown in the terminal (usually http://localhost:5173)

## Usage

1. Start the server (it will run on port 7860)
2. Start the client server (it will run on port 5173)
3. Open http://localhost:5173 in your browser
4. Click "Connect" to start a session with the bot

## Requirements

- Python 3.10+
- Node.js 14+
- Modern web browser with WebRTC support
