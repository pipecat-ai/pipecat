# Vonage Chatbot (Pipecat)

A real-time voice chatbot built using **Pipecat AI** with **Vonage Audio Connector** over **WebSocket**.
This project streams caller audio to **OpenAI STT**, processes the conversation using an LLM, converts the AI's response to speech via **OpenAI TTS**, and streams it back to the caller in real time. The server exposes a WebSocket endpoint (via **VonageAudioConnectorTransport**) that the Vonage **/connect API** connects to, bridging a live session into the **OpenAI STT → LLM → TTS** pipeline.


## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Expose Local Server with ngrok](#expose-local-server-with-ngrok)
- [Configure Vonage Voice)](#configure-vonage-voice)
- [Running the Application](#running-the-application)
- [Testing the Chatbot](#testing-the-chatbot)

## Features

- **Real-time WebSocket audio** to/from Vonage over WebSocket
- **OpenAI-powered pipeline** STT → LLM → TTS pipeline
- **Silero VAD** for accurate talk-pause detection
- **Dockerized** for easy deployment

## Requirements

- Python **3.10+**
- A **Vonage account**
- An **OpenAI API key**
- **ngrok** (or any HTTPS tunnel) for local testing
- Docker (optional)

## Installation

1. **Clone the repo and enter it**

    ```sh
    git clone https://github.com/opentok/vonage-pipecat.git
    cd vonage-pipecat/
    ```

2. **Set up a virtual environment** (recommended):

    ```sh
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    ```

3. **Install Pipecat AI (editable mode)**:

    ```sh
    pip install -e ".[openai,websocket,vonage,silero,runner]"
    ```

4. **Install example dependencies**:

    ```sh
    cd examples/vonage-chatbot
    pip install -r requirements.txt
    ```

5. **Create .env file**:

    Copy the example environment file and update with your settings:

    ```sh
    cp env.example .env
    ```

6. **Add your OpenAI Key to .env**:

    ```sh
    OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
    # Do not include quotes ("")
    ```

7. **Install ngrok**:

   Follow the instructions on the [ngrok website](https://ngrok.com/download) to download and install ngrok. You’ll use this to securely expose your local WebSocket server for testing.

## Expose Local Server with ngrok

1. **Start ngrok**:

    In a new terminal, start ngrok to tunnel the local server:

    ```sh
    ngrok http 8005
    #Copy the wss URL, e.g. "uri": "wss://<your-ngrok-domain>",
    ```

    You’ll see output like:

    ```sh
    Forwarding    https://a5db22f57efa.ngrok-free.app -> http://localhost:8005
    ```

    The https:// address is your public ngrok domain. To create the WebSocket Secure (WSS) URL for Vonage, simply replace https:// with wss://.

    Example:

    ```sh
    "websocket": {
        "uri": "wss://a5db22f57efa.ngrok-free.app",
        "audioRate": 16000,
        "bidirectional": true
    }
    ```

## Configure Vonage Voice
1. Open the **Vonage Video API Playground** (or your own application).
2. Create a new session and publish the stream.
3. Make a POST request to:
    ```sh
    /v2/project/{apiKey}/connect
    ```
4. Include the following in the JSON body:
    - sessionId
    - token
    - The WebSocket URI from ngrok (e.g. "wss://a5db22f57efa.ngrok-free.app")
    - "audioRate": 16000
    - "bidirectional": true
5. This connects your Vonage session to your locally running Pipecat WebSocket server through ngrok.
6. For a working example of the /connect API request, see [Testing the Chatbot](#testing-the-chatbot)

## Running the Application

Choose one of the following methods to start the chatbot server.

### Option 1: Run with Python

**Run the Server application**:

    ```sh
    # Ensure you're in the example directory (examples/vonage-chatbot) and your virtual environment is active
    python server.py
    ```

### Option 2: Run with Docker

1. **Build the Docker image**:

    ```sh
    docker build -f examples/vonage-chatbot/Dockerfile -t vonage-chatbot .
    ```

2. **Run the Docker container**:
    ```sh
    docker run -it --rm -p 8005:8005 --env-file examples/vonage-chatbot/.env vonage-chatbot
    ```

The server will start on port 8005. Keep this running while you test with Vonage.

## Testing the Chatbot

1. Start publishing audio in the Vonage Playground
2. Follow the examples/vonage-chatbot/client/README.md and run the connect_and_stream.py.
Once established then speak. Your audio will reach STT → LLM → TTS pipeline and you’ll hear AI-generated voice reply.
