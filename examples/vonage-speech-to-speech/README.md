# Vonage Chatbot (Pipecat)

Voice chatbot using **Pipecat AI** with **Vonage Audio Connector** over **WebSocket**.
This example uses OpenAI Realtime for speech-in → speech-out (no separate STT/TTS services). The server exposes a WS endpoint (via **VonageAudioConnectorTransport**) that you target from the Vonage **/connect API**, it streams the session’s audio into an OpenAI Realtime speech↔speech model and returns the synthesized reply to the caller in real time.


## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configure Vonage URLs](#configure-vonage-urls)
- [Configure Vonage Voice)](#configure-vonage-voice)
- [Running the Application](#running-the-application)
- [Usage](#usage)

## Features

- **Real-time WebSocket audio** to/from Vonage
- **OpenAI Realtime** native speech↔speech (no separate STT/TTS)
- **Silero VAD** for talk-pause detection
- **Dockerized** for easy deployment

## Requirements

- Python **3.12+**
- A **Vonage account**
- An **OpenAI API key**
- **ngrok** (or any HTTPS tunnel) for local testing
- Docker (optional)

## Installation

1. **Clone the repo and enter it**
    ```sh
    git clone https://github.com/opentok/vonage-pipecat.git
    cd vonage-pipecat/examples/vonage-speech-to-speech
    ```

2. **Set up a virtual environment** (optional but recommended):
    ```sh
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Create .env**:
    Copy the example environment file and update with your settings:

    ```sh
    cp env.example .env
    ```

5. **Set the OpenAI Key in .env**:
    ```sh
    OPENAI_API_KEY=
    #do not use the key with quotes like "" just use it without quotes like OPENAI_API_KEY=sk-proj-toaK2p....
    ```

6. **Install ngrok**:
   Follow the instructions on the [ngrok website](https://ngrok.com/download) to download and install ngrok.

## Configure Vonage URLs

1. **Start ngrok**:
    In a new terminal, start ngrok to tunnel the local server:

    ```sh
    ngrok http 8005
    #Copy the wss URL, e.g. "uri": "wss://<your-ngrok-domain>",
    ```

## Configure Vonage Voice
Use the Vonage Video API Playground (or your app) to create a session & token, then call /v2/project/{apiKey}/connect with your sessionId, token, and the WS uri from ngrok (set audioRate=16000, bidirectional=true).

## Running the Application

Choose one of these two methods to run the application:

### Using Python (Option 1)

**Run the Server application**:

    ```sh
    # Make sure you're in the project directory and your virtual environment is activated
    python server.py
    ```

### Using Docker (Option 2)

1. **Build the Docker image**:

    ```sh
    docker build -f examples/vonage-speech-to-speech/Dockerfile -t vonage-speech-to-speech .
    ```

2. **Run the Docker container**:
    ```sh
    docker run -it --rm -p 8005:8005 --env-file examples/vonage-speech-to-speech/.env vonage-speech-to-speech
    ```

The server will start on port 8005. Keep this running while you test with Vonage.

## Usage

Call the Connect API
Go to the examples/vonage-speech-to-speech/client/README.md and run the connect_and_stream.py.
Start publishing in Opentok Session (via Playground or your app) and then speak. Your audio will reach OpenAI Realtime speech↔speech model and returns the synthesized reply to the caller in real time.
