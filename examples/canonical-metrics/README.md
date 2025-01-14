# Chatbot with canonical-metrics

This project implements a chatbot using a pipeline architecture that integrates audio processing, transcription, and a language model for conversational interactions. The chatbot operates within a daily communication environment, utilizing various services for text-to-speech and language model responses.

## Features

- **Audio Input and Output**: Captures microphone input and plays back audio responses.
- **Voice Activity Detection**: Utilizes Silero VAD to manage audio input intelligently.
- **Text-to-Speech**: Integrates ElevenLabs TTS service to convert text responses into audio.
- **Language Model Interaction**: Uses OpenAI's GPT-4 model to generate responses based on user input.
- **Transcription Services**: Captures and transcribes participant speech for analytics.
- **Metrics Collection**: Sends audio data for analysis via Canonical Metrics Service.

## Requirements

- Python 3.10+
- `python-dotenv`
- Additional libraries from the `pipecat` package.

## Setup

1. Clone the repository.
2. Install the required packages.
3. Set up environment variables for API keys:
   - `OPENAI_API_KEY`
   - `ELEVENLABS_API_KEY`
   - `CANONICAL_API_KEY`
   - `CANONICAL_API_URL`
4. Run the script.

## Usage

The chatbot introduces itself and engages in conversations, providing brief and creative responses. Designed for flexibility, it can support multiple languages with appropriate configuration.

## Events

- Participants joining or leaving the call are handled dynamically, adjusting the chatbot's behavior accordingly.


ℹ️ The first time, things might take extra time to get started since VAD (Voice Activity Detection) model needs to be downloaded.

## Get started

```python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp env.example .env # and add your credentials

```

## Run the server

```bash
python server.py
```

Then, visit `http://localhost:7860/` in your browser to start a chatbot session.

## Build and test the Docker image

```
docker build -t chatbot .
docker run --env-file .env -p 7860:7860 chatbot
```
