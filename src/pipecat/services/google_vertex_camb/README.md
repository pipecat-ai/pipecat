# Google Vertex CAMB TTS Service

This service integrates CAMB.AI's MARS7 text-to-speech model deployed on Google Cloud Vertex AI with Pipecat.

## Features

- High-quality text-to-speech synthesis using MARS7 model
- Voice cloning capabilities with reference audio
- Multilingual support (en-us, es-es, etc.)
- Integration with Google Cloud Vertex AI

## Setup

1. **Install dependencies:**
   ```bash
   pip install -e ".[google-vertex-camb]"
   ```

2. **Set up Google Cloud credentials:**
   - Download your service account key file from Google Cloud Console
   - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to the key file

3. **Configure environment variables:**
   ```bash
   PROJECT_ID=cambai-public
   LOCATION=us-central1
   ENDPOINT_ID=your_endpoint_id_here
   GOOGLE_APPLICATION_CREDENTIALS=./path/to/service-account-key.json
   REFERENCE_AUDIO_PATH=./path/to/reference/audio.wav  # Optional
   REFERENCE_TEXT=Optional reference transcription     # Optional
   ```

## Usage

```python
from pipecat.services.google_vertex_camb.tts import GoogleVertexCambTTSService

# Basic usage
tts = GoogleVertexCambTTSService(
    project_id="cambai-public",
    location="us-central1", 
    endpoint_id="your_endpoint_id",
    credentials_path="./service-account-key.json"
)

# With voice cloning
params = GoogleVertexCambTTSService.InputParams(
    reference_audio_path="./reference_voice.wav",
    reference_text="This is my reference voice",
    language="en-us"
)

tts = GoogleVertexCambTTSService(
    project_id="cambai-public",
    location="us-central1",
    endpoint_id="your_endpoint_id", 
    credentials_path="./service-account-key.json",
    params=params
)

# Generate speech
async for frame in tts.run_tts("Hello, world!"):
    if hasattr(frame, 'audio'):
        # Process audio frame
        pass
```

## Testing

Run the test example:

```bash
python examples/foundational/01-say-one-thing-google-vertex-camb.py
```

Make sure to set the required environment variables before running the test.

## Requirements

- Google Cloud service account with Vertex AI permissions
- MARS7 model endpoint deployed on Vertex AI
- Reference audio file for voice cloning (optional)