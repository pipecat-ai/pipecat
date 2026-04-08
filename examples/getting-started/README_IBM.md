# Pipecat Quickstart with IBM Speech Services

This example demonstrates how to build a voice AI bot using IBM Speech-to-Text and Text-to-Speech services with Pipecat.

## Features

- **IBM Speech-to-Text**: Real-time speech recognition with WebSocket streaming
- **IBM Text-to-Speech**: Natural-sounding voice synthesis
- **Google Gemini LLM**: Conversational AI powered by Google's Gemini models
- **Silero VAD**: Voice Activity Detection for natural conversation flow

## Prerequisites

1. **IBM Cloud Account**: Sign up at https://cloud.ibm.com
2. **IBM Speech Services**:
   - Create a Speech-to-Text service instance
   - Create a Text-to-Speech service instance
3. **Google API Key**: Get from https://aistudio.google.com/app/apikey

## Setup

### 1. Install Pipecat with IBM Support

From the root of the pipecat repository:

```bash
# Install pipecat with IBM Speech Services support
pip install -e ".[ibm,google,silero,runner]"
```

Or using uv:

```bash
uv pip install -e ".[ibm,google,silero,runner]"
```

### 2. Configure Environment Variables

Copy the example environment file:

```bash
cd examples/getting-started
cp .env.example.ibm .env
```

Edit `.env` and add your credentials:

```bash
# IBM Speech-to-Text
IBM_STT_API_KEY=your-stt-api-key
IBM_STT_URL=https://api.us-south.speech-to-text.watson.cloud.ibm.com

# IBM Text-to-Speech
IBM_TTS_API_KEY=your-tts-api-key
IBM_TTS_URL=https://api.us-south.text-to-speech.watson.cloud.ibm.com

# Google Gemini
GOOGLE_API_KEY=your-google-api-key
```

### 3. Get IBM Speech Services Credentials

#### Speech-to-Text:
1. Go to https://cloud.ibm.com/catalog/services/speech-to-text
2. Create a new instance (Lite plan is free)
3. Go to "Manage" → "Credentials"
4. Copy the `apikey` and `url`

#### Text-to-Speech:
1. Go to https://cloud.ibm.com/catalog/services/text-to-speech
2. Create a new instance (Lite plan is free)
3. Go to "Manage" → "Credentials"
4. Copy the `apikey` and `url`

## Running the Bot

### Local Development

After installing pipecat with IBM support, run the bot:

```bash
# From the examples/getting-started directory
python bot_ibm.py
```

The bot will start and provide a URL to connect via your browser.

### Command Line Options

The bot supports various transport options:

```bash
# Use Daily.co transport (default)
python bot_ibm.py --transport daily

# Use local WebRTC
python bot_ibm.py --transport webrtc

# Use Twilio
python bot_ibm.py --transport twilio
```

### Using Daily.co Transport

```bash
# Set up Daily.co room
export DAILY_API_KEY=your-daily-api-key
export DAILY_ROOM_URL=https://your-domain.daily.co/your-room

# Run the bot
python bot_ibm.py --transport daily
```

## Customization

### Change IBM STT Model

Edit `bot_ibm.py` and modify the STT service initialization:

```python
stt = IBMSTTService(
    api_key=os.getenv("IBM_STT_API_KEY"),
    url=os.getenv("IBM_STT_URL"),
    model="en-US_Telephony",  # For phone audio
    params=IBMSTTService.InputParams(
        interim_results=True,
        smart_formatting=True,
    ),
)
```

Available models:
- `en-US_BroadbandModel` (default, 16kHz+)
- `en-US_Telephony` (8kHz phone audio)
- `en-GB_BroadbandModel` (British English)
- `es-ES_BroadbandModel` (Spanish)
- See full list: https://cloud.ibm.com/docs/speech-to-text?topic=speech-to-text-models

### Change IBM TTS Voice

Edit `bot_ibm.py` and modify the TTS service initialization:

```python
tts = IBMTTSService(
    api_key=os.getenv("IBM_TTS_API_KEY"),
    url=os.getenv("IBM_TTS_URL"),
    params=IBMTTSService.InputParams(
        voice="en-US_AllisonV3Voice",  # Female voice
        accept="audio/wav;rate=16000",
        rate_percentage=10,  # 10% faster
        pitch_percentage=5,  # Slightly higher pitch
    ),
)
```

Available voices:
- `en-US_MichaelV3Voice` (Male, default)
- `en-US_AllisonV3Voice` (Female)
- `en-US_LisaV3Voice` (Female)
- `en-GB_CharlotteV3Voice` (British Female)
- `es-ES_EnriqueV3Voice` (Spanish Male)
- See full list: https://cloud.ibm.com/docs/text-to-speech?topic=text-to-speech-voices

### Adjust Speaking Rate and Pitch

```python
params=IBMTTSService.InputParams(
    voice="en-US_MichaelV3Voice",
    rate_percentage=20,   # 20% faster (-100 to +100)
    pitch_percentage=-10, # Lower pitch (-100 to +100)
)
```

### Change Audio Format

```python
params=IBMTTSService.InputParams(
    accept="audio/ogg;codecs=opus",  # Opus codec
    # or "audio/mp3"
    # or "audio/flac"
)
```

## Architecture

```
User Audio Input
    ↓
IBM STT (WebSocket)
    ↓
User Aggregator (with Silero VAD)
    ↓
OpenAI LLM
    ↓
IBM TTS (WebSocket)
    ↓
Audio Output to User
```

## Troubleshooting

### Authentication Errors

**Problem**: `Failed to obtain IAM token`

**Solution**: 
- Verify your API keys are correct
- Check that the service URLs match your IBM Cloud region
- Ensure your IBM Cloud account is active

### Connection Issues

**Problem**: `Unable to connect to IBM STT/TTS`

**Solution**:
- Check your internet connection
- Verify the service URLs are correct
- Ensure firewall allows WebSocket connections

### Audio Quality Issues

**Problem**: Poor audio quality or choppy playback

**Solution**:
- Try different audio formats (wav, ogg, mp3)
- Adjust sample rate: `accept="audio/wav;rate=22050"`
- Check network bandwidth

### Rate Limits

IBM Lite plans have usage limits:
- **STT**: 500 minutes/month
- **TTS**: 10,000 characters/month

Upgrade to a paid plan for higher limits.

## Resources

- [IBM STT Documentation](https://cloud.ibm.com/docs/speech-to-text)
- [IBM TTS Documentation](https://cloud.ibm.com/docs/text-to-speech)
- [Pipecat Documentation](https://docs.pipecat.ai)
- [IBM Cloud Console](https://cloud.ibm.com)

## Support

For issues specific to:
- **Pipecat**: https://github.com/pipecat-ai/pipecat/issues
- **IBM Cloud**: https://cloud.ibm.com/docs/get-support

## License

BSD 2-Clause License - see LICENSE file for details.