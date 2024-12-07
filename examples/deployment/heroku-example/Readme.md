# Medical Intake Virtual Assistant

A virtual assistant system that conducts medical intake interviews using voice interactions. The system uses Daily.co for video/audio communication, OpenAI's GPT-4 for natural language processing, and Cartesia for text-to-speech conversion.

## Features

- Real-time voice interaction with patients
- Automated medical intake process
- Collection of:
  - Patient identity verification
  - Current prescriptions
  - Allergies
  - Medical conditions
  - Reason for visit
- Voice activity detection using Silero VAD
- Text-to-speech conversion for natural communication

## Required Files

Make sure these files are in your repository:

1. `Procfile` containing:
```
web: python server.py
```

2. `runtime.txt` containing:
```
python-3.9.x
```

3. `requirements.txt` with all dependencies

## Heroku Deployment

### Setup Steps

1. Go to [Heroku Dashboard](https://dashboard.heroku.com)
2. Click "New" → "Create new app"
3. Choose an app name and region
4. In the Deploy tab:
   - Connect to your GitHub repository
   - Enable Automatic Deploys (optional)
   - Click "Deploy Branch"

### Configuration

1. In the Settings tab:
   - Click "Reveal Config Vars"
   - Add these environment variables:
     ```
     DAILY_API_KEY=your_daily_api_key
     CARTESIA_API_KEY=your_cartesia_api_key
     OPENAI_API_KEY=your_openai_api_key
     ```

2. In the Resources tab:
   - Ensure the web dyno is enabled

### Monitoring

- View logs: "More" → "View logs"
- Restart app: "More" → "Restart all dynos"

## System Architecture

The system uses a pipeline with:
1. Daily Transport for audio/video communication
2. Context Aggregator for conversation management
3. LLM Service using GPT-4 for language processing
4. TTS Service with Cartesia for speech conversion
5. Frame Logger for system events

## Security Features

- One bot per room limit
- One-hour room token expiration
- Secure environment variable storage

