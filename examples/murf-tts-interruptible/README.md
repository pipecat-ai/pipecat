# Murf TTS Interruptible Example

This example demonstrates how to create an interruptible voice chat bot using:
- Murf for Text-to-Speech (TTS)
- Deepgram for Speech-to-Text (STT)
- OpenAI for Language Model (LLM)
- Local audio transport with Silero VAD for voice activity detection

## Setup

1. Copy `.env.example` to `.env`
2. Fill in your API keys:
   - Get a Murf API key from [Murf.ai](https://murf.ai)
   - Get a Deepgram API key from [Deepgram](https://deepgram.com)
   - Get an OpenAI API key from [OpenAI](https://openai.com)

## Running the Example

```bash
python bot.py
```

The bot will:
1. Listen for your voice input using your microphone
2. Convert speech to text using Deepgram
3. Process the text with OpenAI's LLM
4. Convert the response to speech using Murf TTS
5. Play the audio response through your speakers

The conversation is interruptible - you can start speaking while the bot is talking to interrupt it. 