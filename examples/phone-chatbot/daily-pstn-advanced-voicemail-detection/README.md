# Daily PSTN Advanced Voicemail Detection Bot

This project demonstrates an improved voice bot that uses Daily's PSTN capabilities to make calls to phone numbers with intelligent voicemail detection and human conversation handling. Unlike the previous version with sequential pipelines, this implementation uses a **single parallel pipeline architecture** for better performance and simpler state management.

## Key Improvements

- **Parallel Pipeline Architecture**: Single pipeline with parallel branches instead of sequential pipeline switching
- **Simplified State Management**: Uses simple mode constants instead of complex state classes
- **Improved Audio Handling**: Better audio buffering and VAD prebuffering for cleaner speech detection
- **Enhanced Flow Management**: Uses Pipecat Flows for more structured human conversation handling
- **Better Resource Management**: More efficient processor design with conditional frame filtering

## How it works

1. **Call Initiation**: Server receives a request with phone number and creates a Daily room with SIP capabilities
2. **Dial-out Process**: Bot joins room and initiates call to the provided phone number
3. **Parallel Processing**: Single pipeline with three parallel branches:
   - **Voicemail Detection Branch**: Uses Gemini Flash Lite for fast voicemail pattern recognition
   - **Voicemail Response Branch**: Handles leaving voicemail messages when detected
   - **Human Conversation Branch**: Uses Gemini Flash with Pipecat Flows for natural conversations
4. **Intelligent Switching**: Bot automatically routes to appropriate branch based on detection confidence
5. **Clean Termination**: Proper cleanup and call termination handling

## Architecture Overview

### Pipeline Structure

```
Transport Input
    ↓
Parallel Pipeline:
├── Voicemail Detection Branch
│   ├── VAD Prebuffer Processor
│   ├── Audio Collector
│   ├── Detection Context Aggregator
│   └── Detection LLM (Gemini Flash Lite)
├── Voicemail Response Branch
│   └── Voicemail TTS
└── Human Conversation Branch
    ├── Block Audio Frames (conditional)
    ├── Speech-to-Text
    ├── Human Context Aggregator
    ├── Human LLM (Gemini Flash)
    └── Human TTS
    ↓
Transport Output
```

### Key Components

- **VADPrebufferProcessor**: Buffers audio frames before speech detection to prevent cutoff
- **BlockAudioFrames**: Conditionally blocks audio based on current mode (MUTE/VOICEMAIL/HUMAN)
- **FlowManager**: Handles structured conversation flows for human interaction
- **VoicemailDetectionObserver**: Monitors voicemail speaking patterns

## Prerequisites

- Daily account with API key and purchased phone number
- US phone number to call
- Dial-out enabled on your domain ([request here](https://docs.daily.co/guides/products/dial-in-dial-out#main))
- Google API key for LLM services
- Cartesia API key for text-to-speech
- Deepgram API key for speech-to-text

## Setup

1. **Create virtual environment and install dependencies**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment variables**

```bash
cp .env.example .env
# Edit .env with your API keys:
# DAILY_API_KEY=your_daily_api_key
# GOOGLE_API_KEY=your_google_api_key
# CARTESIA_API_KEY=your_cartesia_api_key
# DEEPGRAM_API_KEY=your_deepgram_api_key
```

3. **Purchase phone number**

Follow [Daily's phone number documentation](https://docs.daily.co/reference/rest-api/phone-numbers/buy-phone-number)

4. **Request dial-out enablement**

Submit the form at the [dial-out documentation page](https://docs.daily.co/guides/products/dial-in-dial-out#main)

## Running the Bot

1. **Start the webhook server**

```bash
python server.py
```

2. **Test the bot**

```bash
curl -X POST "http://127.0.0.1:7860/start" \
  -H "Content-Type: application/json" \
  -d '{
    "dialout_settings": {
      "phone_number": "+12345678910"
    }
  }'
```

## Testing Scenarios

### Voicemail Detection

Say phrases like:

- "You've reached [name]'s voicemail. Please leave a message after the beep."
- "Sorry I missed your call. Leave your name and number."
- "This is [name]. I'm not available right now."

**Expected behavior**: Bot detects voicemail, waits for greeting to finish, leaves message, and terminates call.

### Human Conversation

Say phrases like:

- "Hello?"
- "Hi, who is this?"
- "[Company name], how can I help you?"

**Expected behavior**: Bot detects human, switches to conversation mode, and engages in natural dialogue.

## Configuration Options

### Detection Confidence Thresholds

```python
VOICEMAIL_CONFIDENCE_THRESHOLD = 0.6
HUMAN_CONFIDENCE_THRESHOLD = 0.6
```

### VAD Parameters

```python
VADParams(
    start_secs=0.1,      # How quickly to detect speech start
    confidence=0.4,      # VAD confidence threshold
    min_volume=0.4       # Minimum volume threshold
)
```

### Voicemail Message

Customize the message left on voicemail systems:

```python
message = "Hello, this is a message for Pipecat example user. This is Chatbot. Please call back on 123-456-7891. Thank you."
```

## Key Features

### Intelligent Detection

- **Confidence Scoring**: LLM provides confidence scores for voicemail/human detection
- **Reasoning Logs**: Detailed explanations for detection decisions
- **Adaptive Thresholds**: Configurable confidence thresholds for different scenarios

### Audio Processing

- **Prebuffering**: Prevents speech cutoff at conversation start
- **Conditional Blocking**: Blocks audio processing based on current mode

### Flow Management

- **Structured Conversations**: Uses Pipecat Flows for human interactions
- **Node-based Logic**: Greeting → Conversation → End flow progression
- **Function Callbacks**: Clean separation of conversation handling logic

### Error Handling

- **Retry Logic**: Automatic dialout retry on failures
- **Graceful Degradation**: Proper cleanup on errors or early termination
- **Timeout Management**: 90-second idle timeout with proper cancellation

## Troubleshooting

### Detection Issues

- **Bot doesn't detect voicemail**: Check if your voicemail follows common patterns in the system prompt
- **False human detection**: Increase `HUMAN_CONFIDENCE_THRESHOLD` or refine detection prompt
- **Audio cutoff**: Adjust `VADParams` or `prebuffer_frame_count`

### Connection Issues

- **Dialout fails**: Verify phone number format and dial-out enablement
- **No audio**: Check Cartesia API key and voice ID configuration
- **STT not working**: Verify Deepgram API key and audio format

### Performance Issues

- **Slow detection**: Ensure using Gemini Flash Lite for detection
- **High latency**: Check network connectivity and API response times
- **Memory usage**: Monitor audio buffer sizes and frame processing

## Monitoring and Logging

The bot provides comprehensive logging for:

- Detection decisions with confidence scores
- Mode transitions and state changes
- Audio processing and buffering status
- Flow progression and function calls
- Error conditions and retry attempts

Enable debug logging for detailed troubleshooting:

```python
logger.add(sys.stderr, level="DEBUG")
```
