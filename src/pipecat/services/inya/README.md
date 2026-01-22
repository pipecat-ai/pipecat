# Inya STT Service

Speech-to-Text (STT) service integration for Inya's multilingual transcription API, specializing in Indian languages.

## Features

- 10+ Indian languages support (Hindi, Tamil, Telugu, Kannada, and more)
- REST API based with segmented audio processing
- Dynamic language switching
- Built-in metrics support

## Supported Languages

- English (India) - `en-IN`
- Hindi - `hi-IN`
- Gujarati - `gu-IN`
- Tamil - `ta-IN`
- Kannada - `kn-IN`
- Telugu - `te-IN`
- Marathi - `mr-IN`
- Bengali - `bn-IN`
- Malayalam - `ml-IN`
- Punjabi - `pa-IN`

## Installation

```bash
pip install pipecat-ai[daily]
pip install aiohttp
```

## Setup

Get your API credentials from Inya and set environment variables:

```bash
export INYA_API_KEY="your-api-key"
export INYA_ORGANIZATION_ID="your-org-id"
export INYA_USER_ID="your-user-id"  # Optional
```

## Usage with Pipecat Pipeline

```python
import os
from pipecat.services.inya import InyaSTTService
from pipecat.transcriptions.language import Language
from pipecat.pipeline.pipeline import Pipeline

stt = InyaSTTService(
    api_key=os.getenv("INYA_API_KEY"),
    organization_id=os.getenv("INYA_ORGANIZATION_ID"),
    params=InyaSTTService.InputParams(
        language=Language.HI_IN,  # Hindi
        api_user_id=os.getenv("INYA_USER_ID", "pipecat-user"),
    ),
)

pipeline = Pipeline([
    transport.input(),
    stt,
    context_aggregator.user(),
    llm,
    tts,
    transport.output(),
    context_aggregator.assistant(),
])
```

## Running the Example

See the foundational example for a complete working bot:

```bash
# Set up environment
cd examples/foundational
cp env.inya.example .env
# Edit .env with your credentials

# Install dependencies
pip install pipecat-ai[daily,openai,cartesia]

# Run the bot with Daily transport
python 07x-interruptible-inya.py --transport daily
```

## Dynamic Language Switching

```python
from pipecat.transcriptions.language import Language

# Switch to Tamil during runtime
await stt.set_language(Language.TA_IN)
```

## Notes

- Requires VAD (Voice Activity Detection) like `SileroVADAnalyzer`
- Maximum audio duration per segment: 60 seconds
- Only final transcriptions (no interim results)

## Pipecat Version Compatibility

Tested with Pipecat v0.0.86+

## License

BSD 2-Clause License

