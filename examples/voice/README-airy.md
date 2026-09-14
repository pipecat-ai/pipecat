# Airy TTS

`AiryHttpTTSService` synthesizes Korean and English speech through the
[Airy streaming API](https://airy.so/cloud-api/docs/api/tts/speech-synthesis-stream).
It accepts complete text for each request and streams the resulting audio into a
Pipecat pipeline. No Airy SDK or additional provider dependency is required.

## Prerequisites

- A checkout containing `pipecat.services.airy`. The commands below run this
  checkout, not a separately published Airy package.
- An Airy API key and available credits. See the
  [Airy console](https://airy.so/cloud-api/console) and
  [authentication guide](https://airy.so/cloud-api/docs/guide/authentication).
- An OpenAI API key to run the conversation example. The Airy service itself
  does not require an OpenAI account.

## Run the conversation example

`voice-airy.py` combines OpenAI speech recognition, an OpenAI LLM, and Airy TTS.
It responds in Korean and supports the WebRTC and eval transports.

Add your keys to `.env`:

```dotenv
AIRY_API_KEY=your-airy-api-key
OPENAI_API_KEY=your-openai-api-key
```

Run from the repository root:

```bash
uv run --extra airy --extra runner --extra webrtc examples/voice/voice-airy.py -t webrtc
```

Open [the local client](http://localhost:7860/client), connect, and allow microphone
access. Ask a question in Korean, then speak again while the bot is answering.
The current answer should stop and the bot should respond to the new utterance.

The example sends microphone audio to OpenAI for transcription, conversation
text to OpenAI for responses, and synthesis text to Airy. Use non-sensitive
sample content when evaluating it. API calls consume credits with the respective
providers; the example does not set an account-level spending limit.

## Using the service

`AiryHttpTTSService` accepts an `aiohttp.ClientSession`. Keep the session open
while the pipeline runs and close it afterwards. Create it inside the async
function that runs your bot:

```python
import os

import aiohttp

from pipecat.services.airy.tts import AiryHttpTTSService
from pipecat.transcriptions.language import Language

async def run_bot():
    async with aiohttp.ClientSession() as session:
        tts = AiryHttpTTSService(
            api_key=os.environ["AIRY_API_KEY"],
            aiohttp_session=session,
            settings=AiryHttpTTSService.Settings(
                language=Language.KO,
                voice="a597bb7a98fc9ec1",
                style="normal",
            ),
        )
        # Create and run the pipeline here, while the HTTP session is open.
```

The defaults are `airy-tts-v1`, Silvia (`a597bb7a98fc9ec1`), English, and the
`normal` speaking style. Airy supports `en` and `ko`, and the styles `normal`,
`bright`, `calm`, and `whisper`. Find voice IDs in the
[Airy voice list](https://airy.so/cloud-api/docs/guide/voices).

| Argument | Purpose |
| --- | --- |
| `api_key` | Airy API key, sent as a Bearer token. |
| `aiohttp_session` | Caller-owned HTTP session; keep it open for the pipeline's lifetime. |
| `settings` | Runtime `model`, `voice`, `language`, and `style` values. |
| `sample_rate` | Output rate in Hz; defaults to the pipeline's configured rate. |
| `request_timeout` | Total time allowed for one HTTP request, including streaming the body; 30 seconds by default. |
| `base_url` | API origin, defaulting to `https://api.airy.so`, without `/v1`. Credentials are sent to this origin, so use only a trusted endpoint. |

Change model, voice, language, or style between requests with a settings frame:

```python
from pipecat.frames.frames import TTSUpdateSettingsFrame

await worker.queue_frame(
    TTSUpdateSettingsFrame(delta=AiryHttpTTSService.Settings(style="calm"))
)
```

## Request limits and error handling

The service calls `POST /v1/audio/speech/stream` and forwards audio as it arrives.
Airy returns 24 kHz, 16-bit little-endian mono PCM; the service converts it to the
pipeline's output sample rate. Each request accepts at most 1,280 characters.
Longer inputs produce an error rather than being truncated. This limit also
applies to text sent directly through a `TTSSpeakFrame`.

The request timeout defaults to 30 seconds and can be set with `request_timeout`.
API errors and timeouts are reported as nonfatal error frames. Requests are not
retried automatically. Pipecat marks the service unusable after authentication,
authorization, or invalid-request errors; these require application intervention.
Interruptions cancel the HTTP read and discard pending audio; server-side
cancellation and billing follow Airy's API behavior.

The integration uses the PCM streaming endpoint. It does not expose file output
formats, word timestamps, or a persistent server-side synthesis session. See the
[Airy streaming API](https://airy.so/cloud-api/docs/api/tts/speech-synthesis-stream)
for request details.

## Verification

The adapter tests use a local HTTP server and do not require API keys:

```bash
uv run --group dev pytest tests/test_airy_tts.py
```

These tests cover request settings, sample boundaries, resampling, delayed
responses, errors, metrics, and interruption cleanup. They do not assess voice
quality or establish a latency or availability guarantee.

For a behavioral eval, start the bot with the eval transport:

```bash
uv run --extra airy --extra runner --extra webrtc examples/voice/voice-airy.py -t eval --port 7860
```

Then run a scenario against `ws://localhost:7860`. Use Korean inputs and a
Korean-capable transcriber for audio-mode checks; the shared scenarios' default
audio models are English-only. See the
[release eval guide](../../scripts/release-evals/README.md) for model setup and
scenario configuration. Starting the bot with `-t eval` alone does not run or
validate a scenario.
