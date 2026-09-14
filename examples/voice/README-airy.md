# Airy TTS

`voice-airy.py` runs a Korean voice conversation with OpenAI speech recognition,
an OpenAI LLM, and Airy speech synthesis.

Add your keys to `.env`:

```dotenv
AIRY_API_KEY=your-airy-api-key
OPENAI_API_KEY=your-openai-api-key
```

Run from the repository root:

```bash
uv run --extra airy --extra runner --extra webrtc examples/voice/voice-airy.py
```

Open [the local client](http://localhost:7860/client), connect, and allow microphone
access. To check interruption handling, speak while the bot is answering. It
should stop that answer and respond to the new utterance.

## Using the service

`AiryHttpTTSService` accepts an `aiohttp.ClientSession`. Keep the session open
while the pipeline runs and close it afterwards:

```python
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

Change model, voice, language, or style between requests with a settings frame:

```python
await worker.queue_frame(
    TTSUpdateSettingsFrame(delta=AiryHttpTTSService.Settings(style="calm"))
)
```

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
formats or word timestamps. See the
[Airy streaming API](https://airy.so/cloud-api/docs/api/tts/speech-synthesis-stream)
for request details.

## Tests

The adapter tests use a local HTTP server and do not require API keys:

```bash
uv run --group dev pytest tests/test_airy_tts.py
```

For a behavioral eval, start the same bot with `-t eval --port 7860`, then run a
scenario against `ws://localhost:7860`. The
[release eval guide](../../scripts/release-evals/README.md) describes the harness
and audio-mode prerequisites.
