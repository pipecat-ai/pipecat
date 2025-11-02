## Summary

- add an HTTP-based `TypecastTTSService` with language mapping, metrics support, and optional per-request tuning via Pydantic params
- expose the service through `pipecat.services.typecast` and register a `typecast` extra so users can install dependencies explicitly
- include a foundational interruptible example (`07z-interruptible-typecast-http.py`) that wires Deepgram STT, OpenAI LLM, and the new TTS
- refresh environment samples so Typecast credentials can be configured alongside other providers

## What's New

### Typecast TTS Service

- new module `src/pipecat/services/typecast/tts.py` implements Typecast's REST API using a shared `aiohttp` session
- supports ISO-639 language conversion, prompt/output controls (emotion, intensity, pitch, tempo, format), and voice overrides
- request body mirrors the official [Typecast API reference](https://typecast.ai/docs/api-reference/endpoint/text-to-speech/text-to-speech), including ISO 639-3 language codes, `prompt` emotion settings, and `output` audio tuning fields
- integrates with existing TTS metrics (`start_ttfb_metrics`, `start_tts_usage_metrics`) and uses the common frame streaming helper for WAV payloads while keeping the API constrained to the `audio_format: "wav"` option the reference lists as compatible with raw PCM playback
- exported via `src/pipecat/services/typecast/__init__.py` to match other third-party provider packages

### Foundational Example

- adds `examples/foundational/07z-interruptible-typecast-http.py`, mirroring other HTTP-based TTS demos with transport selection, Deepgram STT, and OpenAI LLM
- demonstrates optional voice configuration through `TYPECAST_VOICE_ID` and wraps all network calls in a single `aiohttp.ClientSession`

### Configuration & Packaging

- registers a `typecast` optional dependency group in `pyproject.toml` for targeted installs
- documents `TYPECAST_API_KEY` (and optional `TYPECAST_VOICE_ID`) in `env.example` so new users know which secrets are required

## Usage Example

```python
async with aiohttp.ClientSession() as session:
    tts = TypecastTTSService(
        api_key=os.environ["TYPECAST_API_KEY"],
        aiohttp_session=session,
        params=TypecastTTSService.InputParams(
            prompt_options=PromptOptions(emotion_preset="happy", emotion_intensity=1.2),
        ),
    )
```

## Testing

- `python -m compileall src/pipecat/services/typecast/tts.py`
- `python -m compileall examples/foundational/07z-interruptible-typecast-http.py`
