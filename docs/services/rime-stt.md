# Rime speech recognition

`RimeSTTService` streams each VAD speech segment to Rime and emits interim and final transcripts. `RimeSegmentedSTTService` waits for a complete segment and emits its final transcript. Both use the public Rime Python SDK.

## Installation

```sh
uv add 'pipecat-ai[rime-stt]'
```

Set `RIME_API_KEY` in the environment. The `rime-stt` extra installs `rimelabs-sdk>=0.1.0a20,<0.2`; the separate `rime` extra provides Rime TTS. Install `pipecat-ai[rime-stt,rime]` when using both.

## Pipeline and VAD

Use one VAD source per pipeline. In a voice agent, the user aggregator can supply VAD events upstream:

```text
transport.input() → Rime STT → user aggregator with VAD → LLM → TTS → transport.output() → assistant aggregator
```

For transcription alone, place a `VADProcessor` before the service:

```text
transport.input() → VADProcessor → Rime STT → transport.output()
```

Both services require `VADUserStartedSpeakingFrame` and `VADUserStoppedSpeakingFrame`. Without these events, they retain only an idle audio prefix and send no recognition requests. VAD stop ends the audio upload; the server returns the final transcript.

## Configuration

```python
from pipecat.services.rime.stt import RimeSTTService

stt = RimeSTTService(
    settings=RimeSTTService.Settings(
        language="en",
        mode="written",
        context_terms=["Pipecat", "Rime"],
        output_contract="revised_hypotheses",
    ),
)
```

| Setting | Default | Behavior |
| --- | --- | --- |
| `language` | `"en"` | BCP-47 tag or Pipecat `Language`; region tags are preserved. |
| `mode` | `"written"` | `"written"` or `"verbatim"`. |
| `context_terms` | Empty sequence | Recognition hints, copied when supplied. |
| `output_contract` | `"revised_hypotheses"` | Streaming only; also accepts `"append_fragments"`. |

Both streaming contracts produce cumulative `InterimTranscriptionFrame` text followed by one `TranscriptionFrame` per successful segment. Rime chooses the recognition model; model overrides and arbitrary extra settings are unsupported.

Send `STTUpdateSettingsFrame(delta=stt.Settings(...))` to update settings. Updates apply to subsequent speech segments; an active or queued segment retains its original settings and speaker identity.

Both constructors accept:

| Parameter | Default | Behavior |
| --- | --- | --- |
| `api_key` | `None` | Uses `RIME_API_KEY` when no client is supplied. |
| `client` | `None` | Optional caller-owned `rimelabs_sdk.Rime` instance. Mutually exclusive with `api_key`. |
| `request_timeout` | `120` | Positive finite seconds per request; streaming includes speaking and upload time. Server deadlines still apply. |
| `shutdown_timeout` | `5.0` | Positive finite seconds to drain pending recognition on `EndFrame` before cancellation. |
| `sample_rate` | `None` | Optional expected input rate; mismatching audio is rejected. |

Create an injected client on the pipeline's event loop and keep it open until the worker finishes. The service cancels its own recognition requests but never closes an injected client. Without injection, it creates the SDK client during startup and closes it on shutdown or cleanup.

## Audio, turns, and errors

Input is PCM16 little-endian at 8, 16, 24, or 48 kHz, with one or two channels. Streaming passes the actual format to the SDK, which converts to 16 kHz mono. Segmented recognition converts the completed segment to 16 kHz mono WAV before submission. A rate, channel count, or speaker change during a segment discards that segment and reports an error.

Streaming permits later segments to upload while earlier results are pending. Transcripts are emitted in speech order. An older final has `finalized=False`; only the latest completed segment, with no active speech, can signal finalization to Pipecat's user turn handler. Empty successful finals can settle that handler without adding an empty user message.

Buffers are bounded: four pending segments, 120 seconds per segment, and five seconds of unsent audio per streaming request. Overflow discards the affected segment and emits an error. `STTMuteFrame(mute=True)` discards pending speech; unmuting during speech resumes on fresh audio. `InterruptionFrame` preserves recognition so the caller can interrupt a spoken reply. `CancelFrame` cancels pending work promptly.

Errors use Pipecat categories and include a request ID when available. SDK error messages are not echoed. Transcript `result` includes `request_id` and the provider language tag. Enable metrics on the worker to receive final-transcript latency and audio usage; segmented requests also report processing time. `TranscriptionLogObserver` logs interim and final frames.

## Run the examples

From the repository root, install the standard development environment:

```sh
uv sync --group dev --all-extras --no-extra gstreamer --no-extra local
```

With `RIME_API_KEY` set, run the [transcription example](../../examples/transcription/transcription-rime.py):

```sh
# Streaming replacement hypotheses
uv run --no-sync python examples/transcription/transcription-rime.py -t webrtc

# Streaming append fragments
RIME_STT_CONTRACT=append_fragments uv run --no-sync python examples/transcription/transcription-rime.py -t webrtc

# Completed speech segments
RIME_STT_SEGMENTED=1 uv run --no-sync python examples/transcription/transcription-rime.py -t webrtc

# Spanish streaming recognition
RIME_STT_LANGUAGE=es uv run --no-sync python examples/transcription/transcription-rime.py -t webrtc
```

For the [voice agent](../../examples/voice/voice-rime-stt.py), also set `OPENAI_API_KEY`:

```sh
uv run --no-sync python examples/voice/voice-rime-stt.py -t webrtc
```

Run one example at a time. Open <http://localhost:7860/client/>, click **Connect**, and allow microphone access. Transcription appears in the terminal; the voice agent replies aloud. Disconnect ends the conversation. Press **Ctrl+C** in the server terminal to stop the server.

See [validation and audio evals](rime-stt-validation.md) for repeatable checks.
