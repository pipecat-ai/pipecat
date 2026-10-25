# Rime Prism realtime

`RimeRealtimeLLMService` handles speech input, typed turns, transcripts, generated speech, and application tools in one Prism conversation. It uses the Rime SDK's direct session interface. Pipecat owns the pipeline, tool execution, and playback.

Install `pipecat-ai[rime-realtime]` and set `RIME_API_KEY`. You can also pass a caller-owned `rimelabs_sdk.Rime` client with `client=`. The service closes only clients it creates. Existing Rime TTS services use the separate `rime` extra.

Use one service instance per conversation and a shared `LLMContextAggregatorPair`:

```text
transport.input() → user aggregator → RimeRealtimeLLMService → transport.output() → assistant aggregator
```

For a complete bot, see [realtime-rime.py](../../examples/realtime/realtime-rime.py). This excerpt uses its `lookup_invoice` handler. Queue frames after adding `llm` to the worker's pipeline:

```python
from rimelabs_sdk.realtime import PrismFiller
from pipecat.frames.frames import LLMInputTextFrame, LLMUpdateSettingsFrame
from pipecat.services.rime.realtime.llm import RimeRealtimeLLMService

llm = RimeRealtimeLLMService(
    settings=RimeRealtimeLLMService.Settings(
        system_instruction="Handle invoice questions. Keep answers short.",
        tools=[lookup_invoice],
        tool_fillers={"lookup_invoice": PrismFiller(type="text", text="Checking your invoice.")},
    )
)
await worker.queue_frame(LLMInputTextFrame(text="Check invoice 123", audio_response=True))
await worker.queue_frame(
    LLMUpdateSettingsFrame(
        delta=llm.Settings(
            system_instruction="Handle billing questions.",
            tools=[lookup_invoice],
            truncation="auto",
        )
    )
)
```

## Connection and audio

Omit `endpoint` to use the SDK's default Prism endpoint. Custom deployments must support the SDK's authentication and realtime interface; the SDK validates the endpoint.

Input uses PCM16. The SDK converts 8, 16, 24, and 48 kHz mono or stereo input. The service downmixes and resamples other PCM16 input within 8 to 192 kHz and 1 to 32 channels. Configure mono output; tracked audio rejects incompatible output channels.

Prism detects turns on the server. Service metadata recommends external turn strategies, and application strategies still control interruption permission. Set `turn_detection=` with SDK `TurnDetection` timing options at construction. `interrupt_on_speech=False` disables interruption while keeping turn detection and speech announcements.

## Input and accepted operations

Live text uses `LLMInputTextFrame`, which submits one user turn and requests its reply. Optional fields are `user_id`, `audio_response`, and `client_message_id`. RTVI `send-text` uses this frame when Prism is active. Immediate input interrupts first; non-immediate input waits for SDK admission. Other providers keep their existing RTVI behavior.

Use `LLMRunFrame` to request a reply to current context. The append, update, and transform message frames edit history; `run_llm=True` requests a reply after acceptance. Initial history imports generate no reply and execute no tools.

Prism can reject an update, so Pipecat waits for acknowledgment before changing local context. Aggregators preserve operation intent when the service advertises `acknowledges_operations=True`. Each accepted operation commits once, preserving records that arrive during the wait.

Observe `LLMOperationResultFrame` or `on_operation_result` for the initiating frame's `request_id`, operation kind, service name, and outcome:

| Outcome | Meaning |
| --- | --- |
| `accepted` | Prism acknowledged the operation and Pipecat committed local state. |
| `rejected` | No change was accepted. The conversation remains usable. |
| `unknown` | The request's outcome is unknown. Recovery is required. |
| `partial` | Some history mutations committed before failure. Recovery is required. |

See [Limits and recovery](#limits-and-recovery) before retrying a failed operation. Request IDs identify operations; they do not make repeated submissions safe.

## Settings and tools

Omitted settings retain their values. Instructions, tools, fillers, voice, interruption policy, and history truncation share one SDK update and acknowledgment. `LLMSetToolsFrame` follows the same path. For a settings-only handoff, send instructions and tools in one delta to retain the conversation.

Tool fillers accept SDK `PrismFiller` values `none`, `text`, and `auto`. Automatic fillers can include generation instructions. Fillers must name tools in the resulting tool set. Removing a tool removes its filler; an empty tool list removes all tools.

Each reply retains its starting interruption policy and tool handlers. Live tools execute through Pipecat. Final results are recorded before continuation; if continuation has expired, the service can request a reply about the recorded result.

Response labels use a separate SDK operation with `operation="response_metadata"` in its result. A settings update can succeed while its metadata update fails.

## Diagnostics

Use these interfaces to inspect accepted state and correlate replies:

| Interface | Information |
| --- | --- |
| `session_info` | Accepted settings revision and the provider request ID actually used. |
| `on_session_updated` | Accepted SDK session information. |
| `on_rime_event` | SDK events, including response/input IDs, speech end reason, metadata, and faults. |
| `LLMFullResponseStartFrame.metadata["rime"]` | Provider request ID, response ID, input item IDs, settings revision, response metadata, input frame ID, and RTVI client message ID. |

Fault exceptions retain code, scope, retryability, and correlation. Retryability describes the request; it does not establish that the conversation can continue.

## History and truncation

Supported history contains user, assistant, and system text, plus completed function call/result pairs. Messages have a 4,000-character limit. Imported results must follow their calls. Invalid edits are rejected before the first mutation.

A transform that retains existing message objects can delete records while preserving server IDs. New dictionaries, changed content, or reordered records require a rebuild, even when text is identical. Rebuilds require no active replies or pending calls and can partially apply without atomic rollback.

Reading `await llm.get_history()` returns completed server history without replacing Pipecat's record. Active records can be absent from the snapshot. Microphone turns reserve their position before the final transcript arrives.

Context truncation controls future model history capacity. Startup and runtime settings accept `"auto"`, `"disabled"`, and SDK `RetentionRatio`. The setting is omitted until supplied and requires deployment support. It does not remove local records. Interrupted audio truncation is a separate playback operation.

## Playback and interruption

Each output has a globally unique `context_id`. `AudioOutputStartFrame` registers it, `AudioOutputEndFrame` closes it, and `AudioOutputStopFrame` stops its interruptible audio. Lifecycle markers survive interruption, and identity is preserved through resampling, chunking, filters, and mixing. Speech counts exclude padding and background audio.

Reports travel upstream as `AudioPlaybackReportFrame`, once per output and destination, including disabled output. They reach the original Rime owner after a service switch. Prism permits one authoritative destination. Duplicate reports are ignored, and finished receipts wait for every output, including empty responses.

| Transport path | Timing guarantee |
| --- | --- |
| `LocalAudioTransportParams(audio_out_playback_clock=True)` | Measures speech using PortAudio callback DAC times and the stream clock. Underflow or invalid timing produces unknown timing. |
| Other `BaseOutputTransport` paths, including eval and WebRTC | Reports byte acceptance with `played_ms=None` after audio is written. Listener playback is unknown. |
| Disabled or fully discarded output | Reports known zero playback. |

When timing is unknown, the SDK/server can estimate playback. A measured caller interruption reports per-output audio truncation. Application stops cancel generation and report finished playback. Uninterruptible audio stays queued and counts in the terminal report.

Speech hints lower output gain without cancelling the reply. The `SpeechHint` defaults are gain `0.25` and expiry `0.6` seconds. Repeated hints extend expiry; confirmed speech, shutdown, errors, and expiry remove only the temporary multiplier. Application volume and gain stay independent. Audio already submitted to a device or remote buffer can retain its earlier gain.

## Limits and recovery

Prism does not support disabling server turn detection, manual audio commit, video, image history, forced or disabled tool selection, per-reply tool overrides, or intermediate tool results. Use a separate TTS service for arbitrary application-directed speech.

Queues and retained state have explicit limits. Overflow makes the conversation unusable. The SDK retains 128 responses; unresolved receipts are never silently evicted. End and cancel share a bounded cleanup deadline. Pausing microphone capture leaves the conversation open.

A request can be accepted even if its acknowledgment is lost. Do not resend it automatically. The service does not automatically retry input, settings, tool execution, or history mutations.

Rejected admission before submission permits an explicit new request. Unknown outcomes and partially applied history edits require an application recovery decision.

For restart or failover, create a fresh service with accepted settings and supported completed context. Pending tool calls cannot resume in a fresh conversation. Old playback reports stay with their original owner.

Missing or expired settings revisions can also require recovery. Without revisions, a policy or handler change can make later response ownership ambiguous.

## Example and validation

Run the [example](../../examples/realtime/realtime-rime.py) with the development runner. It includes invoice lookup and a billing handoff that updates instructions and tools together. Initial system history requests the greeting. The typed scenario needs only `RIME_API_KEY`:

```sh
uv run --extra rime-realtime --extra runner examples/realtime/realtime-rime.py --port 7860
uv run pipecat eval run evals/release/scenarios/scripted/rime_prism.yaml --bot-url ws://localhost:7860/ws -v
```

Audio scenarios use Kokoro for caller speech, Moonshine to transcribe replies, and a Typesafe judge. Install those optional dependencies and set `TYPESAFE_API_KEY` for the judge. The scenario disables the optional Ollama explainer.

```sh
PYTHONPATH=. uv run pipecat eval run evals/release/scenarios/scripted/rime_prism_audio.yaml \
  --bot-url ws://localhost:7860/ws --audio --record-dir /tmp/rime-recordings -v
```

These scenarios cover memory, interruption followed by another turn, tools, and handoff. See the [validation report](rime-realtime-validation.md) for observed behavior and playback measurement limits.
