# Rime Prism realtime

`RimeRealtimeLLMService` uses one Rime SDK conversation for microphone audio, live typed turns, transcripts, replies, and application tools. Install `pipecat-ai[rime-realtime]`. Set `RIME_API_KEY`, or pass a caller-owned `rimelabs_sdk.Rime` client with `client=`. The service closes clients that it creates. Existing Rime TTS services use the separate `rime` extra.

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

Put the service between the user aggregator and the output transport. Put the assistant aggregator after the output transport. Use a shared `LLMContextAggregatorPair`. Prism requires server turn detection. Its service metadata recommends external turn strategies. Application strategy settings still control interruption permission. `turn_detection=` accepts SDK `TurnDetection` timing settings at construction only. `interrupt_on_speech=False` keeps turn detection and speech announcements enabled.

Omit `endpoint` for the SDK's Prism endpoint. Pass an endpoint only for a deployment that supports the SDK's authentication and realtime API. Endpoint validation belongs to the SDK. Use one service instance per conversation. Configure mono output. Tracked audio rejects incompatible output channel settings. Input uses PCM16. The SDK converts 8, 16, 24, and 48 kHz mono or stereo input. The service downmixes and resamples other PCM16 input within 8 to 192 kHz and 1 to 32 channels.

## Input and accepted operations

`LLMInputTextFrame` submits one live user turn and requests its reply. It carries optional `user_id`, `audio_response`, and `client_message_id` values. RTVI `send-text` uses this frame when Prism is active. The immediate option interrupts first and preserves pipeline order. A non-immediate Prism typed turn waits for SDK admission and still requests a reply. Other providers retain their existing RTVI behavior.

`LLMRunFrame` requests a reply to accepted context. `LLMMessagesAppendFrame`, `LLMMessagesUpdateFrame`, and `LLMMessagesTransformFrame` edit history. They do not submit a typed turn. Set `run_llm=True` to request a reply after the history edit is accepted. Initial history imports do not execute tools or generate a reply.

The aggregators capture these operations when a service advertises `acknowledges_operations=True`. They do not change shared context before provider acceptance. The conversation owner commits each accepted operation once. Records that arrive during a history edit remain in context.

Observe `LLMOperationResultFrame` or `on_operation_result`. Results carry the initiating frame's `request_id`, the operation kind, the service name, and one outcome:

| Outcome | Meaning |
| --- | --- |
| `accepted` | The provider acknowledged the operation and the service committed its local state. |
| `rejected` | No provider change was accepted. Accepted state remains available. |
| `unknown` | Submission or acknowledgment failed without a known result. The conversation is unusable. |
| `partial` | Some history mutations committed before failure. The conversation is unusable. |

A result identifies an operation. It does not provide server-side idempotency. The service does not automatically retry input, settings, tool execution, or history mutations. An admission rejection before submission permits an explicit new request. Do not resubmit accepted input or input with an unknown outcome.

## Settings, tools, and history

Settings deltas retain omitted values. Instructions, tools, fillers, voice, interruption policy, and truncation share one SDK update and one acknowledgment. `LLMSetToolsFrame` uses that same path. Empty tools remove all advertised tools. Removing a tool also removes its filler. Explicit fillers must name tools in the resulting tool set.

Fillers use SDK `PrismFiller` values with type `none`, `text`, or `auto`. An automatic filler can include generation instructions. Active replies retain their starting policy and handlers when a later update removes or replaces a tool. Live tools use Pipecat's function machinery. The service records each final result before it requests continuation. If continuation has expired, it uses the SDK's recorded-result reply operation where available. It never runs the tool again for result delivery.

`response_metadata` uses a separate SDK operation and a separate result with operation kind `response_metadata`. Acceptance of a combined settings update does not promise acceptance of its metadata update. `session_info` exposes the accepted settings revision and the actual provider request ID. `on_session_updated` reports accepted session information. `on_rime_event` exposes SDK events, including response and input IDs, speech end reason, metadata, and faults. `LLMFullResponseStartFrame.metadata["rime"]` carries the provider request ID, response ID, input item IDs, settings revision, response metadata, input frame ID, and RTVI client message ID. Fault exceptions retain code, scope, retryability, and correlation. Retryability does not imply that a conversation is usable. A response with an unknown or expired settings revision requires recovery. If a deployment omits revisions, a policy or handler change can make later response ownership ambiguous and require a fresh conversation.

History supports user, assistant, and system text, plus completed function call/result pairs. Messages have a 4,000-character limit. Imported calls require ordered completed results. Unsupported roles, images, invalid encoding, incomplete calls, and invalid edits are rejected before the first history mutation. Text equality does not establish identity. A transform that retains existing message objects can delete records with their server IDs intact. A replacement with new dictionaries, a content change, or a reorder requires a rebuild. Rebuilds require no active replies or pending calls. A rebuild can partially apply and has no atomic rollback.

`await llm.get_history()` reads completed server history. It does not replace local conversation records. Active records can be absent from the server snapshot. Microphone turns reserve their place before their final transcript arrives.

History truncation controls future model context capacity. Startup and runtime settings accept `"auto"`, `"disabled"`, and SDK `RetentionRatio`. The service omits truncation until supplied. Deployment support is required. This setting does not remove local records. Interrupted audio truncation is a separate playback operation.

## Playback and interruption

Each output has a globally unique `context_id`. `AudioOutputStartFrame` registers it and `AudioOutputEndFrame` closes it. `AudioOutputStopFrame` stops interruptible audio for one output. These lifecycle markers survive queue interruption. Audio identities survive output resampling, chunking, filters, and mixing. Speech counts exclude padding and mixer-only background audio. Outputs with disabled audio still receive a terminal report.

`AudioPlaybackReportFrame` travels upstream with one terminal outcome per output and destination. The Rime owner maps it to the original SDK references, including reports that arrive after a service switch. Prism supports one authoritative destination. Concurrent destinations are rejected. Duplicate reports do not generate duplicate receipts. A response's finished receipt waits for all of its output reports, including an empty response.

| Transport path | Timing guarantee |
| --- | --- |
| `LocalAudioTransportParams(audio_out_playback_clock=True)` | Uses PortAudio callback DAC times and the PortAudio stream clock. Counts valid speech samples whose device time has passed. Invalid callback timing or underflow produces unknown timing. This does not establish physical audibility. |
| Other `BaseOutputTransport` paths, including eval and WebRTC | A successful write establishes byte acceptance. The report uses unknown timing, `played_ms=None`, after any accepted audio. It does not claim listener playback. |
| Disabled or fully discarded output | Known zero playback. |
| Test measured-output adapter | Deterministic completion of accepted valid samples, with padding excluded. |

The SDK/server can estimate playback when timing is unknown. The service does not invent a zero interruption position for partly played output. A measured caller interruption reports per-output audio truncation. Application stops cancel generation and report finished playback. Uninterruptible audio remains in the output queue and is included in the terminal report.

`SpeechHint` applies an immediate temporary multiplier at the output transport. Defaults are gain `0.25` and expiry `0.6` seconds. Repeated hints extend expiry. Confirmed speech, shutdown, errors, and expiry remove only this multiplier. Application volume and gain remain independent. Samples already submitted to a device or remote buffer can retain their prior gain.

## Limits and recovery

Input queues, requests, transcripts, tool calls, response references, and unresolved output state have explicit limits. Overflow marks the conversation unusable. Responses are retained within the SDK's 128-response window. Unresolved receipts are never silently evicted. End and cancel use one bounded cleanup deadline. A pause in microphone capture does not close the conversation.

For a settings-only handoff, send instructions and tools in one settings delta. This retains the same conversation. For restart or failover, create a fresh service with a copy of accepted settings and supported completed context. Pending calls cannot resume in a fresh conversation. History import does not replay user input or execute imported tools. Unknown outcomes require an explicit application recovery decision. Old playback reports remain with their original service owner.

Prism does not support disabling server turn detection, manual audio commit, video, image history, forced or disabled tool selection, per-reply tool overrides, or intermediate tool results. Use a separate TTS service for arbitrary application-directed speech.

## Example and validation

Run `examples/realtime/realtime-rime.py` with the development runner:

```sh
uv run --extra rime-realtime --extra runner examples/realtime/realtime-rime.py --transport eval --port 7860
uv run pipecat eval run evals/release/scenarios/scripted/rime_prism.yaml --bot-url ws://localhost:7860/ws -v
```

The example has invoice lookup and a billing handoff. The handoff updates instructions and tools together. The eval scenario covers typed multi-turn context, tool execution, and an observable settings change. Voice and interruption also require audio-mode evals and a separate listener or recording when audible output must be established.
