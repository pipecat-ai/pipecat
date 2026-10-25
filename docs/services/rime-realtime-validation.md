# Rime Prism live validation

Validation date: October 24, 2026, America/Los_Angeles. Branch: `feat/rime-prism-realtime`.
The environment used Python 3.13.14, Pipecat 1.14.1.dev2, and `rimelabs-sdk` 0.1.0a22.
The checks used the SDK's public Prism endpoint and the credentials supplied for this validation.
Credentials remained outside the repository.

The integration uses `Rime.realtime.connect`, session events, and direct session operations.
Pipecat runs the pipeline, context aggregators, and application tools. No SDK conversation runner
was used. `RIME_API_KEY` authenticates the service. `TYPESAFE_API_KEY` authenticates the audio
eval judge. The typed scenario does not need the Typesafe key.

## Live checks

The final run of the saved scenarios passed all four checks in 84 seconds. The repository
example's final server log had no service errors or shutdown warnings.

| Check | Evidence |
| --- | --- |
| Typed conversation | The bot retained invoice number 123, called `lookup_invoice`, returned paid status, called `transfer_to_billing`, and then answered with the billing prefix. |
| Voice conversation | Kokoro supplied caller audio. The bot remembered Alex and answered Berlin. Moonshine transcribed returned audio independently of Rime's text events. The Typesafe judge accepted the answers. |
| Interruption | Caller audio interrupted a long story. The client received `bot_interrupted`, the bot answered Tokyo, and the next turn answered Paris. |
| Voice tools and handoff | Audio input triggered invoice lookup and the billing transfer. The spoken replies included paid status and the changed billing prefix. |
| Initial history | The server accepted two history messages without generating a reply. A typed query recovered KILO. |
| Append without inference | A system-history append generated no reply. The next typed query recovered NOVA. |
| Stable history identity | A transform deleted the first of two identical messages. The remaining message retained its server ID. |
| History replacement | A full replacement contained two messages. The next typed query recovered ZULU. |
| Runtime truncation | The server accepted `disabled`, `RetentionRatio(0.7)`, and `auto`, each through its correlated operation result. |
| Settings and metadata | A combined update accepted instructions, tools, automatic filler, the current voice, interruption disabled, and truncation. A separate metadata update accepted `{"validation": "live"}`. The published session revision was 6. |
| Tool after settings update | After invoice confirmation, `lookup` executed once for ABC123. The continuation returned paid status. Six replies received six terminal output reports, with no service errors. |

The control check used `RimeRealtimeLLMService` in a Pipecat `PipelineWorker` with normal context
aggregators. Its output adapter accepted bytes immediately and reported unknown playback timing.
It did not measure device playback. The model asked for invoice confirmation before calling the
tool. The check answered that question before it expected paid status.

## Changes required by validation

Four regression cases exposed faults and passed after the fixes:

- Terminal playback reports can pass through processors during cancellation.
- Cancellation keeps the output transport open until the service collects terminal reports,
  within the configured shutdown deadline. This includes a pending output with no audio.
- Tool continuations retain the initiating typed turn's audio setting and request identifiers.
- A metadata acknowledgment records its published settings revision before later replies use it.

A separate direct SDK probe tested an opening reply with no history. The deployment returned
`turn_failed` with empty text and no audio. An initial greeting history message or an explicit reply
instruction succeeded. The example uses an initial system-history message to request its greeting.
The service does not retry a failed opening reply automatically.

## Playback evidence and limits

Audio evals recorded returned speech and used Moonshine transcription for response assertions.
The final audio run produced nonzero 16 kHz mono recordings of 16.855 seconds for memory,
16.978 seconds for interruption, and 37.000 seconds for tools and handoff. These recordings contain
the conversation audio. They are evidence of captured speech, not physical speaker playback.

The eval transport reports successful byte acceptance with unknown playback timing after audio
has been written. The service leaves `played_ms=None` and lets the SDK/server estimate playback.
It reports known zero for suppressed or discarded audio. The final typed run produced no bot
speaking events. Tool continuations also retained text-only output in the regression test.

The local PortAudio path has tests for callback DAC times, valid speech sample counts, padding,
underflow, and invalid timing. Device enumeration found zero audio devices and zero output devices
in this environment. A live PortAudio playback-clock check and physical audibility remain
unverified. No live WebRTC listener playback measurement was performed.

Opening and runtime truncation have local protocol tests. Live runtime acceptance is recorded
above. Voice changes to a different voice, timeout recovery against a failing public endpoint,
and long-running retention-limit behavior were not tested live. Local protocol tests cover their
accepted-operation and failure contracts where supported. These results do not establish support
for unsupported provider operations.

## Repeatable checks and local evidence

Start the repository example with `RIME_API_KEY` in its environment:

```sh
uv run --extra rime-realtime --extra runner examples/realtime/realtime-rime.py --port 7860
```

Run the saved scenarios with the audio judge dependencies and key available:

```sh
PYTHONPATH=. uv run pipecat eval run \
  evals/release/scenarios/scripted/rime_prism.yaml \
  evals/release/scenarios/scripted/rime_prism_audio.yaml \
  --bot-url ws://localhost:7860/ws --audio --record-dir /tmp/rime-recordings \
  --timeout 35 --debug -v
```

The regression selection covers Rime realtime and TTS, base output, frame processing and queues,
context aggregation, settings, service switching, tools, RTVI, audio volume, user turns,
OpenAI realtime audio, and deprecation markers. It passed 628 tests and 24 subtests. Ruff lint,
Ruff format, Pyright, lock consistency, the CLI service registry check, and the changelog draft
also passed.

Local artifacts from this validation remain outside the repository:

- `/private/tmp/pipecat-rime-verified-eval.log`: final scenario results.
- `/private/tmp/pipecat-rime-verified-bot.log`: final example server log.
- `/private/tmp/pipecat-rime-live/verified/`: per-scenario eval and debug logs.
- `/private/tmp/pipecat-rime-live/recordings-verified/`: final conversation recordings.
- `/private/tmp/pipecat-rime-live/controls/results.json`: accepted history/settings operations
  and the tool result from the control check.
- `/private/tmp/pipecat-rime-verified-tests.log`: regression results.

These temporary artifacts can be removed by the operating system. The saved scenarios and this
report remain in the repository.
