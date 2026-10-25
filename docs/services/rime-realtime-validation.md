# Rime Prism live validation

On October 24, 2026, the `feat/rime-prism-realtime` branch was checked against the public Prism endpoint with Python 3.13.14, Pipecat 1.14.1.dev2, and `rimelabs-sdk` 0.1.0a22.

## Saved scenarios

All four saved scenarios passed in 84 seconds. The example server log had no service errors or shutdown warnings.

| Scenario | Observed behavior |
| --- | --- |
| Typed conversation | Retained invoice 123, called `lookup_invoice`, returned paid status, and called `transfer_to_billing`. Later replies used the billing prefix. |
| Voice conversation | Remembered Alex and answered Berlin. Kokoro supplied caller speech; Moonshine independently transcribed replies, which the Typesafe judge accepted. |
| Interruption | Caller speech interrupted a long story. The client received `bot_interrupted`, and subsequent turns answered Tokyo and Paris. |
| Voice tools and handoff | Speech input triggered invoice lookup and billing transfer. Replies included paid status and the changed billing prefix. |

## Additional controls

Control checks used `RimeRealtimeLLMService` in a `PipelineWorker` with normal context aggregators. The output adapter accepted bytes and reported unknown playback timing. It did not measure device playback.

| Control | Observed behavior |
| --- | --- |
| Initial history | Imported two messages without generating a reply; a later typed query recovered KILO. |
| Append without inference | Added system history without a reply; the next query recovered NOVA. |
| Stable history identity | Deleted the first of two identical messages while retaining the second message's server ID. |
| History replacement | Replaced history with two messages; the next query recovered ZULU. |
| Runtime truncation | Accepted `disabled`, `RetentionRatio(0.7)`, and `auto` through correlated operation results. |
| Settings and metadata | Accepted combined instructions, tools, automatic filler, current voice, interruption disabled, and truncation. A separate metadata update accepted `{"validation": "live"}`. The published revision was 6. |
| Tool after update | After invoice confirmation, `lookup` executed once for ABC123 and returned paid status. Six replies received six terminal output reports with no service errors. |

## Opening reply

An opening reply with no history returned `turn_failed`, empty text, and no audio in a direct SDK probe. Initial greeting history or an explicit reply instruction succeeded. The example uses initial system history to request its greeting; failed opening replies are not retried automatically.

## Playback evidence and limits

Recorded conversation audio was non-silent, 16 kHz mono, and independently transcribed by Moonshine. This establishes returned speech, not physical speaker playback.

Eval output reports byte acceptance with `played_ms=None`; the SDK/server estimates playback. Suppressed or fully discarded output reports zero. Text-only output was also checked for tool continuations.

No audio devices were available for a live PortAudio check. Device-clock timing, valid speech counts, padding, underflow, and invalid timing have local tests. Physical audibility and live WebRTC listener timing remain unverified.

Live checks also exclude a change to a different voice, timeout recovery against a failing public endpoint, and long-running retention-limit behavior. Opening and runtime truncation have local protocol tests; runtime acceptance was checked live.

## Reproduce the scenarios

Start the example with `RIME_API_KEY` available:

```sh
uv run --extra rime-realtime --extra runner examples/realtime/realtime-rime.py --port 7860
```

Audio checks require caller speech, transcription, and judge dependencies plus `TYPESAFE_API_KEY`. See the [usage guide](rime-realtime.md#example-and-validation) for setup, then run:

```sh
PYTHONPATH=. uv run pipecat eval run \
  evals/release/scenarios/scripted/rime_prism.yaml \
  evals/release/scenarios/scripted/rime_prism_audio.yaml \
  --bot-url ws://localhost:7860/ws --audio --record-dir /tmp/rime-recordings \
  --timeout 35 --debug -v
```

## Regression checks

The recorded regression run passed 628 tests and 24 subtests across Rime realtime/TTS, shared pipeline and output behavior, tools, RTVI, user turns, and OpenAI realtime audio. Ruff lint and format, Pyright, lock consistency, the CLI service registry check, and the changelog draft also passed.

Raw logs and recordings are not stored in the repository. The saved scenarios provide repeatable checks; the results above record this validation environment.
