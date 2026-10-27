# Rime Prism validation

The local protocol tests use the published `rimelabs-sdk` with a WebSocket peer and a `PipelineWorker`. They exercise the Rime context aggregator pair and standard output transport. Shared runtime modules match this branch's `main`. On October 27, 2026, with SDK `0.1.0a24`, all 55 Rime protocol tests passed within a set of 290 protocol and related regression tests. The 36 deprecation checks, Ruff, Pyright, lock consistency, registry checks, and changelog draft also passed.

```sh
uv run --extra rime-realtime pytest -n 0 tests/test_rime_realtime_service.py
```

The tests cover typed history input, speech transcripts, successive turns, tool continuation, settings and filler updates, history edits, interruption, audio conversion, and cleanup. They check that speech aggregation requests no duplicate reply, partial local text does not replace provider history, and deferred application edits work through either aggregator. They also check SDK retained-text events and prompt-exclusion summaries without local context replacement or history deletion. The peer controls provider acknowledgments, rejection, and readiness events. These tests establish frame and protocol behavior, not public endpoint behavior or physical playback.

The saved scenarios cover typed tools and handoff, voice memory, interruption followed by another turn, and voice tools and handoff:

```sh
uv run --extra rime-realtime --extra runner \
  examples/realtime/realtime-rime.py --port 7860

uv run pipecat eval run evals/release/scenarios/scripted/rime_prism.yaml \
  --bot-url ws://localhost:7860/ws
```

Audio scenarios are in `evals/release/scenarios/scripted/rime_prism_audio.yaml`. See the [release eval guide](../../evals/release/README.md) for caller speech, transcription, and judge setup.

Public endpoint checks on October 27, 2026 used SDK `0.1.0a24` and the Rime aggregator pair with direct output and a LiveKit room. Both runs passed spoken turns without extra proactive replies, typed recall, three caller interruptions followed by spoken turns, application interruption, and explicit history replacement. All eight playback-truncation acknowledgments reached `on_rime_event` with `kept_text` available. Populated prompt-exclusion summaries were tested with the local peer. All owned clients closed, and the test room was absent after cleanup.

The caller used synthetic speech. Checks compared provider text with expected content and verified nonzero output audio. They did not independently transcribe the generated audio. These checks do not establish reliable use of tools replaced during a session; that provider behavior remains unresolved.

Playback positions are estimates. Device timing, physical audibility, and remote listener timing are not measured by this implementation.
