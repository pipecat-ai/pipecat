# Rime Prism validation

The local protocol tests use the published `rimelabs-sdk` with a WebSocket peer and a `PipelineWorker`. They exercise the standard context aggregators and output transport. Shared runtime modules match this branch's `main`. On October 26, 2026, all 38 Rime protocol tests and 443 related regression tests passed. Ruff, Pyright, lock consistency, registry checks, and the changelog draft also passed.

```sh
uv run --extra rime-realtime pytest -n 0 tests/test_rime_realtime_service.py
```

The tests cover typed history input, speech transcripts, successive turns, tool continuation, settings and filler updates, history edits, interruption, audio conversion, and cleanup. The peer controls provider acknowledgments, rejection, and readiness events. These tests establish frame and protocol behavior, not public endpoint behavior or physical playback.

The saved scenarios cover typed tools and handoff, voice memory, interruption followed by another turn, and voice tools and handoff:

```sh
uv run --extra rime-realtime --extra runner \
  examples/realtime/realtime-rime.py --port 7860

uv run pipecat eval run evals/release/scenarios/scripted/rime_prism.yaml \
  --bot-url ws://localhost:7860/ws
```

Audio scenarios are in `evals/release/scenarios/scripted/rime_prism_audio.yaml`. See the [release eval guide](../../evals/release/README.md) for caller speech, transcription, and judge setup.

Public endpoint checks recorded on October 24, 2026 used the earlier implementation with shared operation and playback contracts. Those results do not validate the current implementation. Public endpoint checks must be repeated with `RIME_API_KEY` available. Audio judgment also needs the configured speech, transcription, and judge dependencies.

Playback positions are estimates. Device timing, physical audibility, and remote listener timing are not measured by this implementation.
