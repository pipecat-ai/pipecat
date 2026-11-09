# Rime Prism validation

The protocol tests use the published `rimelabs-sdk` with a controlled WebSocket peer and a real `PipelineWorker`. They exercise the Rime context aggregator pair, frame ordering, standard tool execution, and output transport.

```sh
uv run --extra rime-realtime pytest -n 0 tests/test_rime_realtime*.py
```

Coverage includes ordered native text and raw text, input identities, history edits and partial recovery, retained text, concurrent turns, tool progress and cancellation, tool suppression, discard receipts, microphone clearing, usage metrics, startup retries, and shutdown. A pending tool is also exercised across more than 128 completed responses. The peer supplies acknowledgments, readiness, refusals, and timeouts. These checks establish frame and protocol behavior, not deployment behavior or model quality.

The saved behavioral scenarios cover typed tools and handoff, voice memory, interruption followed by another turn, and voice tools:

```sh
uv run --extra rime-realtime --extra runner \
  examples/realtime/realtime-rime.py --port 7860

uv run pipecat eval run evals/release/scenarios/scripted/rime_prism.yaml \
  --bot-url ws://localhost:7860/ws
```

Audio scenarios are in `evals/release/scenarios/scripted/rime_prism_audio.yaml`. See the [release eval guide](../../evals/release/README.md) for caller speech, transcription, and judge setup.

Live validation must check tool invocation counts and arguments, received audio, and cleanup. Provider text alone does not prove what reached the caller. Use an independent transcriber on recorded output after closing the call, so local inference does not delay realtime processing. For room tests, use the standard Pipecat LiveKit transport and a separate RTC caller. Encoded silence can decode to small nonzero samples; check that discard mode emits no TTS audio frames and inspect the received signal level.

The public deployment can accept a runtime tool update but answer without invoking the new tool. A direct SDK control can distinguish that model behavior from a Pipecat integration failure. An accepted settings update alone does not establish a successful business handoff.

Playback positions are estimates. Device timing and physical audibility are not measured by this implementation. Live tests must retain failed attempts and report deployment or transcription failures separately from local protocol results.
