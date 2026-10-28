# Rime STT validation

The unit tests cover settings snapshots, audio formats, transcript ordering, buffering limits, muting, interruption, errors, client ownership, and shutdown. The protocol tests run the published SDK against a loopback gRPC peer through a `PipelineWorker`. Only the connection factory is redirected; protobuf serialization, SDK audio conversion, transcript validation, status mapping, and request cancellation run normally.

```sh
uv run --extra rime-stt pytest -n 0 \
  tests/test_rime_stt.py tests/test_rime_stt_stream.py tests/test_rime_stt_protocol.py
```

The peer controls replies and completion with events and queues. It covers both transcript contracts, unary WAV requests, overlapping streaming requests, malformed completion, gRPC failures, and cancellation while an injected client remains usable. These tests require no provider credentials. Their loopback channel is plaintext; they do not validate TLS, the public Rime endpoint, or recognition accuracy.

## Audio evals

The [audio scenarios](../../evals/release/scenarios/scripted/rime_stt_audio.yaml) exercise conversation memory, interruption during bot speech, and a subsequent spoken turn. They send Kokoro-generated caller audio through the Rime STT voice agent and transcribe its output with Moonshine. Assertions check the spoken reply for expected words; no LLM judge credential is required for these scenarios.

Use the development environment described in the [usage guide](rime-stt.md#run-the-examples). Set `RIME_API_KEY` and `OPENAI_API_KEY`, then start the example:

```sh
uv run --no-sync python examples/voice/voice-rime-stt.py --port 7860
```

In a second terminal:

```sh
uv run --no-sync pipecat eval run evals/release/scenarios/scripted/rime_stt_audio.yaml \
  --bot-url ws://localhost:7860/ws -v
```

Kokoro and Moonshine may download model assets on first use. The [release eval guide](../../evals/release/README.md) describes the audio dependencies and recording options. The release manifest also includes these scenarios for `voice-rime-stt.py`; its existing capital-question scenario uses the shared release judge configuration.

These synthetic audio checks exercise the STT → LLM → TTS pipeline. They do not establish accuracy on accents, noise, real microphones, or a broad language corpus. Protocol tests verify segmentation and transcript ordering independently of recognition accuracy.
