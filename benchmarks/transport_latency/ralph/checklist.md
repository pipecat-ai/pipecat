# Phase B checklist — Pipecat Cloud scenarios

Worked through by `ralph.sh`, one task per iteration, strictly in order.
A task is checked off only after its acceptance criteria are verified.
Human-gated steps land in `HUMAN_TODO.md` and pause the loop.

- [ ] **B1: `daily_client.py` — DailyConnector.**
  Create `benchmarks/transport_latency/daily_client.py` with a `DailyConnector(room_url, token)`
  matching the connector shape in `webrtc_client.py` (`start` / `send_chunk` / `recv_chunks` / `stop`).
  Design (verified against daily-python 0.29.x as used in `src/pipecat/transports/daily/transport.py`):
  - Send: `CustomAudioSource(48000, 1, auto_silence=True)` + `CustomAudioTrack(source)`; pass the
    track id at join under `client_settings["inputs"]["microphone"]["settings"]["customTrack"]["id"]`
    with `publishing.microphone = {"isPublishing": True, "sendSettings": {"channelConfig": "mono"}}`.
    `send_chunk` = `source.write_frames(pcm)` fire-and-forget (never await its completion in the
    paced loop — client_core's absolute schedule stays authoritative).
  - Receive: after `update_subscriptions(participant_settings={pid: {"media": {"microphone": "subscribed"}}})`,
    `client.set_audio_renderer(pid, cb, audio_source="microphone", sample_rate=48000, callback_interval_ms=20)`.
    The callback runs on daily's thread: hop each `AudioData.audio_frames` chunk into an asyncio
    queue via `loop.call_soon_threadsafe`; `recv_chunks` yields from that queue. Assert 48 kHz mono.
  - Lifecycle: `Daily.init()` once per process behind a class flag; bridge `join`/`leave` completions
    to futures (see `completion_callback` in the daily transport); attach the renderer from
    `on_participant_joined` **and** by scanning `client.participants()` after join (bot may already
    be in the room); gate `start()` return on first received audio (30 s timeout); `stop()` =
    `leave()` await, then `client.release()` in an executor. Never call-and-wait a `CallClient`
    method from inside a daily callback (deadlock).
  - Include a `_main()` self-test entry point: `--room-url`, `--token`, `--duration` (default 15),
    running `run_trial` and printing n/drops/p50/p95 like `webrtc_client._main`.
  Acceptance: `uv run python -m py_compile benchmarks/transport_latency/daily_client.py` passes and
  `uv run python -c "import sys; sys.path.insert(0,'benchmarks/transport_latency'); import daily_client"` passes.

- [ ] **B2: echo bot Daily support + local smoke test.**
  Add a `"daily"` entry to `transport_params` in `benchmarks/transport_latency/echo_bot.py`:
  `DailyParams(audio_in_enabled=True, audio_out_enabled=True, audio_in_passthrough=True,
  audio_in_sample_rate=48000, audio_out_sample_rate=48000)` — no VAD, no filters (fairness: same
  pipeline as the other transports). Then smoke-test locally: this needs `DAILY_API_KEY`
  (human gate if unset — ask for it in HUMAN_TODO.md). With the key: start
  `uv run python benchmarks/transport_latency/echo_bot.py -t daily`, obtain room URL + token from
  its `POST /start` response, and run
  `uv run python benchmarks/transport_latency/daily_client.py --room-url ... --token ... --duration 15`.
  Acceptance: the self-test reports n ≥ 30 measured markers and a finite p50.

- [ ] **B3: PCC deploy assets.**
  Create `benchmarks/transport_latency/pcc/`: a `Dockerfile` that packages `echo_bot.py` (+ the
  benchmark modules it imports) on the standard Pipecat Cloud base image, a `pcc-deploy.toml`
  (agent name `latency-echo-bot`), and a short `README.md` with the build/push/deploy commands.
  Follow current PCC docs/examples for the base image and entry point (check
  `pipecat-context-hub` MCP or the pipecat-examples repo `deployment/` dirs for the pattern).
  Then HUMAN GATE: append the exact `docker build/push` + `pcc deploy` (or `pipecat cloud deploy`)
  commands to HUMAN_TODO.md for the human to run — do not run them.
  Acceptance: files exist, Dockerfile references the real echo_bot path, README commands are
  complete enough to run verbatim.

- [ ] **B4: `daily-pcc` scenario wiring.**
  In `benchmarks/transport_latency/transport_latency.py`, implement the `bot == "pcc"` path for
  `daily-pcc`: `POST https://api.pipecat.daily.co/v1/public/{agent}/start` with
  `Authorization: Bearer $PIPECAT_CLOUD_API_KEY` and `{"createDailyRoom": true}`; take the Daily
  room URL + token from the response and run `DailyConnector` for each trial (one PCC session per
  trial). Add `--pcc-agent` (default `latency-echo-bot`). No floor exists for daily —
  `Scenario.floor_key` already returns None; skip the floor step.
  Human gate: needs the agent from B3 deployed and `PIPECAT_CLOUD_API_KEY` set.
  Acceptance: `--scenario daily-pcc --trials 1 --duration 30` completes, writes
  `results/daily-pcc-1.json` with n > 0, and `chart-cloud.png` renders.

- [ ] **B5: `moq-pcc` scenario wiring.**
  The PCC bot runs with `--moq-connect <deployed relay>` (client mode). Rendezvous: generate a
  unique namespace per trial in the harness, pass it in the PCC start body, and give
  `MoqConnector` an explicit mode that skips the local `/start` handshake — connect straight to
  `--relay-url` with that namespace and the default `request`/`response` participant ids.
  OPEN ITEM: confirm how the PCC start body reaches the runner's moq args on this branch
  (see `src/pipecat/runner/` and the MoQ-on-PCC integration work) before wiring; if it is not yet
  supported upstream, write what's missing to HUMAN_TODO.md and stop.
  Human gate: deployed relay URL + deployed agent + `PIPECAT_CLOUD_API_KEY`.
  Acceptance: `--scenario moq-pcc --trials 1 --duration 30 --relay-url ...` completes with n > 0.

- [ ] **B6: full cloud runs + docs.**
  Run both cloud scenarios at 3 trials × 60 s, confirm `summary.md` orders them last and
  `chart-all.png` + `chart-cloud.png` include them, and update
  `benchmarks/transport_latency/README.md`: replace the *(phase B)* markers with the real run
  commands (PCC API key, `--pcc-agent`, relay URL).
  Acceptance: charts include both cloud scenarios; README documents the cloud runs.
