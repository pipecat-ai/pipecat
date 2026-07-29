# Transport latency benchmark

Measures client → echo-bot → client audio round-trip latency across
transports and topologies: SmallWebRTC vs MoQ vs Daily, on loopback, through
local docker relays, through deployed relays on the internet, and against
Pipecat Cloud bots.

## How it measures

A headless Python client sends a probe (silence + a 20 ms chirp every
`--period-ms`, default 500 ms — RTTs approaching the period alias in onset
pairing, so it sits well above the worst RTT under test; 48 kHz mono s16) in
20 ms chunks paced on an absolute `time.monotonic()` schedule, through a minimal echo bot
(`transport.input() -> EchoProcessor -> transport.output()`), and matched-filters
the returned audio for each chirp onset. RTT per marker = arrival stamp of the
chunk containing the onset − handoff stamp of the chunk that carried it. One
process, one clock — no clock sync. Chirps survive lossy Opus; a lost chirp is
a counted drop, not a corrupted number.

Each scenario also has a **browser twin** (`browser-*`): the same bot and
media path, but the measuring client is Chrome running the pipecat client SDK
(`@pipecat-ai/client-js` with `@pipecat-ai/small-webrtc-transport` or
`@pipecat-ai/moq-transport`) — the stack real users ship, including the
browser's Opus and its jitter buffering. Two methods:

- **Method B** (default, the comparison number): the page (`web/`) generates
  the identical probe on its AudioContext clock, publishes it as the SDK's
  mic track, and taps the returned bot track into a recorder worklet stamping
  20 ms chunks on the same clock. Send times are implied by the buffer
  schedule, so one clock covers both directions and the capture goes through
  the same analysis as the headless client.
- **Method A** (`--browser-method a`, the glass-to-glass number): Chrome uses
  two BlackHole virtual devices as mic and speaker; Python plays/records the
  probe through them with `sounddevice` on its monotonic clock — the full
  user-felt path including OS audio I/O. Tagged `browser_method: "a"` and
  never charted against the network-path numbers; A−B on the same scenario
  isolates the fixed browser/OS audio cost.

Beyond steady-state RTT, every trial also records **join** (connect start →
first received audio), **first-echo** (connect start → first echoed chirp),
a **per-marker RTT time series** (`chart-series-*.png` — shows buffer creep
and recovery that medians hide), and — with `--gap-every N` — **talk-spurt
onset RTT** (first chirp after a 2 s silence gap; the turn-start latency a
voice agent user actually feels).

## Scenarios

| slug | bot | media path |
|---|---|---|
| `webrtc-local` | local | SmallWebRTC, host candidates on loopback |
| `moq-serve` | local | MoQ server mode, direct QUIC on loopback |
| `webrtc-turn-local` | local | TURN-forced via dockerized coturn (:3478) |
| `moq-relay-local` | local | via dockerized moq-relay (:4443) |
| `webrtc-turn-deployed` | local | TURN-forced via a deployed TURN server |
| `moq-relay-deployed` | local | via a deployed standalone relay |
| `daily-pcc` | Pipecat Cloud | Daily infrastructure *(phase B`)* |
| `moq-pcc` | Pipecat Cloud | via a deployed standalone relay *(phase B)* |

The six non-PCC scenarios each have a `browser-<slug>` twin: same bot, same
media path, Chrome + the pipecat client SDK as the measuring client.

## Run

All run procedure — commands, gates, local relay setup, agent notes — lives
in [RUNBOOK.md](RUNBOOK.md). The deployed tier's relay box is operated
separately; the runbook only needs its connection values, provided out of
band. This README carries no commands; it explains what the benchmark
measures and how to read the output.

Results land in `results/` (gitignored — never checked in): per-trial JSON, per-trial bot logs
(`bot-<scenario>-<trial>.log`), `summary.md`, and charts —
`chart-all.png` plus one sub-chart per tier pairing the WebRTC and MoQ
scenario (`chart-local-direct.png`, `chart-local-relay.png`,
`chart-deployed-relay.png`, `chart-cloud.png`). Regenerate charts alone with
`uv run python benchmarks/transport_latency/charts.py`.

## Fairness checklist

1. One shared echo bot file (`echo_bot.py`); no VAD, no AI services,
   `audio_in_passthrough=True`.
2. 48 kHz mono both directions on both transports — no resamplers in either
   path.
3. Identical base-transport 10 ms chunking (defaults, both).
4. Opus 20 ms frames both sides (MoQ `audio_out_frame_ms=20`; aiortc default).
5. Opus bitrate/FEC: library defaults both sides; encoders differ (moq-ffi
   Rust vs aiortc/libopus) — part of the stack under test, versions captured.
6. Jitter buffers: one `--jitter-ms` (default 60) pins MoQ's bot
   (`audio_in_max_latency_ms`) and client (`latency_max_ms`) buffers;
   aiortc's implicit prefetch **cannot** be pinned equivalently — quantified
   by the floor trial instead.
7. Same machine and same probe, warmup (first 5 s discarded), duration, and
   cadence for every trial. Deployed/cloud tiers add real network — compare
   them against each other, not against loopback.
8. One process, one monotonic clock for both send and receive stamps.
9. Bot-internal time measured by the same `LatencyObserver` in every run
   (also a canary: asymmetry there would mean the pipelines differ).
10. Topology asserted at runtime, not assumed: the WebRTC client asserts its
    selected ICE pair (host/host direct, local relay when TURN-forced) — in
    the browser tier via `getStats()` with Chrome pinned to
    `iceTransportPolicy: relay`; MoQ serve/relay mode is explicit in the bot
    command and echoed in `results/*.json`.
11. Browser tier: the page's probe generator is asserted marker-identical to
    `probe.py` on every trial, and both browser transports run through the
    same page, worklet, and analysis. Chrome's WebRTC jitter buffer (NetEq)
    is adaptive and cannot be pinned — that adaptivity is part of the stack
    under test; the MoQ web client's buffer floor is pinned to `--jitter-ms`
    (`audioLatencyMs`) like the headless client. `getStats()`
    `jitterBufferDelay` is recorded for WebRTC runs as diagnostic context,
    never charted against MoQ. Browser bars are hatched in the charts:
    comparable with each other, not with the python-client bars.

## Impairment

`impair/impair.sh` (manual sudo, never run by the harness) applies macOS
`dnctl` pipes + `pf` dummynet rules to the local relay ports: `clean`,
`rtt50`, `loss1`, `loss5`, and `burst <profile> <seconds>` for recovery
runs. Trials record the active profile via `--impairment <tag>`. This is
where QUIC-vs-RTP loss behavior actually diverges; clean-localhost numbers
mostly measure stack overhead.
