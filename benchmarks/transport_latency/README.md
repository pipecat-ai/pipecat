# Transport latency benchmark

Measures client → echo-bot → client audio round-trip latency across
transports and topologies: SmallWebRTC vs MoQ vs Daily, on loopback, through
local docker relays, through deployed relays on the internet, and against
Pipecat Cloud bots.

## How it measures

A headless Python client sends a probe (silence + a 20 ms chirp every 250 ms,
48 kHz mono s16) in 20 ms chunks paced on an absolute `time.monotonic()`
schedule, through a minimal echo bot
(`transport.input() -> EchoProcessor -> transport.output()`), and matched-filters
the returned audio for each chirp onset. RTT per marker = arrival stamp of the
chunk containing the onset − handoff stamp of the chunk that carried it. One
process, one clock — no clock sync. Chirps survive lossy Opus; a lost chirp is
a counted drop, not a corrupted number.

## Scenarios

| slug | bot | media path |
|---|---|---|
| `webrtc-local` | local | SmallWebRTC, host candidates on loopback |
| `moq-serve` | local | MoQ server mode, direct QUIC on loopback |
| `webrtc-turn-local` | local | TURN-forced via dockerized coturn (:3478) |
| `moq-relay-local` | local | via dockerized moq-relay (:4443) |
| `webrtc-turn-deployed` | local | TURN-forced via Cloudflare's hosted TURN |
| `moq-relay-deployed` | local | via a deployed standalone relay |
| `daily-pcc` | Pipecat Cloud | Daily infrastructure *(phase B, see `ralph/`)* |
| `moq-pcc` | Pipecat Cloud | via a deployed standalone relay *(phase B)* |

## Run

All run procedure — commands, gates, relay setup, agent notes — lives in
[RUNBOOK.md](RUNBOOK.md), with relay-box provisioning in
[deploy-aws.md](deploy-aws.md) / [deploy-oci.md](deploy-oci.md). This README
carries no commands; it explains what the benchmark measures and how to read
the output.

Results land in `results/` (gitignored — never checked in): per-trial JSON, per-trial bot logs
(`bot-<scenario>-<trial>.log`), `summary.md`, and charts —
`chart-all.png` plus one sub-chart per tier pairing the WebRTC and MoQ
scenario (`chart-local-direct.png`, `chart-local-relay.png`,
`chart-deployed-relay.png`, `chart-cloud.png`). Regenerate charts alone with
`uv run python benchmarks/transport_latency/charts.py`.

## Reading the numbers

Absolute RTTs include each **client stack's own floor** — and the stacks are
wildly different: aiortc's receive path imposes a fixed jitter-buffer prefetch
(~86 ms per leg, ~172 ms for the echo loop measured in-process), while the moq
stack's in-process floor is ~1 ms. A browser client (adaptive NetEq) would sit
elsewhere. That's why `summary.md` reports **excess over the same stack's
floor** as the comparable bot-path number, and the charts draw the floor as a
muted segment under the solid excess. Daily has no in-process floor (media
always traverses Daily infrastructure), so its bars are a single solid
segment.

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
    selected ICE pair (host/host direct, local relay when TURN-forced); MoQ
    serve/relay mode is explicit in the bot command and echoed in
    `results/*.json`.

## Impairment (not yet built)

Design: macOS `dnctl` pipes + `pf` dummynet rules on the relay ports
(`clean`, `rtt50`, `loss1`, `loss5`), applied manually via a sudo helper —
never by the harness. This is where QUIC-vs-RTP loss behavior will actually
diverge; clean-localhost numbers mostly measure stack overhead.
