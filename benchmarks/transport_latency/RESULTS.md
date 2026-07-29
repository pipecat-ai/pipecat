# Transport latency — campaign results

**Campaign 2026-08-06, partial: local tiers complete (4/6), deployed tier
pending a cloud box** (AWS vs OCI undecided — see [deploy-oci.md](deploy-oci.md)).
Procedure: [RUNBOOK.md](RUNBOOK.md); raw per-trial data in `campaigns/<date>/`
once the campaign completes.

## Headline numbers (3 × 60 s trials, Apple M4 Pro, commit bf5e1909e)

Values are medians across trials; drops are totals across all 660 markers.

| scenario | topology | p50 RTT | excess over own client-stack floor | jitter (RFC3550) | drops |
|---|---|---|---|---|---|
| moq-serve | MoQ direct (QUIC, loopback) | 23.8 ms | **22.4 ms** | 20.9 ms | 0/660 |
| webrtc-local | SmallWebRTC direct (SRTP, loopback) | 210.6 ms | **37.3 ms** | 19.4 ms | 0/660 |
| moq-relay-local | MoQ via dockerized moq-relay | 27.5 ms | **26.1 ms** | 22.1 ms | 17/660 * |
| webrtc-turn-local | SmallWebRTC via dockerized coturn | 222.8 ms | **49.4 ms** | 0.9 ms | 0/660 |
| moq-relay-deployed | MoQ via cloud moq-relay | — | — | — | — |
| webrtc-turn-deployed | SmallWebRTC via cloud coturn | — | — | — | — |

\* Drops occur on the relay→client return leg (the bot echoed complete
streams); reproduced across three full runs (28, 21, 17 of 660 — the last on
a quiet machine, ruling out session churn). Latency is unaffected and the
direct QUIC path drops zero, so docker's UDP port forwarding is the prime
suspect; the deployed relay (no docker port mapping) is the cross-check.
The prior runs' trial JSONs are kept in `results/drops-investigation/`.

Compare **excess over own floor** within a tier, never raw p50 across client
stacks — see "Reading the numbers" in [README.md](README.md). Floors this
campaign: moq 1.32 ms, webrtc 173.34 ms.

## Environment

```
pipecat bf5e1909e · branch vp-claude-moq-client-js-python-0dff25-rebased-bench
moq-rs 0.3.3 · moq-ffi 0.2.30 · aiortc 1.14.0
macOS 15.7 · Apple M4 Pro · docker engine: Docker Desktop 28.4.0
relay image: pipecat-moq-relay (MOQ_TAG moq-relay-v0.13.5) · coturn/coturn:4.7
```

## Findings (local tiers)

- **Excess over floor** is the comparable number, and MoQ carries less of it
  at both local tiers: 22.4 vs 37.3 ms direct, 24.8 vs 49.4 ms relayed.
- **A loopback relay hop is nearly free for MoQ** (+2.4 ms excess) and
  costs WebRTC more (+12.1 ms excess, TURN allocation path).
- **Jitter numbers are not comparable across stacks**: aiortc's jitter
  buffer absorbs variance into its fixed ~173 ms floor (hence ~1 ms measured
  jitter through TURN), while the MoQ client surfaces arrival variance
  (~21 ms) instead of pre-buffering it.
- **moq-relay-local drops** (see * above) are the one regression vs July's
  smoke numbers; July's 82/660 TURN drops did *not* reproduce (0/660 today).
