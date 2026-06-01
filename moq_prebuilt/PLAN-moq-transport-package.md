# Plan: `@pipecat-ai/moq-transport` and follow-up work

This file collects the work that's *not* in the current PR — the things
we deliberately deferred so the first MoQ landing stays reviewable.

## 1. `@pipecat-ai/moq-transport` Client SDK plugin

A future Pipecat Client SDK transport plugin for Media-over-QUIC. Lets
any Pipecat JS UI — including the prebuilt `smallwebrtc` playground —
talk to a MOQ-backed bot with zero UI changes, just by swapping the
transport.

The Pipecat Client SDK (`@pipecat-ai/client-js`) is transport-agnostic:
UIs talk to a `PipecatClient`, which delegates to a pluggable
`Transport`. The existing transports are
`@pipecat-ai/small-webrtc-transport`, `@pipecat-ai/daily-transport`,
`@pipecat-ai/websocket-transport`, and
`@pipecat-ai/gemini-live-websocket-transport`. There is no
`@pipecat-ai/moq-transport`, so MOQ bots today need the hand-rolled
client in this folder.

Publishing a proper transport plugin would:

1. Replace `moq_prebuilt/client/` with the prebuilt smallwebrtc UI —
   same conversation panel, metrics, device pickers, transcript
   overlay, etc.
2. Let downstream Pipecat apps drop MoQ in as easily as WebRTC.
3. Centralize the MoQ wiring in one package instead of every demo
   re-doing it.

### What we already have

The wire-protocol work is **done upstream** in
[`@moq/net`](https://www.npmjs.com/package/@moq/net) and
[`@moq/publish`](https://www.npmjs.com/package/@moq/publish). On the
Python side, [`moq-rs`](https://pypi.org/project/moq-rs/) does the
same. The current `moq_prebuilt/client/app.js` is ~450 lines of glue
that:

- Opens a `Moq.Connection.Reload` (with WebTransport cert pinning).
- Publishes a `Publish.Broadcast` containing the mic via
  `Publish.Source.Microphone` + `Audio.Encoder` (Opus encoded).
- Consumes the bot's broadcast and decodes Opus via
  `@moq/hang.Container.Consumer` + WebCodecs `AudioDecoder`.
- Parses RTVI on the bot's transcript track.

The package's job is to **wrap that in the `Transport` interface** and
emit the events `PipecatClient` expects.

### Effort estimate

#### Usable v1 — 2-3 days

For a single bot / single client dev setup that runs against the
playground UI.

| Day | Work |
| --- | ---- |
| 1 | TypeScript package skeleton (build, tsconfig, ESM/CJS exports, npm scripts). Lift `moq_prebuilt/client/app.js` into a class with a `Transport`-shaped surface. |
| 2 | Implement `Transport` interface: `connect()`, `disconnect()`, `sendMessage()`, `getDevices()`. Wire RTVI parsing into the events `PipecatClient` expects (`connected`, `disconnected`, `trackStarted`, `userTranscript`, `botStartedSpeaking`, …). |
| 3 | Run against `ConsoleTemplate` from `@pipecat-ai/voice-ui-kit`. Polish lifecycle (mid-call mute, reconnect already free via Reload, device picker). Docs + example. |

#### Production-grade — ~2 weeks on top

- CA-signed cert path documented as the prod TLS knob (`--moq-cert` /
  `--moq-key` already work; just need the docs).
- Test suite (unit + at least one e2e against a real relay).
- Cross-browser sanity (Chrome reference, Safari WebTransport just
  shipped, Firefox already falls back to WebSocket via `@moq/qmux`).
- npm publish workflow / CI.

### Known unknowns

- **PipecatClient `Transport` surface area.** Haven't done a full
  read of `@pipecat-ai/client-js`. If the interface expects something
  MoQ can't model (peer-connection introspection, ICE candidates,
  anything WebRTC-specific that leaks into the abstraction), we need
  workarounds.
- **Audio interop with voice-ui-kit visualizers.** `<VoicePresence>`,
  level meters, etc. read from a `MediaStreamTrack`. MoQ delivers raw
  PCM, not a media-stream-track. We'd need to synthesize one — could
  be a half-day rabbit hole.

## 2. Production browser bundle (Vite build)

The current demo uses `esm.sh` to fetch `@moq/net`, `@moq/publish`,
`@moq/hang`, `@moq/signals` at runtime. Fine for a dev demo; not
acceptable for prod because:

- Runtime fetch from a third-party CDN is a hard external dependency
  and a privacy / availability risk.
- Per-package versions can drift (we hit this once already: the four
  packages have inter-dependent semver ranges).
- `@moq/publish`'s capture AudioWorklet is published as a dynamic
  ESM chunk. It works through esm.sh today, but isn't a contract.

The path is a Vite build of `moq_prebuilt/client/`:

- `package.json` with the four `@moq/*` packages as direct deps.
- `vite.config.js` configured for ESM output and AudioWorklet
  inlining.
- Build step produces `dist/`; runner serves `dist/` instead of
  `client/`.

About a half-day's work, plus committing to JS toolchain in the dev
loop.

## 3. Video support

Mechanical:

- Add a `publish_video` mirror to `publish_audio` in `MOQTransport`
  (the Python `moq` library exposes
  `BroadcastProducer.publish_video` once it lands; today that crate is
  still empty).
- On the browser, `@moq/publish.Video.Root` already exists — just
  point it at a camera source on `Publish.Broadcast`.
- Pipecat already pushes `OutputImageRawFrame`s; need to route them
  to a `MoqVideoProducer` similar to how `AudioProducer` works.

Wait for `rs/moq-video` to fill in upstream before doing this — until
then there's no Python API to call.

## 4. Bot-side reconnect

`MOQTransport._run()` runs once and exits when the session closes. No
retry. If the relay restarts (client mode) the bot is dead. The
browser already auto-reconnects via `Moq.Connection.Reload`.

Sketch:

```python
async def _run(self):
    backoff = 1.0
    while not self._cancelled:
        try:
            async with self._make_transport(origin) as transport:
                ...
            backoff = 1.0  # reset on clean disconnect
        except Exception as e:
            logger.warning(f"session error: {e}; reconnecting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
```

Complications:

- In serve mode, the cert fingerprints change if `tls_generate` runs
  again. The browser would need new pinning. Probably skip reconnect
  in serve mode.
- The publish broadcast is owned by the transport (created in
  `__init__`). Re-attaching it to a fresh origin per loop iteration
  should work, but needs testing.

## 5. Multi-participant

`MOQParams.participant_id` / `peer_id` are scalar — one bot, one
client. The catalog/announce model in MoQ supports n×m (one broadcast
per participant, discover by announcement prefix). Real prod usage
(group calls, multi-bot orchestration) needs:

- A discovery loop using `consumer.announced(prefix)` instead of a
  pinned `announced_broadcast(path)`.
- Per-peer state (one audio decoder per remote, mix to local output).
- Per-peer event handlers (`on_client_connected(client_id)` etc.).

Not in scope for the first landing; defer.

## 6. Metrics

Pipecat has a metrics framework (`enable_metrics=True` in
`PipelineParams`). The MoQ transport doesn't plug into it. Things
worth exposing:

- Bytes sent / received per track.
- Frames delivered / dropped (the MoQ subscriber's "skipped late
  group" counter — that's the proof we're better than WebRTC under
  congestion).
- RTT (`moq-lite-04` exposes it on the `Session`).
- Opus encode latency.

Without numbers, "MoQ is better than WebRTC under congestion" is a
claim, not a demonstrated property.

## 7. Production TLS

In server mode, `--moq-serve` always self-signs via `tls_generate`.
For prod deploys (CDN edges, real hostnames) we want LetsEncrypt or
pre-issued certs.

The mechanism already works — `--moq-cert ${PEM_CHAIN}
--moq-key ${PRIVATE_KEY} --moq-serve --moq-host my.example.com` does
the right thing — but it isn't called out as "the prod path." Worth
documenting in the runner help text and an `examples/transports/`
deployment recipe.

## 8. Open questions to resolve as we go

- One participant identity per `Transport` instance, or should the
  transport manage multiple peers internally (matching how WebRTC
  SFUs do it)?
- How should `getDevices()` map to MoQ — pure browser device
  enumeration (no SDK-side device negotiation)?
- Where does the per-call ID come from? Server-allocated via
  `/start`, or client-generated UUID? (Today we hardcode `bot0` /
  `client0`.)
