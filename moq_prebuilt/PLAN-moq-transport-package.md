# Plan: `@pipecat-ai/moq-transport`

A future Pipecat Client SDK transport plugin for Media-over-QUIC. Lets any
Pipecat JS UI — including the smallwebrtc playground — talk to a
MOQ-backed bot with zero UI changes, just by swapping the transport.

## Why this exists

The Pipecat Client SDK (`@pipecat-ai/client-js`) is transport-agnostic. UIs
talk to a `PipecatClient`, which delegates to a pluggable `Transport`. The
existing transports are:

- `@pipecat-ai/small-webrtc-transport` (WebRTC)
- `@pipecat-ai/daily-transport` (Daily)
- `@pipecat-ai/websocket-transport` (raw WebSocket / telephony)
- `@pipecat-ai/gemini-live-websocket-transport`

There is **no** `@pipecat-ai/moq-transport`. So MOQ bots today need a
small hand-rolled client (this `moq_prebuilt/client/` demo). Publishing a
proper transport plugin would:

1. Replace `moq_prebuilt/client/` with the prebuilt smallwebrtc UI — same
   conversation panel, metrics, device pickers, transcript overlay, etc.
2. Let downstream Pipecat apps drop MOQ in as easily as WebRTC.
3. Centralize the MOQ wiring in one package instead of every demo
   re-doing it.

## What we already have

The wire-protocol work is **done upstream** in
[`@moq/net`](https://www.npmjs.com/package/@moq/net) (and on the Python
side, [`moq-rs`](https://pypi.org/project/moq-rs/)). The current
`moq_prebuilt/client/app.js` is just ~400 lines of glue:

- `Moq.Connection.connect(url, { webtransport })` for the QUIC bring-up
  and certificate pinning.
- Per-participant broadcast paths (`<namespace>/<participant_id>`),
  driven by `connection.publish(path, broadcast)` and
  `connection.consume(path).subscribe(track, priority)`.
- Mic capture via AudioWorklet → 16 kHz PCM frames into the broadcast
  track the bot subscribes to.
- Bot-audio 24 kHz PCM playback via Web Audio.
- RTVI message parsing on the bot's transcript track.

The package's job is to **wrap that in the `Transport` interface** and
emit the events `PipecatClient` expects.

## Effort estimate

### Usable v1 — 2–3 days

For a single bot / single client dev setup that runs against the
playground UI.

| Day | Work |
| --- | ---- |
| 1 | TypeScript package skeleton (build, tsconfig, ESM/CJS exports, npm scripts). Lift `moq_prebuilt/client/app.js` into a class with a `Transport`-shaped surface. |
| 2 | Implement `Transport` interface: `connect()`, `disconnect()`, `sendMessage()`, `getDevices()`. Wire the RTVI message parsing on the transcript track into the event names `PipecatClient` expects (`connected`, `disconnected`, `trackStarted`, `userTranscript`, `botStartedSpeaking`, …). |
| 3 | Run against `ConsoleTemplate` from `@pipecat-ai/voice-ui-kit`. Polish lifecycle (mid-call mute, reconnect, device picker, AudioWorklet inlined as a string blob). Docs + example. |

### Production-grade — ~2 weeks on top

- Outbound discovery via `connection.announced()` so multiple
  bots/clients can share a namespace.
- CA-signed cert path (not just self-signed `serverCertificateHashes`
  pinning).
- Reconnection / network-blip recovery.
- Test suite (unit + at least one e2e against a real relay).
- Cross-browser sanity (Chrome reference, Safari WebTransport just
  shipped, Firefox still behind a flag — but `@moq/net` already auto-
  falls-back to WebSocket on Firefox).
- npm publish workflow / CI.

## Known unknowns

- **PipecatClient `Transport` surface area** — haven't done a full read
  of `@pipecat-ai/client-js`. If the interface expects something MOQ
  can't cleanly model (peer-connection introspection, ICE candidates,
  anything WebRTC-specific that leaks into the abstraction), we need
  workarounds/polyfills.
- **Audio interop with the voice-ui-kit's audio visualizers** —
  `<VoicePresence>`, level meters, etc. read from a `MediaStreamTrack`.
  MOQ delivers raw PCM frames, not a media-stream-track. We'd need to
  synthesize a `MediaStreamTrack` from the decoded PCM so the rest of
  the kit just works. Could be a half-day rabbit hole or could be quick
  — there are off-the-shelf libraries (`AudioWorklet` +
  `MediaStreamDestination`) but the integration details aren't trivial.

## Suggested next step

Spend ~2 hours reading:

- `@pipecat-ai/client-js/src/transport.ts` — the `Transport` abstract
  interface.
- `@pipecat-ai/small-webrtc-transport` — closest analogue, since both
  open a session and stream media bidirectionally.
- `@pipecat-ai/websocket-transport` — simpler reference; might be a
  closer fit for MOQ since neither is WebRTC.

Output: a concrete TypeScript API surface for `MoqTransport`
(constructor options, method signatures, events emitted) + a sharper
estimate.

## Open questions for whoever picks this up

- One participant identity per `Transport` instance, or should the
  transport manage multiple peers internally (matching how WebRTC SFUs
  do it)?
- How should `getDevices()` map to MOQ — pure browser device enumeration
  (since there's no SDK-side device negotiation)?
- Where does the per-call ID come from? Server-allocated via
  `/start`, or client-generated UUID? (Today we hardcode `bot0` /
  `client0`.)
- Cert pinning UX — for prod, can we get away with assuming the relay
  has a real CA cert, or do we need an explicit
  `serverCertificateHashes` config knob?
- **Use `publish_media` / `subscribe_media` (codec'd audio with built-in
  `max_latency_ms`)?** The current demo publishes raw PCM via the
  generic `publish_track` API to avoid carrying an Opus encoder in the
  browser/bot. Moving to `publish_media` would unlock MoQ's
  OrderedConsumer — flushes ASAP if no gaps, blocks up to N ms for late
  groups — which is the headline congestion-control win over WebRTC.
