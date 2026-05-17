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

There is **no** `@pipecat-ai/moq-transport`. So MOQ bots today need a hand-
rolled client (which is what `moq_prebuilt/client/` is). Publishing a
proper transport plugin would:

1. Replace `moq_prebuilt/client/` with the prebuilt smallwebrtc UI — same
   conversation panel, metrics, device pickers, transcript overlay, etc.
2. Let downstream Pipecat apps drop MOQ in as easily as WebRTC.
3. Centralize the MOQ wire-protocol work in one package instead of every
   demo reinventing it.

## What we already have

About 80% of the wire-protocol work is done in
`moq_prebuilt/client/app.js` and `moq_prebuilt/client/moq-client.js`:

- moq-lite-02 codec (CLIENT_SETUP, SERVER_SETUP, ANNOUNCE_PLEASE /
  ANNOUNCE_INIT, SUBSCRIBE / SUBSCRIBE_OK, GROUP / FRAME).
- WebTransport bring-up with certificate pinning.
- Per-participant broadcast paths (`<namespace>/<participant_id>`) and
  multi-track routing (`bot-audio`, `user-audio`, `transcript`).
- Mic capture via AudioWorklet → 16 kHz PCM publish.
- Bot-audio 24 kHz PCM playback via Web Audio.
- RTVI message parsing on the transcript track (`bot-llm-*`,
  `user-transcription`, etc.).

The package's job is to **wrap that in the `Transport` interface** and
emit the events `PipecatClient` expects.

## Effort estimate

### Usable v1 — 3–5 days

For a single bot / single client dev setup that runs against the
playground UI.

| Day | Work |
| --- | ---- |
| 1 | TypeScript package skeleton (build, tsconfig, ESM/CJS exports, npm scripts). Port existing protocol logic into a class. |
| 2 | Implement `Transport` interface: `connect()`, `disconnect()`, `sendMessage()`, `getDevices()`. Wire the RTVI message parsing on the transcript track into the event names `PipecatClient` expects (`connected`, `disconnected`, `trackStarted`, `userTranscript`, `botStartedSpeaking`, …). |
| 3 | Device handling — mic picker, mute, sample-rate negotiation, AudioWorklet inlined as a string blob. |
| 4 | Run against `ConsoleTemplate` from `@pipecat-ai/voice-ui-kit`. Fix lifecycle bugs (mid-call mute, reconnect, etc.). |
| 5 | Docs, example app, README. |

### Production-grade — ~2 weeks on top

- Multi-participant discovery via outbound `ANNOUNCE_PLEASE` + reacting to
  `ANNOUNCE_UPDATE` so multiple bots / multiple clients can share a namespace.
- CA-signed cert path (not just self-signed `serverCertificateHashes`
  pinning).
- Reconnection / network-blip recovery.
- Test suite (unit + at least one e2e against a real relay).
- Cross-browser sanity (Chrome reference, Safari WebTransport just
  shipped, Firefox still behind a flag).
- npm publish workflow / CI.

## Known unknowns (could blow the estimate)

- **PipecatClient `Transport` surface area** — haven't done a full read of
  `@pipecat-ai/client-js`. If the interface expects something MOQ can't
  cleanly model (peer-connection introspection, ICE candidates, anything
  WebRTC-specific that leaks into the abstraction), we need
  workarounds/polyfills.
- **Audio interop with the voice-ui-kit's audio visualizers** —
  `<VoicePresence>`, level meters, etc. read from a `MediaStreamTrack`. MOQ
  delivers raw PCM uni-streams, not a media-stream-track. We'd need to
  synthesize a `MediaStreamTrack` from the decoded PCM so the rest of the
  kit just works. Could be a half-day rabbit hole or could be quick — there
  are off-the-shelf libraries (`AudioWorklet` + `MediaStreamDestination`) but
  the integration details aren't trivial.

## Suggested next step (cheap reconnaissance)

Spend ~2 hours reading:
- `@pipecat-ai/client-js/src/transport.ts` — the `Transport` abstract
  interface.
- `@pipecat-ai/small-webrtc-transport` — closest analogue, since both
  open a session and stream media bidirectionally.
- `@pipecat-ai/websocket-transport` — simpler reference; might be a closer
  fit for MOQ since neither is WebRTC.

Output: a concrete TypeScript API surface for `MoqTransport` (constructor
options, method signatures, events emitted) + a sharper estimate. Two
hours of investment to either commit to the project or pass on it
informedly.

## Open questions for whoever picks this up

- One participant identity per `Transport` instance, or should the
  transport manage multiple peers internally (matching how WebRTC SFUs do
  it)?
- How should `getDevices()` map to MOQ — pure browser device enumeration
  (since there's no SDK-side device negotiation)?
- Where does the per-call ID come from? Server-allocated via
  `/start`, or client-generated UUID? (Today we hardcode `bot0` /
  `client0`.)
- Cert pinning UX — for prod, can we get away with assuming the relay has
  a real CA cert, or do we need an explicit `serverCertificateHashes`
  config knob?
