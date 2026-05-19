# Plan: `@pipecat-ai/moq-transport` + unified MoQ

A future Pipecat Client SDK transport plugin for Media-over-QUIC that
plugs into the existing `pipecat-prebuilt` React UI, so MoQ becomes just
another transport choice alongside SmallWebRTC, Daily, WebSocket, and
Twilio — no separate frontend, no separate `/start` flow.

> Findings below are from a recon pass against
> `@pipecat-ai/client-js@HEAD`, `@moq/lite`, `@moq/publish`, and
> `@moq/watch` source. License, cert, and ESM unknowns are now resolved.

## Why this exists

The Pipecat Client SDK (`@pipecat-ai/client-js`) is transport-agnostic.
UIs talk to a `PipecatClient`, which delegates to a pluggable `Transport`.
The existing transports are:

- `@pipecat-ai/small-webrtc-transport` (WebRTC)
- `@pipecat-ai/daily-transport` (Daily)
- `@pipecat-ai/websocket-transport` (raw WebSocket / telephony)
- `@pipecat-ai/gemini-live-websocket-transport`

There is **no** `@pipecat-ai/moq-transport`, so MoQ bots today ship with
a hand-rolled HTML/JS client (`moq_prebuilt/`) that reinvents protocol
plumbing, mic capture, transcript rendering. The custom client also
collided with the unified routes (`/`, `/client`, `/start`) until we
split them with `if args.transport == "moq" else ...` in
`_configure_server_app`.

Publishing the transport plugin gets us:

1. **Delete `moq_prebuilt/` entirely.** The `pipecat-prebuilt` React UI
   (`ConsoleTemplate` from `@pipecat-ai/voice-ui-kit`) renders MoQ
   sessions with zero changes — same conversation panel, metrics, device
   pickers, transcript overlay.
2. **Unified server routes.** MoQ becomes a `transport == "moq"` branch
   inside `_setup_unified_start_route` (alongside webrtc/daily/etc.),
   removing the need for the `if args.transport == "moq" else ...`
   block in `_configure_server_app`.
3. **No more bespoke protocol code in the browser.** The
   [`@moq/lite`](https://www.npmjs.com/package/@moq/lite) and
   [`@moq/publish`](https://www.npmjs.com/package/@moq/publish) npm
   packages own the wire format and media pipeline. We become consumers.

## Target architecture

```
pipecat-prebuilt (React)
  └─ PipecatClient (@pipecat-ai/client-js)
      └─ @pipecat-ai/moq-transport (this package)
          ├─ @moq/lite       (https://www.npmjs.com/package/@moq/lite)
          └─ @moq/publish    (https://www.npmjs.com/package/@moq/publish)
```

Server side (`src/pipecat/runner/run.py`):

- `_setup_unified_start_route` gets an `elif transport == "moq":` branch
  that creates `MOQRunnerArguments` with a `ready_event` and returns
  once the bot signals ready. **(Done — see commit history.)**
- `_setup_moq_routes` and the `if args.transport == "moq" else ...`
  switch in `_configure_server_app` go away entirely once the JS
  transport package exists.
- The Python bot transport (`src/pipecat/transports/moq/`) is unchanged
  — it still speaks moq-lite-02 via aioquic and sends raw PCM (see
  "Path A vs Path B" below).

## Recon: `Transport` interface surface

Source: `/Users/vipyned2/Documents/repos/pipecat-ai/pipecat-client-web/client-js/client/transport.ts`.

15+ abstract methods to implement. Grouped:

**Lifecycle**
- `_connect(connectParams)` / `_disconnect()` / `sendReadyMessage()`
- `state` getter/setter (`TransportState` enum)
- `_validateConnectionParams(connectParams)`
- `initialize(options, messageHandler)` / `initDevices()`

**Devices** (audio-only MoQ can stub the cam/screen ones with no-ops)
- `getAllMics()` / `getAllCams()` / `getAllSpeakers()` → `MediaDeviceInfo[]`
- `updateMic(id)` / `updateCam(id)` / `updateSpeaker(id)`
- `selectedMic` / `selectedCam` / `selectedSpeaker` getters
- `enableMic(bool)` / `enableCam(bool)` / `enableScreenShare(bool)`
- `isMicEnabled` / `isCamEnabled` / `isSharingScreen` getters

**Messaging**
- `sendMessage(message: RTVIMessage)` — wraps our transcript track
  publish on the bot's broadcast.

**Critical**: `tracks(): { local: {audio?: MediaStreamTrack, …}, bot?: {audio?: MediaStreamTrack, …} }`. Voice-ui-kit's `<VoicePresence>`,
level meters, mute controls all consume real `MediaStreamTrack`
objects. Not negotiable.

## Recon: `@moq/lite`

- **API**: `connect(url, props?): Promise<Established>` → has
  `Broadcast`/`Track` abstractions with `subscribe(name, priority)`,
  `recvGroup()`, `appendGroup()`, `writeFrame()`, `readFrame()`, etc.
- **WebTransport + WebSocket fallback** built in. WebTransport options
  pass through, so **`serverCertificateHashes` self-signed pinning works**
  ✅. Firefox falls back to WebSocket automatically (Firefox WebTransport
  has issues with server-initiated bidi streams).
- **License**: dual Apache 2.0 / MIT — compatible with BSD-2-Clause
  pipecat ✅.
- **ESM** ✅.

## Recon: `@moq/publish`

- High-level wrapper on `@moq/lite`. Has a `<moq-publish>` custom
  element OR direct `Broadcast` class.
- `Microphone` source exposes `Signal<MediaStreamTrack>` — **we get the
  local mic track for free** ✅ (no synthesis needed for the
  `tracks().local.audio` side).
- Encodes via WebCodecs internally — uses `@moq/hang` catalog/container
  format for the wire bytes.
- License compatible ✅.

## Recon: `@moq/watch`

The companion package for the subscriber side. Plays decoded audio
through an `AudioContext`. **Does not expose a `MediaStreamTrack`
natively** — we'd need to bridge via `MediaStreamAudioDestinationNode`.

## The architectural fork: Path A vs Path B

Our Python bot sends **raw PCM** over moq-lite tracks.
`@moq/publish` / `@moq/watch` use **WebCodecs Opus inside `@moq/hang`
containers**. So:

### Path A — match `@moq/publish`'s wire format (the "official" pipeline)

The Python bot emits `@moq/hang` catalog + WebCodecs-style Opus
container. JS client uses `@moq/publish` for mic / `@moq/watch` for bot
audio playback. Gets us hardware-accelerated codec for free + a clean
upgrade path to video.

**Cost**: Python side needs an Opus encoder bound to aioquic + container
framing. **Week+ of bot-side work** as a separate project.

### Path B — use only `@moq/lite` for the protocol

Keep raw PCM on the wire. Hand-roll PCM publish/subscribe on top of
`@moq/lite`'s `Track.writeFrame` / `readFrame`. Bridge PCM →
`MediaStreamTrack` ourselves via `MediaStreamAudioDestinationNode`
(~½ day Web Audio plumbing). Server stays unchanged.

**Recommendation: Path B for v1.** Ship the JS transport against the
existing Python bot. Migrate to Path A later as a separate effort if
hardware codec efficiency becomes important.

## Effort estimate (Path B)

### v1 — 4–5 days

| Day | Work |
| --- | ---- |
| 1 | TS package skeleton (Vite/tsup, ESM exports, types). Wrap `@moq/lite` `connect()` + `Established` in `_connect`/`_disconnect`/`state`. Set up `Signal` subscriptions for state changes. |
| 2 | Device enumeration via `navigator.mediaDevices.enumerateDevices`. Local audio via `getUserMedia` → AudioWorklet → PCM → `Track.writeFrame` (raw PCM, not WebCodecs). Expose mic `MediaStreamTrack` via `tracks().local.audio`. |
| 3 | Bot audio: subscribe to `pipecat/<bot>/bot-audio` via `@moq/lite`, decode PCM frames, route through `AudioBufferSourceNode` → `MediaStreamAudioDestinationNode` → `tracks().bot.audio`. Plus playback. |
| 4 | Transcript track subscription. Port RTVI message dispatch — emit `PipecatClient` callbacks (`onUserTranscript`, `onBotTtsText`, etc.) instead of DOM updates. Stub unimplemented `Transport` methods (camera, screen share). |
| 5 | Server: drop the if/else in `_configure_server_app`. Delete `_setup_moq_routes` and `moq_prebuilt/` entirely. Add MoQ to `pipecat-prebuilt`'s `TRANSPORT_OPTIONS`. Smoke-test end to end. |

Higher than the earlier 2–4 day estimate. Reasons:
- `Transport` interface is larger than guessed (15+ methods).
- `MediaStreamTrack` synthesis for bot audio is **mandatory**, not
  optional — voice-ui-kit requires it.

### Production-grade — ~1 week on top

- Multi-participant discovery via outbound `ANNOUNCE_PLEASE` + reacting
  to `ANNOUNCE_UPDATE` so multiple bots / multiple clients can share a
  namespace.
- CA-signed cert path (not just self-signed `serverCertificateHashes`
  pinning).
- Reconnection / network-blip recovery.
- Test suite (unit + at least one e2e against a real relay).
- Cross-browser sanity (Chrome reference; Safari WebTransport shipped;
  Firefox falls back to WebSocket automatically via `@moq/lite`).
- npm publish workflow / CI.

## Resolved questions (from recon)

- ✅ **`@moq/lite` self-signed cert support.** Yes — WebTransport
  options pass through, `serverCertificateHashes` works.
- ✅ **License compatibility.** Dual Apache-2.0 / MIT, compatible with
  BSD-2-Clause pipecat.
- ✅ **ESM packaging.** Yes.
- ✅ **Local mic `MediaStreamTrack`.** `@moq/publish`'s `Microphone`
  source exposes one as a `Signal<MediaStreamTrack>`.
- ⚠️ **Bot audio `MediaStreamTrack`.** `@moq/watch` does not expose
  one. We synthesize via `MediaStreamAudioDestinationNode`. Standard
  Web Audio pattern, half a day.
- ⚠️ **Audio container format.** `@moq/publish` uses `@moq/hang`
  WebCodecs containers. Our Python bot sends raw PCM. Picking Path B
  above sidesteps this entirely.

## Open questions still to answer

- **How MoQ config reaches the browser in the unified flow.** Currently
  `/api/config` returns relay host/port/cert-hash/participant id.
  Options: fold into the `/start` response, keep a `/api/moq-config`
  endpoint, or compute it from the start-response `transport: "moq"`
  context. Probably the second — simpler.
- **Per-call participant id allocation.** Server-allocated via
  `/start`, or client-generated UUID? Today we hardcode `bot0` /
  `client0`. The unified `/start` already returns `sessionId` so we
  could derive participant id from that.
- **Cert pinning UX in `MoqTransport` constructor.** Need a
  `serverCertificateHashes` option that the dev workflow sets and prod
  apps leave undefined.

## What this PR's `moq_prebuilt/` becomes

Everything in `moq_prebuilt/` is throwaway once the transport package
exists. Specifically:

- `moq_prebuilt/client/moq-client.js` — hand-rolled moq-lite-02 codec.
  Replaced by `@moq/lite`.
- `moq_prebuilt/client/app.js` — mic capture, audio playback, RTVI
  parsing, transcript DOM. Replaced by `@moq/publish` (mic) + raw-PCM
  subscribe on `@moq/lite` (bot audio) + `@pipecat-ai/moq-transport`
  (lifecycle + events) + `ConsoleTemplate` (UI).
- `moq_prebuilt/client/index.html` / `style.css` — replaced by the
  React `ConsoleTemplate`.
- `moq_prebuilt/frontend.py` — replaced by the existing
  `pipecat-prebuilt` mount.
- `moq_prebuilt/__init__.py` — gone with the rest.

Keep `src/pipecat/transports/moq/` (Python bot side) and
`scripts/moq-dev-setup.sh` (relay setup) as-is.

## Suggested next step (if/when committing to the work)

Skip the recon (it's done — this doc captures the findings) and start
straight on Day 1. Spike the TS package skeleton + `@moq/lite` wrapping
against `transport.ts`'s lifecycle methods first; if you hit a snag
within four hours, the rest of the estimate is suspect and you should
re-scope before continuing.
