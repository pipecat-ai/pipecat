// Browser measuring client for the transport latency benchmark.
//
// Runs the pipecat client SDK (@pipecat-ai/client-js with the SmallWebRTC or
// MoQ transport) against the echo bot, exactly as a real web client would,
// and exposes a `window.bench` API for the Playwright driver
// (browser_client.py).
//
// Two methods share this page:
// - method "b" (in-page): the probe is generated on the AudioContext clock
//   and fed to the SDK as a synthetic mic track; the bot's returned track is
//   tapped into a recorder AudioWorklet stamping 20 ms chunks on the same
//   clock. Send times are implied by the buffer schedule (t0 + i * 20 ms), so
//   one clock covers both directions and Python reuses its normal analysis.
// - method "a" (device loopback): the SDK captures a real input device
//   (BlackHole) and plays bot audio to a real output device; Python owns the
//   probe and the clock via sounddevice. The page is only the connector.
//
// Query params: transport=smallwebrtc|moq, method=a|b, server=<bot base url>,
// jitter_ms, turn_url/turn_username/turn_credential + force_relay=1,
// mic_label/spk_label (method a).

import { PipecatClient } from "@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";
import { MoqTransport } from "@pipecat-ai/moq-transport";

const SAMPLE_RATE = 48000;
const CHUNK_S = 0.02;

const params = new URLSearchParams(location.search);
const P = {
  transport: params.get("transport") || "smallwebrtc",
  method: params.get("method") || "b",
  server: params.get("server") || "http://localhost:7860",
  jitterMs: parseInt(params.get("jitter_ms") || "60", 10),
  turnUrl: params.get("turn_url"),
  turnUsername: params.get("turn_username"),
  turnCredential: params.get("turn_credential"),
  forceRelay: params.get("force_relay") === "1",
  micLabel: params.get("mic_label"),
  spkLabel: params.get("spk_label"),
  certHashHex: params.get("cert_hash_hex"),
};

const statusEl = document.getElementById("status");
function status(msg) {
  statusEl.textContent += `\n${msg}`;
  console.log(`[bench] ${msg}`);
}

// ---------------------------------------------------------------------------
// Probe generation — must mirror probe.py exactly (same marker positions).

function chirpTemplate(chirpMs = 20) {
  const n = Math.floor((SAMPLE_RATE * chirpMs) / 1000);
  const out = new Float32Array(n);
  const f0 = 300.0;
  const f1 = 3000.0;
  const T = n / SAMPLE_RATE;
  for (let i = 0; i < n; i++) {
    const t = i / SAMPLE_RATE;
    const phase = 2 * Math.PI * (f0 * t + ((f1 - f0) / (2 * T)) * t * t);
    const hann = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
    out[i] = Math.sin(phase) * hann;
  }
  return out;
}

function genProbe(durationS, { periodMs = 250, amplitude = 0.5, gapEveryS = null, gapS = 2.0 } = {}) {
  const total = Math.floor(durationS * SAMPLE_RATE);
  const template = chirpTemplate();
  const period = Math.floor((SAMPLE_RATE * periodMs) / 1000);
  const signal = new Float32Array(total);
  const positions = [];
  let pos = period;
  let spurtStart = pos;
  while (pos + template.length <= total) {
    if (gapEveryS !== null && pos - spurtStart >= Math.floor(gapEveryS * SAMPLE_RATE)) {
      pos += Math.floor(gapS * SAMPLE_RATE);
      spurtStart = pos;
      continue;
    }
    for (let i = 0; i < template.length; i++) signal[pos + i] += template[i] * amplitude;
    positions.push(pos);
    pos += period;
  }
  return { signal, positions };
}

// ---------------------------------------------------------------------------
// RTCPeerConnection wrapper: records instances for stats and, when relay is
// forced, pins iceTransportPolicy so media must traverse the TURN allocation
// (the browser twin of the candidate stripping in webrtc_client.py).

const pcs = [];
const NativePC = window.RTCPeerConnection;
window.RTCPeerConnection = class extends NativePC {
  constructor(config = {}, ...rest) {
    if (P.forceRelay) config = { ...config, iceTransportPolicy: "relay" };
    super(config, ...rest);
    pcs.push(this);
  }
};

// getUserMedia wrapper: method "b" swaps in the synthetic probe stream so the
// SDK publishes our signal instead of a physical mic; method "a" pins the
// requested input device.
const nativeGum = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
navigator.mediaDevices.getUserMedia = async (constraints) => {
  if (P.method === "b" && constraints && constraints.audio) {
    if (!window.__benchProbeBus) throw new Error("bench probe bus not ready");
    // Fresh MediaStreamAudioDestinationNode per call, fed from the shared
    // probe bus: transports stop() "their" mic track on capture re-init (some
    // re-init several times, with cleanups racing new inits), so every caller
    // must get an independent track — stopping one can never silence the
    // source or another caller's track. Pin getSettings() to stable values;
    // WebAudio tracks report sparse settings, and reactive capture pipelines
    // rebuild on every settings change.
    const dest = new MediaStreamAudioDestinationNode(window.__benchCtx, {
      channelCount: 1,
      channelCountMode: "explicit",
    });
    window.__benchProbeBus.connect(dest);
    const track = dest.stream.getAudioTracks()[0];
    const settings = {
      deviceId: "bench-synthetic",
      groupId: "bench",
      sampleRate: SAMPLE_RATE,
      channelCount: 1,
      sampleSize: 16,
      latency: 0.01,
    };
    track.getSettings = () => settings;
    return dest.stream;
  }
  if (P.method === "a" && constraints && constraints.audio && P.micLabel) {
    const deviceId = await deviceIdByLabel("audioinput", P.micLabel);
    const audio = typeof constraints.audio === "object" ? constraints.audio : {};
    return nativeGum({ ...constraints, audio: { ...audio, deviceId: { exact: deviceId } } });
  }
  return nativeGum(constraints);
};

async function deviceIdByLabel(kind, label) {
  // Labels are only populated once some gUM call has been granted.
  await nativeGum({ audio: true }).then((s) => s.getTracks().forEach((t) => t.stop()));
  const devices = await navigator.mediaDevices.enumerateDevices();
  const dev = devices.find((d) => d.kind === kind && d.label.includes(label));
  if (!dev) {
    const seen = devices.filter((d) => d.kind === kind).map((d) => d.label);
    throw new Error(`no ${kind} device matching "${label}" (saw: ${seen.join(", ")})`);
  }
  return dev.deviceId;
}

// ---------------------------------------------------------------------------
// Bench harness.

const bench = {
  ready: true,
  error: null,
  _ctx: null,
  _client: null,
  _botTrack: null,
  _recorderNode: null,
  _recvPcm: [],
  _recvStamps: [],
  _recvDone: null,
  _joinStartT: null,
  _trackStartT: null,
  _sendT0: null,
  _probe: null,
  _moqConfig: null,

  async setup() {
    const ctx = new AudioContext({ sampleRate: SAMPLE_RATE, latencyHint: "interactive" });
    this._ctx = ctx;
    await ctx.resume();

    if (P.method === "b") {
      // Probe bus: run() plays the probe into this gain node; every
      // getUserMedia call taps it via its own destination node.
      window.__benchCtx = ctx;
      window.__benchProbeBus = new GainNode(ctx, { gain: 1 });
      this._probeBus = window.__benchProbeBus;
      await ctx.audioWorklet.addModule("./recorder-worklet.js");
    }

    let transport;
    let connectParams;
    if (P.transport === "moq") {
      const resp = await fetch(`${P.server}/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      if (!resp.ok) throw new Error(`/start failed: ${resp.status}`);
      connectParams = await resp.json();
      if (P.certHashHex && !connectParams.moq.certHash) {
        // Bot dialed a self-signed relay insecurely; pin the relay cert
        // ourselves (transport expects base64 of the raw hash bytes).
        const bytes = P.certHashHex.match(/../g).map((h) => parseInt(h, 16));
        connectParams.moq.certHash = btoa(String.fromCharCode(...bytes));
      }
      this._moqConfig = connectParams.moq;
      transport = new MoqTransport({
        relayUrl: connectParams.moq.relayUrl,
        audioLatencyMs: P.jitterMs,
        audioBufferMaxMs: "real-time",
      });
    } else {
      // ICE servers must go in constructor opts: the transport builds its
      // RTCPeerConnection during device init, before connect() reads
      // iceConfig from the connect params.
      const iceServers = P.turnUrl
        ? [{ urls: P.turnUrl, username: P.turnUsername, credential: P.turnCredential }]
        : [];
      transport = new SmallWebRTCTransport({ iceServers });
      connectParams = { connectionUrl: `${P.server}/api/offer` };
    }

    const trackReady = new Promise((resolve) => (this._resolveTrack = resolve));
    this._client = new PipecatClient({
      transport,
      enableMic: true,
      enableCam: false,
      callbacks: {
        onTrackStarted: (track, participant) => {
          if (track.kind === "audio" && (!participant || !participant.local)) {
            this._resolveTrack(track);
          }
        },
        onError: (msg) => status(`client error: ${JSON.stringify(msg)}`),
      },
    });

    this._joinStartT = ctx.currentTime;
    this._joinStartPerf = performance.now();
    await this._client.connect(connectParams);
    status(`connected (${P.transport})`);

    // SmallWebRTC surfaces the bot track via onTrackStarted; MoqTransport
    // only exposes it through tracks() once the decoder graph is up — poll
    // both paths.
    const polled = (async () => {
      for (;;) {
        const tracks =
          (this._client.tracks && this._client.tracks()) ||
          (this._client.transport &&
            this._client.transport.tracks &&
            this._client.transport.tracks()) ||
          {};
        if (tracks.bot && tracks.bot.audio) return tracks.bot.audio;
        await new Promise((r) => setTimeout(r, 200));
      }
    })();
    this._botTrack = await Promise.race([
      trackReady,
      polled,
      new Promise((_, rej) => setTimeout(() => rej(new Error("bot track timeout")), 20000)),
    ]);
    status(`bot track started: ${this._botTrack.id}`);

    // Chrome only reliably decodes/advances a remote WebRTC track that is
    // attached to a media element; keep one muted for both transports.
    this._audioEl = new Audio();
    this._audioEl.srcObject = new MediaStream([this._botTrack]);
    this._audioEl.autoplay = true;

    if (P.method === "a") {
      if (P.spkLabel) {
        const sinkId = await deviceIdByLabel("audiooutput", P.spkLabel);
        await this._audioEl.setSinkId(sinkId);
        status(`sink -> ${P.spkLabel}`);
      }
    } else {
      this._audioEl.muted = true;
      const src = new MediaStreamAudioSourceNode(ctx, {
        mediaStream: new MediaStream([this._botTrack]),
      });
      this._recorderNode = new AudioWorkletNode(ctx, "bench-recorder", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        channelCount: 1,
      });
      this._recvDone = new Promise((resolve) => {
        this._recorderNode.port.onmessage = (e) => {
          const { pcm, stamps, final } = e.data;
          if (pcm.length) {
            this._recvPcm.push(pcm);
            this._recvStamps.push(...stamps);
            if (this._trackStartT === null && stamps.length) this._trackStartT = stamps[0];
          }
          if (final) resolve();
        };
      });
      src.connect(this._recorderNode);
      // Keep the graph pulled without audible output.
      const mute = new GainNode(ctx, { gain: 0 });
      this._recorderNode.connect(mute).connect(ctx.destination);
    }
    return { state: "ready", transport: P.transport, method: P.method };
  },

  // Method B only: play the probe through the synthetic mic and record until
  // duration + tail has elapsed. Returns scheduling facts; PCM is fetched
  // separately via result().
  async run(durationS, { periodMs = 250, gapEveryS = null, gapS = 2.0, tailS = 1.0 } = {}) {
    const ctx = this._ctx;
    const { signal, positions } = genProbe(durationS, { periodMs, gapEveryS, gapS });
    this._probe = { positions, durationS, gapEveryS, gapS };

    const buf = new AudioBuffer({ length: signal.length, sampleRate: SAMPLE_RATE, numberOfChannels: 1 });
    buf.copyToChannel(signal, 0);
    const srcNode = new AudioBufferSourceNode(ctx, { buffer: buf });
    srcNode.connect(this._probeBus);

    const t0 = ctx.currentTime + 0.3;
    srcNode.start(t0);
    this._sendT0 = t0;
    this._srcNode = srcNode;
    status(`probe started at ctx t=${t0.toFixed(3)}`);

    // Deterministic wait — onended is unreliable if rendering hiccups, and a
    // lost worklet "final" message must not hang the trial.
    const totalS = signal.length / SAMPLE_RATE;
    await new Promise((r) => setTimeout(r, (0.3 + totalS + tailS) * 1000));
    this._recorderNode.port.postMessage("flush");
    await Promise.race([this._recvDone, new Promise((r) => setTimeout(r, 5000))]);
    return {
      sendT0: this._sendT0,
      chunkS: CHUNK_S,
      nChunks: Math.floor(signal.length / (SAMPLE_RATE * CHUNK_S)),
      markerPositions: positions,
    };
  },

  // Everything Python needs to run its normal analysis, PCM as base64 s16le.
  async result() {
    const total = this._recvPcm.reduce((a, c) => a + c.length, 0);
    const f32 = new Float32Array(total);
    let off = 0;
    for (const c of this._recvPcm) {
      f32.set(c, off);
      off += c.length;
    }
    const i16 = new Int16Array(total);
    for (let i = 0; i < total; i++) {
      const v = Math.max(-1, Math.min(1, f32[i]));
      i16[i] = Math.round(v * 32767);
    }
    const bytes = new Uint8Array(i16.buffer);
    let b64 = "";
    // Multiple of 3 so no slice emits base64 padding mid-string.
    const SLICE = 32766;
    for (let i = 0; i < bytes.length; i += SLICE) {
      b64 += btoa(String.fromCharCode.apply(null, bytes.subarray(i, i + SLICE)));
    }
    return {
      sendT0: this._sendT0,
      chunkS: CHUNK_S,
      markerPositions: this._probe ? this._probe.positions : null,
      joinStartT: this._joinStartT,
      trackStartT: this._trackStartT,
      recvStamps: this._recvStamps,
      recvPcmB64: b64,
      moqConfig: this._moqConfig,
      webrtc: await this._webrtcDiagnostics(),
      userAgent: navigator.userAgent,
    };
  },

  // Diagnostic-only (never charted against MoQ): selected candidate pair for
  // topology assertion, NetEq jitterBufferDelay for context.
  async _webrtcDiagnostics() {
    if (P.transport !== "smallwebrtc" || !pcs.length) return null;
    const stats = await pcs[pcs.length - 1].getStats();
    const byId = {};
    stats.forEach((r) => (byId[r.id] = r));
    let pair = null;
    let inbound = null;
    stats.forEach((r) => {
      if (r.type === "transport" && r.selectedCandidatePairId) {
        pair = byId[r.selectedCandidatePairId];
      }
      if (r.type === "inbound-rtp" && r.kind === "audio") inbound = r;
    });
    if (!pair) {
      stats.forEach((r) => {
        if (r.type === "candidate-pair" && (r.selected || r.nominated) && r.state === "succeeded") {
          pair = pair || r;
        }
      });
    }
    const local = pair ? byId[pair.localCandidateId] : null;
    return {
      icePair: pair
        ? {
            local: local?.candidateType,
            remote: byId[pair.remoteCandidateId]?.candidateType,
            // Chrome can select a prflx local whose base is the TURN
            // allocation; the port tells whether media crosses the relay.
            localPort: local?.port,
            localRelatedPort: local?.relatedPort,
          }
        : null,
      jitterBufferDelaySeconds: inbound?.jitterBufferDelay,
      jitterBufferEmittedCount: inbound?.jitterBufferEmittedCount,
    };
  },

  async teardown() {
    try {
      if (this._client) await this._client.disconnect();
    } catch (e) {
      status(`disconnect error: ${e}`);
    }
    if (this._ctx) await this._ctx.close();
  },
};

// Surface async failures to the driver instead of hanging it.
for (const name of ["setup", "run", "result", "teardown"]) {
  const fn = bench[name].bind(bench);
  bench[name] = async (...args) => {
    try {
      return await fn(...args);
    } catch (e) {
      bench.error = `${name}: ${e && e.stack ? e.stack : e}`;
      status(bench.error);
      throw e;
    }
  };
}

window.bench = bench;
status(`bench ready: transport=${P.transport} method=${P.method} server=${P.server}`);
