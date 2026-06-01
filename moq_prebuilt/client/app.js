/*
 * Copyright (c) 2024-2026, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/**
 * MOQ browser client for the Pipecat MOQ transport.
 *
 * Mic capture + Opus encoding + publish all comes from @moq/publish's
 * Publish.Broadcast (which internally drives @moq/publish/Audio.Encoder).
 * Bot-audio playback is a thin custom loop on top of @moq/hang's
 * Container.Consumer + the browser's WebCodecs AudioDecoder.
 *
 * Connection management uses Net.Connection.Reload so a dropped
 * connection re-establishes automatically. Catalog discovery (instead
 * of hard-coded track names) lets the bot pick its codec/sample rate;
 * we just decode whatever it advertises.
 *
 * Only @moq/publish is imported at runtime. As of 0.2.10 it re-exports
 * `Net`, `Hang`, and `Signals` so we get a single version pin across
 * the whole moq stack — no per-package semver drift. For production
 * use a local Vite build instead of esm.sh; see
 * moq_prebuilt/PLAN-moq-transport-package.md for the path.
 */

import * as Publish from "https://esm.sh/@moq/publish@0.2.10";

const { Net, Hang, Signals } = Publish;
const { Catalog, Container } = Hang;
const { Effect, Signal } = Signals;

// ---------------------------------------------------------------------------
// State (cleared by doDisconnect)
// ---------------------------------------------------------------------------

let reload = null;            // Net.Connection.Reload — auto-reconnects
let publishBroadcast = null;  // Publish.Broadcast — mic → bot
let microphone = null;        // Publish.Source.Microphone — getUserMedia source
let consumeEffect = null;     // Effect — re-runs consume loops on each reconnect
let statusEffect = null;      // Effect — mirrors reload.status to the DOM
let playbackContext = null;
let playbackTime = 0;
let config = {};

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

function setStatus(text, className) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.className = "status " + (className || "");
}

function log(msg, level = "info") {
  if (level === "error") console.error("[moq]", msg);
  else if (level === "warn") console.warn("[moq]", msg);
  else console.log("[moq]", msg);

  const el = document.getElementById("log");
  const line = document.createElement("div");
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  if (level === "error") line.style.color = "#e74c3c";
  else if (level === "warn") line.style.color = "#f39c12";
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

function setButtonsConnected(isConnected) {
  document.getElementById("connectBtn").disabled = isConnected;
  document.getElementById("disconnectBtn").disabled = !isConnected;
}

// ---------------------------------------------------------------------------
// Transcript rendering — RTVI message → conversation rows
// ---------------------------------------------------------------------------

// A user "turn" runs from the first user-transcription until the bot starts
// speaking; Deepgram can emit several finals inside one utterance, so we
// accumulate them in a single bubble. Mirrors voice-ui-kit's
// ConversationProvider semantics.
const turnState = {
  assistantRow: null,
  assistantText: "",
  userRow: null,
  userCommitted: "",
  userInterim: "",
};

function newTranscriptRow(role) {
  const el = document.getElementById("transcript");
  if (!el) return null;
  const row = document.createElement("div");
  const isUser = role === "user";
  row.className = `transcript-row transcript-${isUser ? "user" : "assistant"}`;
  const label = document.createElement("span");
  label.className = "transcript-role";
  label.textContent = isUser ? "user" : "assistant";
  const body = document.createElement("span");
  body.className = "transcript-text";
  row.appendChild(label);
  row.appendChild(body);
  el.appendChild(row);
  el.scrollTop = el.scrollHeight;
  return row;
}

function setRowText(row, text) {
  if (!row) return;
  const body = row.querySelector(".transcript-text");
  if (body) body.textContent = text;
  const el = document.getElementById("transcript");
  if (el) el.scrollTop = el.scrollHeight;
}

function clearTranscript() {
  const el = document.getElementById("transcript");
  if (el) {
    while (el.firstChild) el.removeChild(el.firstChild);
  }
  turnState.assistantRow = null;
  turnState.assistantText = "";
  turnState.userRow = null;
  turnState.userCommitted = "";
  turnState.userInterim = "";
}

// Some LLM tokens carry leading whitespace already; avoid double spaces.
function appendStreamingText(prev, chunk) {
  if (!prev) return chunk;
  if (!chunk) return prev;
  const needsSpace = !prev.endsWith(" ") && !/^[\s.,!?;:'")\]]/.test(chunk);
  return prev + (needsSpace ? " " : "") + chunk;
}

function handleRtviMessage(msg) {
  if (!msg || typeof msg !== "object") return;
  const type = msg.type;
  const data = msg.data || {};

  switch (type) {
    case "bot-llm-started":
      turnState.assistantRow = null;
      turnState.assistantText = "";
      turnState.userRow = null;
      turnState.userCommitted = "";
      turnState.userInterim = "";
      break;
    case "bot-llm-text":
      if (!data.text) break;
      if (!turnState.assistantRow) {
        turnState.assistantRow = newTranscriptRow("assistant");
      }
      turnState.assistantText = appendStreamingText(turnState.assistantText, data.text);
      setRowText(turnState.assistantRow, turnState.assistantText);
      break;
    case "bot-llm-stopped":
      turnState.assistantRow = null;
      turnState.assistantText = "";
      break;
    case "user-transcription": {
      const text = data.text || "";
      const final = data.final !== false;
      if (!turnState.userRow) {
        turnState.userRow = newTranscriptRow("user");
        turnState.userCommitted = "";
        turnState.userInterim = "";
      }
      if (final) {
        turnState.userCommitted = turnState.userCommitted
          ? appendStreamingText(turnState.userCommitted, text)
          : text;
        turnState.userInterim = "";
        setRowText(turnState.userRow, turnState.userCommitted);
      } else {
        turnState.userInterim = text;
        const combined = turnState.userCommitted
          ? appendStreamingText(turnState.userCommitted, text)
          : text;
        setRowText(turnState.userRow, combined);
      }
      break;
    }
    default:
      // Ignore other RTVI messages (metrics, speech start/stop, bot-output,
      // bot-tts-*, function calls, ...).
      break;
  }
}

// ---------------------------------------------------------------------------
// Bot audio playback — Opus → WebCodecs AudioDecoder → Web Audio
// ---------------------------------------------------------------------------

async function consumeBotAudio(botBroadcast, signal) {
  // Read the catalog once to find an Opus audio track. The bot's catalog
  // is published by moq-rs's publish_audio, which translates the OpusHead
  // we sent into a Catalog.AudioConfig.
  const catalogTrack = botBroadcast.subscribe(
    "catalog.json",
    Catalog.PRIORITY?.catalog ?? 0,
  );
  signal.addEventListener("abort", () => {
    try { catalogTrack.close(); } catch {}
  });
  const catalogFrame = await catalogTrack.readFrame();
  catalogTrack.close();
  if (!catalogFrame || signal.aborted) return;

  let catalog;
  try {
    catalog = Catalog.decode(catalogFrame);
  } catch (e) {
    log(`Catalog decode failed: ${e.message}`, "error");
    return;
  }

  const audio = catalog?.audio?.renditions ?? {};
  const trackName = Object.keys(audio)[0];
  if (!trackName) {
    log("Bot catalog has no audio track", "warn");
    return;
  }
  const audioConfig = audio[trackName];
  log(
    `Bot audio: track=${trackName}, codec=${audioConfig.codec}, ` +
    `rate=${audioConfig.sampleRate}Hz, channels=${audioConfig.numberOfChannels}`,
  );

  // Subscribe and decode frames with bounded latency. The latency signal
  // here is the browser-side jitter buffer (matches the bot's
  // audio_in_max_latency_ms in spirit).
  const moqTrack = botBroadcast.subscribe(trackName, Catalog.PRIORITY?.audio ?? 1);
  const latencySig = new Signal(80);
  const consumer = new Container.Consumer(moqTrack, {
    format: new Container.Legacy.Format(),
    latency: latencySig,
  });

  if (!playbackContext) {
    playbackContext = new AudioContext({ sampleRate: audioConfig.sampleRate });
    playbackTime = playbackContext.currentTime;
  }

  const decoder = new AudioDecoder({
    output: (data) => playAudioData(data),
    error: (err) => log(`Audio decode error: ${err.message}`, "error"),
  });
  // Opus in Container.Legacy uses raw packets — no description needed.
  decoder.configure({
    codec: audioConfig.codec,
    sampleRate: audioConfig.sampleRate,
    numberOfChannels: audioConfig.numberOfChannels,
  });

  signal.addEventListener("abort", () => {
    try { consumer.close(); } catch {}
    try { moqTrack.close(); } catch {}
    try { if (decoder.state !== "closed") decoder.close(); } catch {}
  });

  let received = 0;
  try {
    for (;;) {
      const next = await consumer.next();
      if (!next || signal.aborted) break;
      const { frame } = next;
      if (!frame) continue;

      decoder.decode(
        new EncodedAudioChunk({
          type: "key",
          data: frame.data,
          timestamp: frame.timestamp,
        }),
      );
      received++;
      if (received <= 3 || received % 200 === 0) {
        log(`Audio RX #${received}: ${frame.data.byteLength} bytes`);
      }
    }
  } catch (e) {
    if (!signal.aborted) log(`Audio loop error: ${e.message}`, "warn");
  }
}

function playAudioData(data) {
  // Convert WebCodecs AudioData → AudioBuffer and schedule on the audio
  // context. The bot caps the consumer's latency (audio_in_max_latency_ms),
  // so we just chain buffers as they arrive.
  if (!playbackContext) {
    data.close();
    return;
  }
  const channels = data.numberOfChannels;
  const buf = playbackContext.createBuffer(channels, data.numberOfFrames, data.sampleRate);
  for (let ch = 0; ch < channels; ch++) {
    const chData = new Float32Array(data.numberOfFrames);
    data.copyTo(chData, { planeIndex: ch, format: "f32-planar" });
    buf.copyToChannel(chData, ch);
  }
  data.close();

  const src = playbackContext.createBufferSource();
  src.buffer = buf;
  src.connect(playbackContext.destination);
  if (playbackTime < playbackContext.currentTime) playbackTime = playbackContext.currentTime;
  src.start(playbackTime);
  playbackTime += buf.duration;
}

async function consumeBotTranscript(botBroadcast, signal) {
  const trackName = config.transcript_track || "transcript";
  const track = botBroadcast.subscribe(trackName, 0);
  signal.addEventListener("abort", () => {
    try { track.close(); } catch {}
  });

  try {
    for (;;) {
      const group = await track.recvGroup();
      if (!group || signal.aborted) break;
      for (;;) {
        const frame = await group.readFrame();
        if (!frame) break;
        try {
          const text = new TextDecoder().decode(frame);
          handleRtviMessage(JSON.parse(text));
        } catch (err) {
          log(`Transcript frame decode error: ${err.message}`, "warn");
        }
      }
    }
  } catch (e) {
    if (!signal.aborted) log(`Transcript loop ended: ${e.message}`, "warn");
  }
}

// ---------------------------------------------------------------------------
// Connect / disconnect
// ---------------------------------------------------------------------------

async function doConnect() {
  const t0 = performance.now();
  clearTranscript();

  try {
    log("Fetching /api/config...");
    config = await (await fetch("/api/config")).json();
    log(
      `Config: ${config.serve ? "bot-as-server" : "relay"}=${config.relay_host}:${config.relay_port} ` +
      `path="${config.path}" ns="${config.namespace}" ` +
      `client_id="${config.client_id}" bot_id="${config.bot_id}" ` +
      `insecure=${config.insecure} cert=${config.cert_hash ? "pinned" : "none"}`,
    );

    log("Starting bot via /start...");
    const startResp = await fetch("/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ namespace: config.namespace }),
    });
    if (!startResp.ok) throw new Error(`/start returned HTTP ${startResp.status}`);
    const startInfo = await startResp.json();
    log(`Bot started: session=${startInfo.sessionId}, relay=${startInfo.relay}`);

    const url = new URL(
      `https://${config.relay_host}:${config.relay_port}${config.path || "/moq"}`,
    );
    log(`Connecting to ${url}`);

    const webtransport = {};
    if (config.cert_hash) {
      const hashBytes = Uint8Array.from(atob(config.cert_hash), (c) => c.charCodeAt(0));
      webtransport.serverCertificateHashes = [
        { algorithm: "sha-256", value: hashBytes.buffer },
      ];
      log(`Using certificate pinning: sha256=${config.cert_hash.slice(0, 16)}...`);
    }

    // Reload auto-reconnects on disconnect. Publish.Broadcast and the
    // consume effect below both react to its `established` signal.
    reload = new Net.Connection.Reload({
      enabled: new Signal(true),
      url: new Signal(url),
      webtransport,
    });

    // Mirror Reload's status onto the UI.
    statusEffect = new Effect();
    statusEffect.run((eff) => {
      const status = eff.get(reload.status);
      if (status === "connected") setStatus("Connected", "connected");
      else if (status === "connecting") setStatus("Connecting...", "connecting");
      else setStatus("Disconnected", "disconnected");
    });

    const ourPath = Net.Path.from(config.namespace, config.client_id);
    const botPath = Net.Path.from(config.namespace, config.bot_id);

    // ----------------------------------------------------------------------
    // Publish — @moq/publish.Broadcast handles mic capture, Opus encoding,
    // catalog publishing, and serving subscribe requests. It re-attaches
    // automatically when reload.established flips after a reconnect.
    // ----------------------------------------------------------------------

    microphone = new Publish.Source.Microphone({ enabled: new Signal(true) });
    publishBroadcast = new Publish.Broadcast({
      connection: reload.established,
      enabled: new Signal(true),
      name: new Signal(ourPath),
      audio: {
        source: microphone.source,
        enabled: new Signal(true),
      },
    });
    log(`Publishing broadcast: ${ourPath} (mic via @moq/publish)`);

    // ----------------------------------------------------------------------
    // Consume — re-run the loops on each successful (re)connect.
    // ----------------------------------------------------------------------

    consumeEffect = new Effect();
    consumeEffect.run((eff) => {
      const conn = eff.get(reload.established);
      if (!conn) return;

      const botBroadcast = conn.consume(botPath);
      log(`Consuming broadcast: ${botPath}`);

      const ac = new AbortController();
      consumeBotAudio(botBroadcast, ac.signal).catch((e) =>
        log(`Bot audio loop: ${e.message}`, "warn"),
      );
      consumeBotTranscript(botBroadcast, ac.signal).catch((e) =>
        log(`Transcript loop: ${e.message}`, "warn"),
      );

      eff.cleanup(() => ac.abort());
    });

    log(`Setup complete (${(performance.now() - t0).toFixed(0)}ms)`);
    setButtonsConnected(true);

  } catch (e) {
    const elapsed = (performance.now() - t0).toFixed(0);
    log(`Connection failed after ${elapsed}ms: ${e.message}`, "error");
    if (e.stack) log(`  Stack: ${e.stack.split("\n").slice(1, 3).join(" | ")}`, "error");
    setStatus("Error: " + e.message, "error");
    // Re-enable Connect so the user can retry without refreshing.
    setButtonsConnected(false);
    // Clean up any partial state.
    await doDisconnect({silent: true});
  }
}

async function doDisconnect({silent = false} = {}) {
  if (!silent) log("Disconnecting...");

  if (consumeEffect) { consumeEffect.close(); consumeEffect = null; }
  if (statusEffect) { statusEffect.close(); statusEffect = null; }
  if (publishBroadcast) { publishBroadcast.close(); publishBroadcast = null; }
  if (microphone) { microphone.close(); microphone = null; }
  if (reload) {
    try { reload.signals.close(); } catch {}
    reload = null;
  }
  if (playbackContext) {
    try { await playbackContext.close(); } catch {}
    playbackContext = null;
  }

  if (!silent) {
    setStatus("Disconnected", "disconnected");
    log("Disconnected");
  }
  setButtonsConnected(false);
}

// ---------------------------------------------------------------------------
// Button handlers
// ---------------------------------------------------------------------------

document.getElementById("connectBtn").addEventListener("click", async () => {
  setButtonsConnected(true);
  await doConnect();
});

document.getElementById("disconnectBtn").addEventListener("click", async () => {
  await doDisconnect();
});
