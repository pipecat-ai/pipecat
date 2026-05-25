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
 * Catalog discovery (instead of hard-coded track names) lets the bot
 * pick its codec/sample rate; we just decode whatever it advertises.
 */

import * as Moq from "https://esm.sh/@moq/net@0.1.0";
import * as Publish from "https://esm.sh/@moq/publish@0.2.7";
import * as Catalog from "https://esm.sh/@moq/hang@0.4.0/catalog";
import * as Container from "https://esm.sh/@moq/hang@0.4.0/container";
import { Signal } from "https://esm.sh/@moq/signals@0.1.0";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let connection = null;
let connectionSignal = null;
let publishBroadcast = null;
let microphone = null;
let playbackContext = null;
let playbackTime = 0;
let consumeAbort = null;
let connected = false;
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
  // is published by moq-rs's publish_media, which translates the OpusHead
  // we sent into a Catalog.AudioConfig.
  const catalogTrack = botBroadcast.subscribe(
    "catalog.json",
    Catalog.PRIORITY?.catalog ?? 0,
  );
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

  // Subscribe to the audio track and decode frames with bounded latency.
  const moqTrack = botBroadcast.subscribe(trackName, Catalog.PRIORITY?.audio ?? 1);
  const latencySig = new Signal(80); // ms — matches the bot's max_latency_ms order of magnitude
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
    try {
      consumer.close();
      moqTrack.close();
      if (decoder.state !== "closed") decoder.close();
    } catch {}
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
  // Convert WebCodecs AudioData → AudioBuffer and schedule on the audio context.
  // For low latency we just chain buffers back-to-back; the bot is responsible
  // for keeping output paced.
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
    try {
      track.close();
    } catch {}
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
    setStatus("Connecting...", "connecting");

    const webtransport = {};
    if (config.cert_hash) {
      const hashBytes = Uint8Array.from(atob(config.cert_hash), (c) => c.charCodeAt(0));
      webtransport.serverCertificateHashes = [
        { algorithm: "sha-256", value: hashBytes.buffer },
      ];
      log(`Using certificate pinning: sha256=${config.cert_hash.slice(0, 16)}...`);
    }

    connection = await Moq.Connection.connect(url, { webtransport });
    log(`Connected (ALPN=${connection.version}, ${(performance.now() - t0).toFixed(0)}ms)`);
    setStatus("Connected", "connected");
    connected = true;
    connectionSignal = new Signal(connection);

    const ourPath = Moq.Path.from(config.namespace, config.client_id);
    const botPath = Moq.Path.from(config.namespace, config.bot_id);

    // ----------------------------------------------------------------------
    // Publish side — @moq/publish does mic capture + Opus encoding + catalog
    // ----------------------------------------------------------------------

    microphone = new Publish.Source.Microphone({ enabled: new Signal(true) });
    publishBroadcast = new Publish.Broadcast({
      connection: connectionSignal,
      enabled: new Signal(true),
      name: new Signal(ourPath),
      audio: {
        source: microphone.source,
        enabled: new Signal(true),
      },
      // No video.
    });
    log(`Publishing broadcast: ${ourPath} (mic via @moq/publish)`);

    // ----------------------------------------------------------------------
    // Consume side — read the catalog, decode Opus, play via Web Audio
    // ----------------------------------------------------------------------

    const botBroadcast = connection.consume(botPath);
    log(`Consuming broadcast: ${botPath}`);

    consumeAbort = new AbortController();
    consumeBotAudio(botBroadcast, consumeAbort.signal).catch((e) =>
      log(`Bot audio loop error: ${e.message}`, "warn"),
    );
    consumeBotTranscript(botBroadcast, consumeAbort.signal).catch((e) =>
      log(`Transcript loop error: ${e.message}`, "warn"),
    );

    // Watch for connection close.
    connection.closed
      .then(() => {
        log("Transport closed");
        setStatus("Disconnected", "disconnected");
      })
      .catch((err) => {
        log(`Transport closed with error: ${err?.message || err}`, "error");
        setStatus("Error: " + (err?.message || err), "error");
      });

  } catch (e) {
    const elapsed = (performance.now() - t0).toFixed(0);
    log(`Connection failed after ${elapsed}ms: ${e.message}`, "error");
    if (e.stack) log(`  Stack: ${e.stack.split("\n").slice(1, 3).join(" | ")}`, "error");
    setStatus("Error: " + e.message, "error");
  }
}

async function doDisconnect() {
  log("Disconnecting...");
  connected = false;

  if (consumeAbort) {
    consumeAbort.abort();
    consumeAbort = null;
  }
  if (publishBroadcast) {
    publishBroadcast.close();
    publishBroadcast = null;
  }
  if (microphone) {
    microphone.close();
    microphone = null;
  }
  if (connection) {
    try {
      connection.close();
    } catch (e) {
      log(`Error closing connection: ${e.message}`, "warn");
    }
    connection = null;
    connectionSignal = null;
  }
  if (playbackContext) {
    try {
      await playbackContext.close();
    } catch {}
    playbackContext = null;
  }

  setStatus("Disconnected", "disconnected");
  log("Disconnected");

  document.getElementById("connectBtn").disabled = false;
  document.getElementById("disconnectBtn").disabled = true;
}

// ---------------------------------------------------------------------------
// Button handlers
// ---------------------------------------------------------------------------

document.getElementById("connectBtn").addEventListener("click", async () => {
  document.getElementById("connectBtn").disabled = true;
  document.getElementById("disconnectBtn").disabled = false;
  await doConnect();
});

document.getElementById("disconnectBtn").addEventListener("click", async () => {
  await doDisconnect();
});
