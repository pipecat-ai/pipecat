/*
 * Copyright (c) 2024-2026, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/**
 * MOQ browser client — moq-lite-02.
 *
 * Connects to a moq-lite relay via WebTransport, captures microphone audio,
 * and plays back received audio using stream-per-request model.
 *
 * Flow:
 *   1. WebTransport connect
 *   2. Open bidi stream for SETUP: write CLIENT_SETUP, read SERVER_SETUP
 *   3. Listen for incoming bidi streams (relay sends SUBSCRIBE / ANNOUNCE)
 *   4. Listen for incoming uni streams (receive bot audio as GROUP + FRAME)
 *   5. Open bidi stream for SUBSCRIBE (to bot's audio track)
 *   6. Send mic PCM as uni streams (GROUP + FRAME per chunk)
 */

const {
  MOQL_VERSION,
  Role,
  StreamType,
  CLIENT_SETUP_TYPE,
  SERVER_SETUP_TYPE,
  concat,
  encodeClientSetup,
  parseServerSetup,
  encodeSubscribe,
  encodeSubscribeOk,
  decodeSubscribe,
  encodeGroupAndFrame,
  parseGroupStream,
  encodeVarint,
  decodeVarint,
  encodeString,
  decodeString,
} = window.MOQ;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let transport = null;
let micStream = null;
let audioContext = null;
let micWorklet = null;
let connected = false;

// Subscription IDs
let nextSubscribeId = 1;

// Publishing state
let pubSubscribeId = null; // set when relay subscribes to our audio
let pubGroupSeq = 0;

// Subscription state (for receiving bot audio and transcript)
let subAudioSubscribeId = null;
let subTranscriptSubscribeId = null;

// Playback
let playbackTime = 0;

// Config (populated from /api/config)
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
  if (level === "error") {
    console.error("[moq]", msg);
  } else if (level === "warn") {
    console.warn("[moq]", msg);
  } else {
    console.log("[moq]", msg);
  }
  const el = document.getElementById("log");
  const line = document.createElement("div");
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  if (level === "error") line.style.color = "#e74c3c";
  else if (level === "warn") line.style.color = "#f39c12";
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

function hexdump(data, maxBytes = 40) {
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  const hex = Array.from(bytes.slice(0, maxBytes)).map(b => b.toString(16).padStart(2, '0')).join(' ');
  return bytes.length > maxBytes ? `${hex}... (${bytes.length} bytes total)` : `${hex} (${bytes.length} bytes)`;
}

// Turn-based transcript: each in-progress LLM/user turn is one row whose
// text accumulates as per-token RTVI messages arrive. Mirrors the
// ConversationProvider model in @pipecat-ai/voice-ui-kit.
//
// A user "turn" runs from the first user-transcription until the bot
// starts speaking — Deepgram can emit several final transcripts inside
// one spoken utterance, and we want all of them in the same bubble.
const turnState = {
  assistantRow: null,        // DOM <div> for the in-progress assistant turn
  assistantText: "",         // accumulated text for the in-progress turn
  userRow: null,             // DOM <div> for the in-progress user turn
  userCommitted: "",         // accumulated final user text within the turn
  userInterim: "",           // latest non-final user text (tentative tail)
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
  if (el) el.innerHTML = "";
  turnState.assistantRow = null;
  turnState.assistantText = "";
  turnState.userRow = null;
  turnState.userCommitted = "";
  turnState.userInterim = "";
}

// Concatenate streaming tokens with a sensible separator. Some LLM tokens
// already include leading whitespace (e.g. " world"); avoid double spaces.
function appendStreamingText(prev, chunk) {
  if (!prev) return chunk;
  if (!chunk) return prev;
  const needsSpace = !prev.endsWith(" ") && !/^[\s.,!?;:'")\]]/.test(chunk);
  return prev + (needsSpace ? " " : "") + chunk;
}

// RTVI message dispatcher. Matches the ConversationProvider semantics from
// @pipecat-ai/voice-ui-kit: per-LLM-turn aggregation for the bot, upsert
// for user transcripts.
function handleRtviMessage(msg) {
  if (!msg || typeof msg !== "object") return;
  const type = msg.type;
  const data = msg.data || {};

  switch (type) {
    case "bot-llm-started":
      // Don't create the row yet — wait for the first bot-llm-text so we
      // don't show an empty "assistant" label if no text follows.
      turnState.assistantRow = null;
      turnState.assistantText = "";
      // Bot is responding: the user's turn just ended. The next
      // user-transcription should start a fresh row.
      turnState.userRow = null;
      turnState.userCommitted = "";
      turnState.userInterim = "";
      break;

    case "bot-llm-text":
      if (!data.text) break;
      if (!turnState.assistantRow) {
        // Lazy: covers both the post-started case and pipelines that emit
        // bot-llm-text without a preceding bot-llm-started.
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
      // bot-tts-*, function calls, etc.). Uncomment to discover what's
      // flowing on the transcript track.
      // log(`RTVI: ${type}`);
      break;
  }
}

// ---------------------------------------------------------------------------
// Audio playback — queue received 16-bit PCM into Web Audio
// ---------------------------------------------------------------------------

function initAudioPlayback() {
  if (audioContext) return;
  audioContext = new AudioContext({ sampleRate: 24000 });
  playbackTime = audioContext.currentTime;
}

function playPcmChunk(pcmBytes) {
  if (!audioContext) return;

  // pcmBytes is 16-bit signed LE PCM — copy to aligned buffer to avoid offset issues
  const aligned = new ArrayBuffer(pcmBytes.byteLength);
  new Uint8Array(aligned).set(pcmBytes);
  const samples = new Int16Array(aligned);
  const floats = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    floats[i] = samples[i] / 32768;
  }

  const buffer = audioContext.createBuffer(1, floats.length, 24000);
  buffer.getChannelData(0).set(floats);

  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(audioContext.destination);

  const now = audioContext.currentTime;
  if (playbackTime < now) playbackTime = now;
  source.start(playbackTime);
  playbackTime += buffer.duration;
}

// ---------------------------------------------------------------------------
// Microphone capture — 16kHz 16-bit PCM via AudioWorklet
// ---------------------------------------------------------------------------

const WORKLET_CODE = `
class PcmCapture extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input.length > 0) {
      const floats = input[0];
      const pcm = new Int16Array(floats.length);
      for (let i = 0; i < floats.length; i++) {
        pcm[i] = Math.max(-32768, Math.min(32767, Math.round(floats[i] * 32768)));
      }
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}
registerProcessor('pcm-capture', PcmCapture);
`;

async function startMic() {
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true },
  });

  const ctx = new AudioContext({ sampleRate: 16000 });
  const blob = new Blob([WORKLET_CODE], { type: "application/javascript" });
  const url = URL.createObjectURL(blob);
  await ctx.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);

  const source = ctx.createMediaStreamSource(micStream);
  micWorklet = new AudioWorkletNode(ctx, "pcm-capture");
  micWorklet.port.onmessage = (e) => {
    if (connected && transport && pubSubscribeId !== null) {
      sendAudioUniStream(new Uint8Array(e.data));
    }
  };
  source.connect(micWorklet);
  micWorklet.connect(ctx.destination);

  log("Microphone started (16kHz PCM)");
}

function stopMic() {
  if (micStream) {
    micStream.getTracks().forEach((t) => t.stop());
    micStream = null;
  }
  if (micWorklet) {
    micWorklet.disconnect();
    micWorklet = null;
  }
}

// ---------------------------------------------------------------------------
// Uni stream send (publish audio)
// ---------------------------------------------------------------------------

let audioSendCount = 0;
let audioSendErrors = 0;

async function sendAudioUniStream(pcmBytes) {
  if (!transport || pubSubscribeId === null) return;

  try {
    const groupSeq = pubGroupSeq++;
    const data = encodeGroupAndFrame(pubSubscribeId, groupSeq, pcmBytes);

    const uni = await transport.createUnidirectionalStream();
    const writer = uni.getWriter();
    await writer.write(data);
    await writer.close();

    audioSendCount++;
    if (audioSendCount % 100 === 0) {
      log(`Audio TX: ${audioSendCount} chunks sent (${pcmBytes.byteLength} bytes/chunk, sub=${pubSubscribeId}, seq=${groupSeq})`);
    }
  } catch (e) {
    audioSendErrors++;
    if (audioSendErrors <= 5) {
      log(`Audio send error #${audioSendErrors}: ${e.message} (sub=${pubSubscribeId}, seq=${pubGroupSeq})`, "warn");
    } else if (audioSendErrors === 6) {
      log("Suppressing further audio send errors...", "warn");
    }
  }
}

// ---------------------------------------------------------------------------
// Uni stream receive (receive bot audio)
// ---------------------------------------------------------------------------

let audioRecvCount = 0;

async function receiveUniStreams() {
  log("Uni stream listener started (waiting for bot audio)");
  const reader = transport.incomingUnidirectionalStreams.getReader();
  try {
    while (true) {
      const { value: stream, done } = await reader.read();
      if (done) {
        log("Uni stream reader finished (relay closed incoming streams)");
        break;
      }
      handleIncomingUniStream(stream);
    }
  } catch (e) {
    if (connected) log(`Uni stream reader stopped: ${e.message}`, "warn");
  }
}

async function handleIncomingUniStream(stream) {
  try {
    // Read all data from the stream
    const reader = stream.getReader();
    let buf = new Uint8Array(0);
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf = concat(buf, new Uint8Array(value));
    }

    if (buf.length === 0) return;

    // Parse GROUP + FRAMEs
    const { subscribeId, groupSeq, frames } = parseGroupStream(buf);

    if (subscribeId === subTranscriptSubscribeId) {
      for (const frame of frames) {
        if (!frame.byteLength) continue;
        try {
          const text = new TextDecoder().decode(frame);
          handleRtviMessage(JSON.parse(text));
        } catch (err) {
          log(`Transcript frame decode error: ${err.message}`, "warn");
        }
      }
      return;
    }

    audioRecvCount++;

    if (audioRecvCount <= 3 || audioRecvCount % 100 === 0) {
      const totalBytes = frames.reduce((sum, f) => sum + f.byteLength, 0);
      log(`Audio RX #${audioRecvCount}: sub=${subscribeId} seq=${groupSeq} frames=${frames.length} bytes=${totalBytes}`);
    }

    for (const frame of frames) {
      if (frame.byteLength > 0) {
        playPcmChunk(frame);
      }
    }
  } catch (e) {
    log(`Uni stream parse error: ${e.message} (buffer may be malformed)`, "warn");
  }
}

// ---------------------------------------------------------------------------
// Incoming bidi stream handler (relay sends SUBSCRIBE / ANNOUNCE)
// ---------------------------------------------------------------------------

let incomingBidiCount = 0;

async function handleIncomingBidiStreams() {
  log("Bidi stream listener started (waiting for relay SUBSCRIBE/ANNOUNCE)");
  const reader = transport.incomingBidirectionalStreams.getReader();
  try {
    while (true) {
      const { value: stream, done } = await reader.read();
      if (done) {
        log("Incoming bidi stream reader finished (relay closed)");
        break;
      }
      incomingBidiCount++;
      log(`Incoming bidi stream #${incomingBidiCount}`);
      handleIncomingBidiStream(stream);
    }
  } catch (e) {
    if (connected) log(`Bidi stream reader stopped: ${e.message}`, "warn");
  }
}

async function handleIncomingBidiStream(stream) {
  try {
    const streamReader = stream.readable.getReader();
    let buf = new Uint8Array(0);

    // Read all available data (relay sends stream type + message body)
    while (true) {
      const { value, done } = await streamReader.read();
      if (value) buf = concat(buf, new Uint8Array(value));
      // Break once we have data or stream ends
      if (done || buf.length > 0) break;
    }

    if (buf.length === 0) {
      log("Incoming bidi stream was empty (0 bytes)", "warn");
      return;
    }

    log(`Incoming bidi data: [${hexdump(buf, 30)}]`);

    // Decode stream type
    const [streamType, offset] = decodeVarint(buf, 0);
    const streamTypeName = streamType === StreamType.SUBSCRIBE ? "SUBSCRIBE"
      : streamType === StreamType.ANNOUNCE ? "ANNOUNCE"
      : streamType === StreamType.SESSION ? "SESSION"
      : `UNKNOWN(${streamType})`;
    log(`Incoming bidi stream type: ${streamTypeName} (${streamType})`);

    if (streamType === StreamType.SUBSCRIBE) {
      // We may need more data for the full subscribe message
      while (buf.length < offset + 3) {
        const { value, done } = await streamReader.read();
        if (value) buf = concat(buf, new Uint8Array(value));
        if (done) break;
      }

      // Relay is subscribing to our track
      const sub = decodeSubscribe(buf, offset);
      log(`SUBSCRIBE from relay: broadcast="${sub.broadcastPath}" track="${sub.trackName}" sub_id=${sub.subscribeId} priority=${sub.priority}`);

      // Accept SUBSCRIBE only for our publish track inside our own
      // broadcast. With per-participant broadcast paths, the relay only
      // routes SUBSCRIBEs targeting <namespace>/<client_id> to us, so a
      // mismatch here means something is misconfigured.
      const ourBroadcast = `${config.namespace}/${config.client_id}`;
      if (sub.broadcastPath !== ourBroadcast || sub.trackName !== config.publish_track) {
        log(`Rejecting SUBSCRIBE for "${sub.broadcastPath}/${sub.trackName}" (we only publish "${ourBroadcast}/${config.publish_track}")`, "warn");
        stream.writable.abort();
        return;
      }

      // Store subscribe_id for publishing
      pubSubscribeId = sub.subscribeId;

      // Send SUBSCRIBE_OK on the writable side
      const writer = stream.writable.getWriter();
      await writer.write(encodeSubscribeOk());
      writer.releaseLock();

      log(`Sent SUBSCRIBE_OK for sub_id=${sub.subscribeId} — ready to publish ${config.publish_track}`);

    } else if (streamType === StreamType.ANNOUNCE) {
      // Relay sending ANNOUNCE_PLEASE — read the full message
      // Minimum is 2 bytes after stream type: varint(body_len) + varint(string_len)
      while (buf.length < offset + 2) {
        const { value, done } = await streamReader.read();
        if (value) buf = concat(buf, new Uint8Array(value));
        if (done) break;
      }

      // Parse ANNOUNCE_PLEASE: varint(body_len) + string(path_prefix)
      let pos = offset;
      let bodyLen;
      [bodyLen, pos] = decodeVarint(buf, pos);
      let pathPrefix = "";
      if (bodyLen > 0) {
        [pathPrefix, pos] = decodeString(buf, pos);
      }
      log(`ANNOUNCE_PLEASE from relay: prefix="${pathPrefix}" bodyLen=${bodyLen}`);

      // Respond with ANNOUNCE_INIT for our per-participant broadcast path.
      // Each participant uses a distinct suffix (e.g. "pipecat/client0")
      // so the relay can route SUBSCRIBEs to the right side.
      const broadcast = `${config.namespace}/${config.client_id}`;
      let suffix = broadcast;
      if (pathPrefix) {
        if (broadcast.startsWith(pathPrefix)) {
          suffix = broadcast.slice(pathPrefix.length).replace(/^\//, "");
        } else {
          suffix = null;
        }
      }

      const writer = stream.writable.getWriter();
      if (suffix !== null) {
        let body = encodeVarint(1); // 1 suffix
        body = concat(body, encodeString(suffix));
        const initMsg = concat(encodeVarint(body.length), body);
        await writer.write(initMsg);
        log(`Sent ANNOUNCE_INIT: 1 suffix="${suffix}" [${hexdump(initMsg)}]`);
      } else {
        const initMsg = concat(encodeVarint(1), encodeVarint(0));
        await writer.write(initMsg);
        log(`Sent ANNOUNCE_INIT: 0 suffixes (our broadcast doesn't match prefix "${pathPrefix}")`);
      }
      writer.releaseLock();

    } else {
      log(`Unknown bidi stream type ${streamType}, raw: [${hexdump(buf)}]`, "warn");
    }
  } catch (e) {
    log(`Incoming bidi stream error: ${e.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// Connect flow
// ---------------------------------------------------------------------------

async function doConnect() {
  const t0 = performance.now();
  clearTranscript();
  try {
    // 1. Fetch config from the FastAPI server
    log("Fetching /api/config...");
    const resp = await fetch("/api/config");
    config = await resp.json();
    log(`Config: relay=${config.relay_host}:${config.relay_port}, path="${config.path}", ns="${config.namespace}", client_id="${config.client_id}", bot_id="${config.bot_id}", publish="${config.publish_track}", subscribe="${config.subscribe_track}", insecure=${config.insecure}, cert_hash=${config.cert_hash ? config.cert_hash.slice(0, 12) + "..." : "none"}`);

    // 1b. Ask the server to start the bot. It blocks until the bot has
    //     finished its MOQ handshake with the relay, so our SUBSCRIBE is
    //     guaranteed to land at a publisher the relay already knows.
    log("Starting bot via /start...");
    const startResp = await fetch("/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ namespace: config.namespace }),
    });
    if (!startResp.ok) {
      throw new Error(`/start returned HTTP ${startResp.status}`);
    }
    const startInfo = await startResp.json();
    log(`Bot started: session=${startInfo.sessionId}, relay=${startInfo.relay}`);

    // 2. Build relay URL
    const scheme = "https";
    const host = config.relay_host;
    const relayUrl = `${scheme}://${host}:${config.relay_port}${config.path || "/moq"}`;
    log(`Connecting to ${relayUrl}`);
    setStatus("Connecting...", "connecting");

    // 3. Open WebTransport (connects via HTTP/3 over QUIC to the relay)
    const options = {};
    if (config.cert_hash) {
      const hashBytes = Uint8Array.from(atob(config.cert_hash), c => c.charCodeAt(0));
      options.serverCertificateHashes = [
        { algorithm: "sha-256", value: hashBytes.buffer },
      ];
      log(`Using certificate pinning: sha256=${config.cert_hash.slice(0, 16)}...`);
    } else {
      log("No certificate pinning (no cert_hash in config)", "warn");
    }
    log(`WebTransport options: ${JSON.stringify({...options, serverCertificateHashes: options.serverCertificateHashes ? "[present]" : undefined})}`);

    transport = new WebTransport(relayUrl, options);

    // Monitor connection closure — extract QUIC error details
    transport.closed.then(info => {
      const elapsed = (performance.now() - t0).toFixed(0);
      log(`Transport closed cleanly after ${elapsed}ms: closeCode=${info?.closeCode} reason="${info?.reason || ""}"`);
    }).catch(err => {
      const elapsed = (performance.now() - t0).toFixed(0);
      // WebTransportCloseInfo may have closeCode and reason
      const code = err?.closeCode ?? err?.code ?? "?";
      const reason = err?.reason ?? err?.message ?? String(err);
      log(`Transport closed with error after ${elapsed}ms: code=${code} reason="${reason}"`, "error");
      log(`  Error type: ${err?.constructor?.name}, keys: ${Object.keys(err || {}).join(",")}`, "error");
    });

    // Wait for the QUIC + HTTP/3 handshake
    const readyStart = performance.now();
    try {
      await transport.ready;
    } catch (readyErr) {
      const readyMs = (performance.now() - readyStart).toFixed(0);
      // Get the real error from transport.closed
      let closedErr = null;
      try { await transport.closed; } catch (e) { closedErr = e; }
      const detail = closedErr?.message || closedErr?.reason || String(closedErr || "");
      const code = closedErr?.closeCode ?? closedErr?.code ?? "?";
      log(`WebTransport handshake FAILED after ${readyMs}ms`, "error");
      log(`  ready error: ${readyErr.message}`, "error");
      log(`  closed error: code=${code} detail="${detail}"`, "error");
      log(`  URL: ${relayUrl}`, "error");
      log(`  Cert pinning: ${config.cert_hash ? "yes" : "no"}`, "error");
      log(`Troubleshooting:`, "error");
      log(`  - Is the relay running at ${host}:${config.relay_port}?`, "error");
      log(`  - Does relay config have [web.http] listen = "[::]:${config.relay_port}"?`, "error");
      log(`  - If using tls.generate in relay config, cert hash will change every restart`, "error");
      log(`  - Try 127.0.0.1 instead of localhost (QUIC/UDP has no IPv6->IPv4 fallback)`, "error");
      throw readyErr;
    }
    const readyMs = (performance.now() - readyStart).toFixed(0);
    log(`WebTransport connected (handshake took ${readyMs}ms)`);

    // 4. Open bidi stream for SETUP handshake
    log("Opening bidi stream for CLIENT_SETUP...");
    let setupBidi;
    try {
      setupBidi = await transport.createBidirectionalStream();
      log("Bidi stream created for SETUP");
    } catch (e) {
      // This is the "Connection lost" case — get more details
      let closedErr = null;
      try { await transport.closed; } catch (ce) { closedErr = ce; }
      const code = closedErr?.closeCode ?? closedErr?.code ?? "?";
      const reason = closedErr?.reason ?? closedErr?.message ?? "";
      log(`Failed to create bidi stream: ${e.message}`, "error");
      log(`  Underlying close: code=${code} reason="${reason}"`, "error");
      log(`  This usually means the relay rejected the WebTransport session.`, "error");
      log(`  Check the relay terminal for error logs (auth failure, path mismatch, etc.)`, "error");
      log(`  Relay URL path was: "${config.path || "/moq"}" — does the relay expect this path?`, "error");
      throw e;
    }
    const setupWriter = setupBidi.writable.getWriter();
    const setupStreamReader = setupBidi.readable.getReader();

    // Write stream type varint(0x20) + CLIENT_SETUP (u16 body size for WebTransport)
    const setupMsg = encodeClientSetup(Role.PUBSUB, [MOQL_VERSION], config.path || "/moq");
    const fullSetup = concat(encodeVarint(CLIENT_SETUP_TYPE), setupMsg);
    log(`Sending CLIENT_SETUP: role=PUBSUB version=0x${MOQL_VERSION.toString(16)} path="${config.path || "/moq"}" [${hexdump(fullSetup)}]`);
    try {
      await setupWriter.write(fullSetup);
      log("CLIENT_SETUP sent, waiting for SERVER_SETUP...");
    } catch (e) {
      log(`Failed to write CLIENT_SETUP: ${e.message}`, "error");
      throw e;
    }

    // 5. Read SERVER_SETUP response
    const setupTimeout = setTimeout(() => {
      log("SERVER_SETUP not received after 5s — relay may not understand our protocol", "warn");
    }, 5000);
    const { value: setupResp, done: setupDone } = await setupStreamReader.read();
    clearTimeout(setupTimeout);

    if (setupDone && !setupResp) {
      log("Setup stream closed by relay without sending SERVER_SETUP", "error");
      log("  The relay may not support moq-lite-02 (0xff0dad02)", "error");
      throw new Error("No SERVER_SETUP received");
    }

    if (setupResp) {
      const respData = new Uint8Array(setupResp);
      log(`Received SERVER_SETUP raw: [${hexdump(respData)}]`);
      // Skip stream type varint (0x21) if present
      let offset = 0;
      if (respData[0] === SERVER_SETUP_TYPE) {
        offset = 1;
      } else {
        const [st, newOff] = decodeVarint(respData, 0);
        if (st === SERVER_SETUP_TYPE) offset = newOff;
      }
      const { version } = parseServerSetup(respData.subarray(offset));
      const versionName = version === 0xff0dad02 ? "moq-lite-02" : `unknown (0x${version.toString(16)})`;
      log(`SERVER_SETUP: relay version=${versionName} (0x${version.toString(16)}), client version=moq-lite-02 (0x${MOQL_VERSION.toString(16)})`);
      if (version !== MOQL_VERSION) {
        log(`PROTOCOL MISMATCH! Browser speaks moq-lite-02 but relay speaks ${versionName}. Things will likely break.`, "error");
      }
    }
    // Keep setup stream open — don't close it
    setupWriter.releaseLock();

    connected = true;
    const totalMs = (performance.now() - t0).toFixed(0);
    setStatus("Connected", "connected");
    log(`Connected to relay (total setup took ${totalMs}ms)`);

    // 6. Start listening for incoming streams FIRST
    //    (relay may send ANNOUNCE_PLEASE and SUBSCRIBE before we subscribe)
    log("Listening for incoming bidi + uni streams...");
    handleIncomingBidiStreams();
    receiveUniStreams();

    // 7. Init audio playback (needs user gesture — we're inside a click handler)
    initAudioPlayback();
    log("Audio playback initialized (24kHz)");

    // 8. Start mic (so we're ready to publish when relay subscribes)
    await startMic();

    // 9. Subscribe to bot's audio + transcript tracks (non-fatal if bot
    //    isn't connected yet)
    await subscribeToAudio();
    subscribeToTranscript();

  } catch (e) {
    const elapsed = (performance.now() - t0).toFixed(0);
    log(`Connection failed after ${elapsed}ms: ${e.message}`, "error");
    if (e.stack) log(`  Stack: ${e.stack.split("\n").slice(1, 3).join(" | ")}`, "error");
    if (e.stack) log(`  Stack: ${e.stack}`);
    setStatus("Error: " + e.message, "error");
  }
}

async function subscribeToAudio(retryCount = 0) {
  const botBroadcast = `${config.namespace}/${config.bot_id}`;
  const botTrack = config.subscribe_track;
  const fullPath = `${botBroadcast}/${botTrack}`;
  const maxRetries = 10;
  const retryDelay = 2000; // 2 seconds between retries

  subAudioSubscribeId = nextSubscribeId++;

  try {
    // Open a new bidi stream for SUBSCRIBE
    log(`Subscribing to ${fullPath} (sub_id=${subAudioSubscribeId}, attempt ${retryCount + 1}/${maxRetries + 1})`);
    const subBidi = await transport.createBidirectionalStream();
    const subWriter = subBidi.writable.getWriter();
    const subReader = subBidi.readable.getReader();

    const subMsg = encodeSubscribe(subAudioSubscribeId, botBroadcast, botTrack);
    log(`Sending SUBSCRIBE: [${hexdump(subMsg)}]`);
    await subWriter.write(subMsg);

    // Read SUBSCRIBE_OK (or handle RESET_STREAM gracefully)
    log("Waiting for SUBSCRIBE_OK from relay...");
    const { value: okResp, done } = await subReader.read();
    if (okResp) {
      const okData = new Uint8Array(okResp);
      log(`Received SUBSCRIBE_OK for ${fullPath}: [${hexdump(okData)}]`);
      return; // Success
    } else if (done) {
      log(`SUBSCRIBE stream closed by relay without response (track "${fullPath}" may not exist — bot not connected yet?)`, "warn");
    }

    subWriter.releaseLock();
  } catch (e) {
    // RESET_STREAM means the track doesn't exist yet — not fatal
    log(`Subscribe to ${fullPath} failed: ${e.message}`, "warn");
    log(`  (This is normal if the bot hasn't connected to the relay yet)`, "warn");
  }

  // Retry if bot isn't connected yet
  if (connected && retryCount < maxRetries) {
    log(`Will retry subscribe in ${retryDelay / 1000}s (attempt ${retryCount + 1}/${maxRetries})...`);
    setTimeout(() => subscribeToAudio(retryCount + 1), retryDelay);
  } else if (retryCount >= maxRetries) {
    log(`Gave up subscribing to ${fullPath} after ${maxRetries} attempts`, "error");
  }
}

async function subscribeToTranscript() {
  const trackName = config.transcript_track || "transcript";
  const botBroadcast = `${config.namespace}/${config.bot_id}`;
  const fullPath = `${botBroadcast}/${trackName}`;
  subTranscriptSubscribeId = nextSubscribeId++;
  try {
    log(`Subscribing to ${fullPath} (sub_id=${subTranscriptSubscribeId})`);
    const subBidi = await transport.createBidirectionalStream();
    const subWriter = subBidi.writable.getWriter();
    const subReader = subBidi.readable.getReader();
    const subMsg = encodeSubscribe(subTranscriptSubscribeId, botBroadcast, trackName);
    await subWriter.write(subMsg);
    subWriter.releaseLock();
    // Keep the reader open so the relay can stream SUBSCRIBE_OK / errors.
    subReader.read().then(({ value, done }) => {
      if (value && value.byteLength) {
        log(`Transcript SUBSCRIBE_OK: [${hexdump(new Uint8Array(value))}]`);
      } else if (done) {
        log(`Transcript subscribe stream closed by relay (track may not exist)`, "warn");
      }
    }).catch(() => {});
  } catch (e) {
    log(`Subscribe to ${fullPath} failed: ${e.message}`, "warn");
  }
}

async function doDisconnect() {
  log(`Disconnecting... (sent=${audioSendCount} chunks, received=${audioRecvCount} chunks, errors=${audioSendErrors})`);
  connected = false;
  stopMic();

  if (transport) {
    try {
      transport.close();
      log("WebTransport closed");
    } catch (e) {
      log(`Error closing transport: ${e.message}`, "warn");
    }
    transport = null;
  }

  pubSubscribeId = null;
  pubGroupSeq = 0;
  subAudioSubscribeId = null;
  subTranscriptSubscribeId = null;
  nextSubscribeId = 1;
  audioSendCount = 0;
  audioSendErrors = 0;
  audioRecvCount = 0;
  incomingBidiCount = 0;

  if (audioContext) {
    audioContext.close();
    audioContext = null;
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
