/*
 * Copyright (c) 2024-2026, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/**
 * MOQ (Media over QUIC) protocol client — moq-lite-02.
 *
 * Implements varint codec, message encode/decode for moq-lite-02 protocol
 * so the browser can talk to a moq-lite relay over WebTransport.
 *
 * Key differences from draft-07:
 *  - Stream-per-request model (no shared control stream)
 *  - Setup uses u8(0x20/0x21) framing; via WebTransport uses u16 body size
 *  - SUBSCRIBE/ANNOUNCE each get their own bidi stream
 *  - Media data flows on uni streams as GROUP + FRAME
 *  - No QUIC datagrams
 */
(function () {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MOQL_VERSION = 0xff0dad02; // moq-lite-02

const Role = Object.freeze({
  PUBLISHER: 0x01,
  SUBSCRIBER: 0x02,
  PUBSUB: 0x03,
});

// Stream type varints (first thing written on a new bidi stream)
const StreamType = Object.freeze({
  SESSION: 0,
  ANNOUNCE: 1,
  SUBSCRIBE: 2,
});

// Setup message type bytes
const CLIENT_SETUP_TYPE = 0x20;
const SERVER_SETUP_TYPE = 0x21;

// Uni stream type
const UNI_STREAM_TYPE_GROUP = 0;

// ---------------------------------------------------------------------------
// QUIC variable-length integer codec
// ---------------------------------------------------------------------------

function encodeVarint(value) {
  if (value < 0x40) {
    return new Uint8Array([value]);
  } else if (value < 0x4000) {
    const buf = new Uint8Array(2);
    new DataView(buf.buffer).setUint16(0, value | 0x4000);
    return buf;
  } else if (value < 0x40000000) {
    const buf = new Uint8Array(4);
    new DataView(buf.buffer).setUint32(0, (value | 0x80000000) >>> 0);
    return buf;
  } else {
    const buf = new Uint8Array(8);
    const dv = new DataView(buf.buffer);
    const hi = Math.floor(value / 0x100000000) | 0xc0000000;
    const lo = value >>> 0;
    dv.setUint32(0, hi >>> 0);
    dv.setUint32(4, lo);
    return buf;
  }
}

function decodeVarint(data, offset) {
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const first = data[offset];
  const lengthBits = first >> 6;

  if (lengthBits === 0) {
    return [first, offset + 1];
  } else if (lengthBits === 1) {
    return [dv.getUint16(offset) & 0x3fff, offset + 2];
  } else if (lengthBits === 2) {
    return [dv.getUint32(offset) & 0x3fffffff, offset + 4];
  } else {
    const hi = dv.getUint32(offset) & 0x3fffffff;
    const lo = dv.getUint32(offset + 4);
    return [hi * 0x100000000 + lo, offset + 8];
  }
}

// ---------------------------------------------------------------------------
// String helpers
// ---------------------------------------------------------------------------

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function encodeString(str) {
  const bytes = encoder.encode(str);
  return concat(encodeVarint(bytes.length), bytes);
}

function decodeString(data, offset) {
  const [len, off] = decodeVarint(data, offset);
  const str = decoder.decode(data.subarray(off, off + len));
  return [str, off + len];
}

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------

function concat(...arrays) {
  let total = 0;
  for (const a of arrays) total += a.byteLength;
  const result = new Uint8Array(total);
  let pos = 0;
  for (const a of arrays) {
    result.set(a instanceof Uint8Array ? a : new Uint8Array(a), pos);
    pos += a.byteLength;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Setup messages (WebTransport uses u16 body size prefix)
// ---------------------------------------------------------------------------

/**
 * Encode CLIENT_SETUP for WebTransport (browser).
 *
 * WebTransport framing: varint(0x20) as stream type, then u16(body_len) + body
 * on a dedicated bidi stream.
 */
function encodeClientSetup(role, versions, path) {
  let body = encodeVarint(versions.length);
  for (const v of versions) {
    body = concat(body, encodeVarint(v));
  }

  // Parameters
  let numParams = 1; // role
  if (path) numParams += 1;
  body = concat(body, encodeVarint(numParams));

  // Role param (key=0)
  body = concat(body, encodeVarint(0), encodeVarint(1), encodeVarint(role));

  // Path param (key=1)
  if (path) {
    const pathBytes = encoder.encode(path);
    body = concat(body, encodeVarint(1), encodeVarint(pathBytes.length), pathBytes);
  }

  // WebTransport framing: u16(body_len) + body
  const frame = new Uint8Array(2 + body.length);
  new DataView(frame.buffer).setUint16(0, body.length);
  frame.set(body, 2);

  return frame;
}

/**
 * Parse SERVER_SETUP from WebTransport.
 *
 * WebTransport framing: u16(body_len) + body
 * Body: varint(selected_version) + params...
 */
function parseServerSetup(data) {
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const bodyLen = dv.getUint16(0);
  let offset = 2;
  const [version, newOff] = decodeVarint(data, offset);
  return { version, bodyEnd: 2 + bodyLen };
}

// ---------------------------------------------------------------------------
// Subscribe messages (on dedicated bidi stream with stream_type=2)
// ---------------------------------------------------------------------------

/**
 * Encode SUBSCRIBE message body.
 *
 * Stream format: varint(2) + varint(body_len) + body
 * Body: varint(sub_id) + string(broadcast_path) + string(track_name) + u8(priority)
 */
function encodeSubscribe(subscribeId, broadcastPath, trackName, priority = 128) {
  let body = concat(
    encodeVarint(subscribeId),
    encodeString(broadcastPath),
    encodeString(trackName),
    new Uint8Array([priority]),
  );

  // Stream type + length-prefixed body
  return concat(
    encodeVarint(StreamType.SUBSCRIBE),
    encodeVarint(body.length),
    body,
  );
}

/**
 * Encode SUBSCRIBE_OK response.
 * Format: varint(0) — empty body.
 */
function encodeSubscribeOk() {
  return encodeVarint(0);
}

/**
 * Decode a SUBSCRIBE message from an incoming bidi stream.
 * Input starts after the stream type varint has been consumed.
 */
function decodeSubscribe(data, offset = 0) {
  let bodyLen;
  [bodyLen, offset] = decodeVarint(data, offset);
  const bodyEnd = offset + bodyLen;

  let subscribeId, broadcastPath, trackName;
  [subscribeId, offset] = decodeVarint(data, offset);
  [broadcastPath, offset] = decodeString(data, offset);
  [trackName, offset] = decodeString(data, offset);
  const priority = data[offset];
  offset += 1;

  return { subscribeId, broadcastPath, trackName, priority, end: bodyEnd };
}

// ---------------------------------------------------------------------------
// GROUP + FRAME messages (on unidirectional streams)
// ---------------------------------------------------------------------------

/**
 * Encode GROUP header + single FRAME for a uni stream.
 *
 * Format: u8(0) + varint(header_body_len) + varint(subscribe_id) + varint(group_seq)
 *         + varint(payload_len) + payload
 */
function encodeGroupAndFrame(subscribeId, groupSeq, payload) {
  const headerBody = concat(
    encodeVarint(subscribeId),
    encodeVarint(groupSeq),
  );

  const frame = concat(
    encodeVarint(payload.byteLength),
    payload instanceof Uint8Array ? payload : new Uint8Array(payload),
  );

  return concat(
    new Uint8Array([UNI_STREAM_TYPE_GROUP]),
    encodeVarint(headerBody.length),
    headerBody,
    frame,
  );
}

/**
 * Parse GROUP header + FRAMEs from a complete uni stream buffer.
 *
 * Returns { subscribeId, groupSeq, frames: [Uint8Array, ...] }
 */
function parseGroupStream(data) {
  let offset = 0;

  // u8(0) stream type
  const streamType = data[offset];
  offset += 1;

  // varint(body_len)
  let bodyLen;
  [bodyLen, offset] = decodeVarint(data, offset);
  const bodyStart = offset;

  // varint(subscribe_id)
  let subscribeId;
  [subscribeId, offset] = decodeVarint(data, offset);

  // varint(group_seq)
  let groupSeq;
  [groupSeq, offset] = decodeVarint(data, offset);

  // Advance past header body
  offset = bodyStart + bodyLen;

  // Parse frames: varint(payload_len) + payload, repeated
  const frames = [];
  while (offset < data.length) {
    let payloadLen;
    [payloadLen, offset] = decodeVarint(data, offset);
    frames.push(data.subarray(offset, offset + payloadLen));
    offset += payloadLen;
  }

  return { subscribeId, groupSeq, frames };
}

// ---------------------------------------------------------------------------
// Exports (global for vanilla JS)
// ---------------------------------------------------------------------------

window.MOQ = {
  MOQL_VERSION,
  Role,
  StreamType,
  CLIENT_SETUP_TYPE,
  SERVER_SETUP_TYPE,
  encodeVarint,
  decodeVarint,
  encodeString,
  decodeString,
  concat,
  encodeClientSetup,
  parseServerSetup,
  encodeSubscribe,
  encodeSubscribeOk,
  decodeSubscribe,
  encodeGroupAndFrame,
  parseGroupStream,
};

})();
