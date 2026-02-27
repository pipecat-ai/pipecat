# SIP/RTP Transport Design

## Summary

Add a SIP UAS transport to Pipecat that allows PBX/SBC SIP calls to enter a pipeline directly. Pure Python (numpy for codecs), G.711 PCMU/PCMA, 20ms RTP packetization, RFC 2833 DTMF detection.

## Architecture

```
┌─────────────────────────────────────┐
│        SIPServerTransport           │
│  (UDP SIP listener, port manager)   │
│                                     │
│  on INVITE → create SIPCallTransport│
│  on BYE    → teardown call          │
└──────────────┬──────────────────────┘
               │ emits per call
    ┌──────────▼──────────────┐
    │    SIPCallTransport     │
    │   (extends BaseTransport)│
    │                         │
    │  .input()  → SIPInputTransport  (BaseInputTransport)
    │  .output() → SIPOutputTransport (BaseOutputTransport)
    │  .session  → SIPSession (call metadata)
    └──────────┬──────────────┘
               │ owns
    ┌──────────▼──────────────┐
    │      RTPSession         │
    │  (UDP send/recv, 20ms)  │
    │  rx_queue → input       │
    │  tx_queue ← output      │
    └─────────────────────────┘
```

**User API:**

```python
server = SIPServerTransport(params=SIPParams(...))

@server.event_handler("on_call_started")
async def on_call(server, call_transport: SIPCallTransport):
    pipeline = Pipeline([
        call_transport.input(), stt, llm, tts, call_transport.output()
    ])
    task = PipelineTask(pipeline)
    await runner.run(task)

await server.start()
```

## Data Flow

**Inbound:** RTP UDP → G.711 decode → resample 8k→16k → InputAudioRawFrame(16kHz) → pipeline

**Outbound:** pipeline → OutputAudioRawFrame → resample to 8k → G.711 encode → RTP UDP

## File Layout

```
src/pipecat/transports/sip/
├── __init__.py      # Exports SIPServerTransport, SIPCallTransport, SIPParams
├── params.py        # SIPParams (extends TransportParams)
├── transport.py     # SIPServerTransport, SIPCallTransport, SIPInput/OutputTransport
├── signaling.py     # SIP message parsing, response building, UDP protocol
├── sdp.py           # SDP parse/generate
├── rtp.py           # RTPSession (send/recv loops, pre-buffering)
├── codecs.py        # G.711 PCMU/PCMA codec (numpy LUT) + RFC 2833 DTMF
```

## Components

### SIPParams

Extends `TransportParams`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| sip_listen_host | str | "0.0.0.0" | SIP listener bind address |
| sip_listen_port | int | 5060 | SIP listener port |
| rtp_port_range | tuple[int,int] | (10000, 20000) | RTP port allocation range |
| codec_preferences | list[str] | ["PCMU","PCMA"] | Codec priority order |
| ptime_ms | int | 20 | Packetization time |
| rtp_prebuffer_frames | int | 3 | Pre-buffer frames before TX playback (~60ms) |
| rtp_dead_timeout_ms | int | 5000 | No RTP received → teardown |
| ack_timeout_ms | int | 3000 | ACK wait timeout → teardown |
| max_calls | int | 100 | Max concurrent calls |
| dtmf_enabled | bool | True | RFC 2833 DTMF detection |

Inherited defaults set: `audio_in_enabled=True`, `audio_out_enabled=True`, `audio_in_sample_rate=16000`, `audio_out_sample_rate=16000`.

### SIPSession

Per-call state dataclass:

- `call_id`, `local_tag`, `remote_tag`
- `from_header`, `to_header`, `via_header`, `cseq`
- `remote_rtp_addr: tuple[str, int]`
- `local_rtp_port: int`, `local_ip: str`
- `codec: str` (negotiated, "PCMU" or "PCMA")

### SIP Signaling

UDP `asyncio.DatagramProtocol`. Handles:

**INVITE flow:**
1. Receive INVITE → parse SIPMessage
2. Send 100 Trying
3. Parse SDP → extract remote RTP IP:port and codec
4. Allocate local RTP port from pool
5. Build SDP answer, send 200 OK
6. Start ACK timeout timer
7. ACK received → cancel timer, create SIPCallTransport, start RTP, fire on_call_started
8. Timeout → send BYE, release port

**BYE (inbound):** Respond 200 OK, stop RTP, release port, fire on_call_ended.

**BYE (outbound, UAS-initiated):** Build BYE (swap From/To per SIP rules), send, stop RTP.

Required headers: Via, From/To (tagged), Call-ID, CSeq, Contact, Content-Length.

### RTP Media

**RTPSession:** Per-call UDP send/recv with two async tasks.

- 20ms frames, 160 samples @ 8kHz, timestamp += 160, sequence += 1
- Monotonic clock drift-corrected pacing
- Random SSRC per session
- RTP header: 12 bytes (V=2, no padding/ext/CSRC)

**Pre-buffering (TX):** Accumulate 3 frames (~60ms) before starting playback. If queue drains, wait 40ms then revert to silence/idle. Absorbs TTS delivery jitter.

**G.711 Codec (numpy LUT):**
- 65536-entry encode LUT (int16 → uint8 μ-law/A-law)
- 256-entry decode LUT (uint8 → int16)
- Singleton pattern for LUT reuse

**Resampling (numpy):**
- Up: `np.interp` with pre-computed indices for 2x (8k→16k)
- Down: moving-average anti-alias filter + decimation

**RFC 2833 DTMF:**
- Detect payload type 101 in RTP packets
- Parse 4-byte event: event ID (digit), end bit, duration
- On end bit → emit InputTransportMessageFrame({"type":"dtmf","digit":"X"})
- Debounce by event ID + timestamp

### Transport Classes

**SIPInputTransport (BaseInputTransport):**
- `_rx_loop` task: pull RTP rx_queue → resample 8k→16k → push_audio_frame()
- `_dtmf_loop` task: pull dtmf_queue → push InputTransportMessageFrame
- BaseInputTransport handles VAD, filtering, turn detection

**SIPOutputTransport (BaseOutputTransport):**
- Override `write_audio_frame()`: resample to 8k → G.711 encode → buffer to 320-byte frames → push to tx_queue
- BaseOutputTransport handles buffering, chunking, interruptions via MediaSender

**SIPCallTransport (BaseTransport):**
- Lazy-creates input/output transports
- Exposes `session: SIPSession`
- Provides `hangup()` for UAS-initiated BYE

**SIPServerTransport (BaseObject):**
- `start()`: create UDP SIP listener
- `stop()`: close listener, teardown all calls
- Events: on_call_started, on_call_ended, on_call_failed
- Manages port pool, active calls dict, max_calls enforcement

### Teardown Guarantees

- RTP port always released (try/finally)
- RTP session tasks always cancelled
- SIP BYE sent on pipeline end or error

## Testing

1. **test_sip_sdp.py** — SDP parse/generate, single/multiple codecs, missing fields
2. **test_sip_codecs.py** — PCMU/PCMA encode/decode roundtrip, known vectors, DTMF parsing
3. **test_sip_rtp.py** — RTP header pack/unpack, sequence/timestamp wraparound
4. **test_sip_signaling.py** — SIP message parsing, response building, ACK timeout
5. **test_sip_resample.py** — Upsample/downsample, roundtrip signal integrity

## Example

`examples/foundational/XX-sip-echo-bot.py`: Accept SIP call, echo audio or play TTS greeting. Document testing with pjsua/linphone.

## Out of Scope (Future PRs)

- SRTP / TLS
- SIP CANCEL, REGISTER, REFER
- Opus codec
- Full jitter buffer with reordering
- Echo cancellation (AEC)
- SIP authentication (digest)

## Dependencies

- numpy (G.711 codec LUTs, resampling)
- No other new dependencies (asyncio UDP, struct for RTP headers)

## Reference

Implementation follows patterns from `/Users/rasonyang/workspaces/mlx/pipecat-bot` (proven SIP/RTP implementation using same architecture).
