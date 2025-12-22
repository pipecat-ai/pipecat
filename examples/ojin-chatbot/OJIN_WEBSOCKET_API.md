# Ojin Oris 1.0 WebSocket API

Real-time talking head synthesis API. Send audio, receive synchronized video frames.

## Setup

### 1. Get Credentials

1. Go to https://ojin.ai/
2. Create an account
3. Navigate to **API Keys** tab → create an API key
4. Navigate to **Oris 1.0** tab → create a persona configuration
5. Copy your API key and config ID

### 2. Connect

```
wss://models.ojin.ai/realtime?config_id={YOUR_CONFIG_ID}
```

**Headers:**
```
Authorization: {YOUR_API_KEY}
```

**Example (Python):**
```python
import websockets

headers = {"Authorization": "your-api-key"}
url = "wss://models.ojin.ai/realtime?config_id=your-config-id"

async with websockets.connect(url, extra_headers=headers) as ws:
    # Connected
    pass
```

## Message Flow

```
Client                          Server
  |                               |
  |-------- Connect ------------->|
  |                               |
  |<----- SessionReady -----------|
  |                               |
  |-- InteractionInput (audio) -->|
  |-- InteractionInput (audio) -->|
  |                               |
  |<-- InteractionResponse -------|
  |<-- InteractionResponse -------|
  |<-- InteractionResponse -------|
  |                               |
  |--- EndInteraction ----------->|
  |                               |
  |<-- InteractionResponse -------|  (is_final_response: true)
```

## Messages

### 1. SessionReady (Server → Client)

Sent immediately after connection. Indicates the server is ready.

**Format:** JSON string

```json
{
  "type": "sessionReady",
  "payload": {
    "trace_id": "uuid-string",
    "status": "success",
    "load": 0.5,
    "timestamp": 1234567890000,
    "parameters": {
      "speech_filter_amount": 1000.0,
      "idle_filter_amount": 1000.0,
      "idle_mouth_opening_scale": 0.0,
      "speech_mouth_opening_scale": 1.0
    }
  }
}
```

**Parameters:**
- `speech_filter_amount`: Smoothing applied to speech frames (higher = smoother transitions)
- `idle_filter_amount`: Smoothing applied to idle frames
- `idle_mouth_opening_scale`: Scale factor for mouth movement during idle (0.0 = closed)
- `speech_mouth_opening_scale`: Scale factor for mouth movement during speech (1.0 = full movement)

### 2. InteractionInput (Client → Server)

Send audio data. The server generates video frames synced to your audio.

**Format:** Binary

**Structure:**
```
[1 byte]    Payload type (1 = audio)
[8 bytes]   Timestamp (uint64, milliseconds since Unix epoch)
[4 bytes]   Params size (uint32)
[N bytes]   Params JSON (if params_size > 0)
[M bytes]   Audio payload
```

**Audio Requirements:**
- Format: PCM int16 (signed 16-bit integers)
- Sample rate: 16kHz
- Channels: Mono
- Max size per message: the whole message should be under 512KB

**Optional Parameters:**

Send as JSON in the params field:

```json
{
  "speech_filter_amount": 1000.0,
  "idle_filter_amount": 1000.0,
  "idle_mouth_opening_scale": 0.0,
  "speech_mouth_opening_scale": 1.0
}
```

These override the session defaults for this specific audio chunk.

**Example (Python):**
```python
import struct
import json
import numpy as np

# Generate or load audio
audio_samples = np.array([...], dtype=np.int16)  # 16kHz mono
audio_bytes = audio_samples.tobytes()

# Optional parameters
params = {
    "speech_filter_amount": 800.0,
    "speech_mouth_opening_scale": 0.8
}
params_bytes = json.dumps(params).encode('utf-8')

# Pack header
header = struct.pack(
    '!BQI',           # Format: byte, uint64, uint32
    1,                 # Payload type (1 = audio)
    int(time.time() * 1000),  # Timestamp
    len(params_bytes)  # Params size
)

# Combine and send
message = header + params_bytes + audio_bytes
await ws.send(message)
```

**NOTE**:
the user can also send `client_frame_index` in the parameters, which is the index of the frame the client currently displaying plus some extra overhead for latency, this is used to make sure the user get smooth transitions between idle and speech animation in case they are caching idle frames.

**Streaming Pattern:**

For smooth playback, send audio at regular intervals. The server outputs 25 fps video, batched in groups of 10 frames.

Recommended: Send audio chunks equivalent to multiples of 400ms (0.4s × 16000 = 6400 samples = 12800 bytes).

```python
sample_rate = 16000
chunk_duration = 0.4  # seconds
chunk_samples = int(sample_rate * chunk_duration)
max_bytes = 500 * 1024  # 500KB

for i in range(0, len(audio_samples), chunk_samples):
    chunk = audio_samples[i:i + chunk_samples]
    
    # Send chunk
    await send_interaction_input(ws, chunk)
    
    # Optional: throttle sends
    await asyncio.sleep(0.01)
```

### 3. InteractionResponse (Server → Client)

Video frame with synchronized audio. The server sends these continuously as it processes your audio.

**Format:** Binary

**Structure:**
```
[1 byte]     Is final flag (1 = last frame in interaction, 0 = more frames coming)
[16 bytes]   Interaction ID (UUID)
[8 bytes]    Timestamp (uint64, milliseconds)
[4 bytes]    Usage (uint32)
[4 bytes]    Frame index (uint32)
[4 bytes]    Number of payload entries (uint32)

For each payload entry:
  [4 bytes]  Payload size (uint32)
  [1 byte]   Payload type (1 = audio, 2 = image)
  [N bytes]  Payload data
```

**Interaction ID:**
- `00000000-0000-0000-0000-000000000000`: Idle frames (persona in rest state)
- Any other UUID: Speech frames (generated from your audio)

**Payload Types:**
- Type 1 (audio): PCM int16, 16kHz mono, typically 640 bytes (40ms at 25fps)
- Type 2 (image): RGB raw bytes, resolution depends on your config (e.g., 1280x720), encoded as JPEG

**Example (Python):**
```python
import struct
import uuid

async def receive_response(ws):
    data = await ws.recv()
    
    # Parse header
    header_fmt = '!B16sQIII'
    header_size = struct.calcsize(header_fmt)
    
    (is_final, interaction_id_bytes, timestamp, 
     usage, index, num_payloads) = struct.unpack(header_fmt, data[:header_size])
    
    interaction_id = str(uuid.UUID(bytes=interaction_id_bytes))
    is_idle = interaction_id == "00000000-0000-0000-0000-000000000000"
    
    # Parse payloads
    offset = header_size
    video_frame = None
    audio_chunk = None
    
    for _ in range(num_payloads):
        size, payload_type = struct.unpack('!IB', data[offset:offset+5])
        offset += 5
        
        payload_data = data[offset:offset+size]
        offset += size
        
        if payload_type == 2:  # Image
            video_frame = payload_data
        elif payload_type == 1:  # Audio
            audio_chunk = payload_data
    
    return {
        'is_final': bool(is_final),
        'interaction_id': interaction_id,
        'is_idle': is_idle,
        'index': index,
        'video_frame': video_frame,
        'audio_chunk': audio_chunk
    }
```

**Playback:**

The server operates at 25 fps. To play frames smoothly:

1. Receive frames and queue them
2. Play at 25 fps (one frame every 40ms)
3. If you receive idle frames (interaction_id = nil UUID), cache them and loop during silence
4. When you receive speech frames, play those instead
5. When `is_final_response = true`, you've received all frames for that interaction

### 4. EndInteraction (Client → Server)

Signal that you've finished sending audio for this interaction. The server will complete processing and send remaining frames.

**Format:** JSON string

```json
{
  "type": "endInteraction",
  "payload": {
    "timestamp": 1234567890000
  }
}
```

After sending this, you'll receive all remaining video frames, with the last one having `is_final_response: true`.

**Example (Python):**
```python
import json
import time

end_message = {
    "type": "endInteraction",
    "payload": {
        "timestamp": int(time.time() * 1000)
    }
}

await ws.send(json.dumps(end_message))
```

### 5. CancelInteraction (Client → Server)

Immediately terminate the current interaction. The server stops processing and drops remaining frames.

**Format:** JSON string

```json
{
  "type": "cancelInteraction",
  "payload": {
    "timestamp": 1234567890000
  }
}
```

Use this for interruptions (e.g., user starts speaking while persona is talking).

**Difference from EndInteraction:**
- `EndInteraction`: Graceful finish, receive all remaining frames
- `CancelInteraction`: Immediate stop, remaining frames discarded

### 6. ErrorResponse (Server → Client)

Sent when something goes wrong.

**Format:** JSON string

```json
{
  "type": "errorResponse",
  "payload": {
    "interaction_id": "uuid-or-null",
    "code": "ERROR_CODE",
    "message": "Human readable error",
    "details": {},
    "timestamp": 1234567890000
  }
}
```

**Common Error Codes:**
- `AUTH_FAILED`: Invalid API key
- `INVALID_PERSONA_ID_CONFIGURATION`: Config ID not found or invalid
- `FAILED_CREATE_MODEL`: Server couldn't load the persona model
- `FRAME_SIZE_EXCEEDED`: Message exceeded 512KB limit
- `INVALID_INTERACTION_ID`: Interaction ID mismatch or invalid
- `NO_BACKEND_SERVER_AVAILABLE`: Service temporarily unavailable

## Rate Limits

- **6 requests per second** (rps)
- **512KB max message size**

Exceeding these limits results in error responses or buffering.

## Complete Example

```python
import asyncio
import json
import struct
import time
import numpy as np
import websockets

API_KEY = "your-api-key"
CONFIG_ID = "your-config-id"
URL = f"wss://models.ojin.ai/realtime?config_id={CONFIG_ID}"

async def send_audio_chunk(ws, audio_int16_bytes, params=None):
    """Send audio chunk to server."""
    params_bytes = b""
    if params:
        params_bytes = json.dumps(params).encode('utf-8')
    
    header = struct.pack(
        '!BQI',
        1,  # Audio payload type
        int(time.time() * 1000),
        len(params_bytes)
    )
    
    await ws.send(header + params_bytes + audio_int16_bytes)

async def receive_frame(ws):
    """Receive and parse video frame."""
    data = await ws.recv()
    
    if isinstance(data, str):
        # JSON message (SessionReady, Error, etc)
        return json.loads(data)
    
    # Binary frame
    header_fmt = '!B16sQIII'
    header_size = struct.calcsize(header_fmt)
    
    (is_final, interaction_id_bytes, timestamp, 
     usage, index, num_payloads) = struct.unpack(header_fmt, data[:header_size])
    
    offset = header_size
    video_frame = None
    audio_chunk = None
    
    for _ in range(num_payloads):
        size, payload_type = struct.unpack('!IB', data[offset:offset+5])
        offset += 5
        payload_data = data[offset:offset+size]
        offset += size
        
        if payload_type == 2:
            video_frame = payload_data
        elif payload_type == 1:
            audio_chunk = payload_data
    
    return {
        'video_frame': video_frame,
        'audio_chunk': audio_chunk,
        'is_final': bool(is_final),
        'index': index
    }

async def main():
    headers = {"Authorization": API_KEY}
    
    async with websockets.connect(URL, extra_headers=headers) as ws:
        # Wait for SessionReady
        session_ready = await receive_frame(ws)
        print(f"Session ready: {session_ready}")
        
        # Generate sample audio (1 second, 440Hz sine wave)
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = (32767 * 0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
        
        # Send audio in chunks
        chunk_size = 6400  # 400ms chunks
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            await send_audio_chunk(ws, chunk.tobytes())
            await asyncio.sleep(0.01)  # Throttle sends
        
        # End interaction
        end_message = {
            "type": "endInteraction",
            "payload": {"timestamp": int(time.time() * 1000)}
        }
        await ws.send(json.dumps(end_message))
        
        # Receive frames until final
        while True:
            frame = await receive_frame(ws)
            if isinstance(frame, dict) and frame.get('is_final'):
                print("Received final frame")
                break
            print(f"Received frame {frame.get('index', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices

1. **Audio Chunking**: Send audio in 400ms chunks (6400 samples at 16kHz) for smooth frame generation
2. **Buffer Management**: Queue received frames and play at exactly 25 fps to avoid jitter
3. **Idle Frames**: Cache idle frames when you first connect and loop them during silence
4. **Error Handling**: Always handle `ErrorResponse` messages and implement reconnection logic
5. **Rate Limiting**: Don't exceed 6 rps - batch audio chunks if needed
6. **Interruptions**: Use `CancelInteraction` for immediate stops, `EndInteraction` for graceful endings
7. **Frame Synchronization**: The server pre-bundles audio with video frames - no client-side sync needed

## Troubleshooting

**Connection fails:**
- Verify API key and config ID
- Check that config exists in dashboard
- Ensure network allows WebSocket connections

**No frames received:**
- Confirm you received `SessionReady` before sending audio
- Verify audio format (16kHz, int16, mono)
- Check message size < 512KB

**Choppy playback:**
- Ensure you're playing at exactly 25 fps
- Buffer at least 10 frames before starting playback
- Check for network latency issues

**Frame lag:**
- Reduce `speech_filter_amount` parameter
- Send smaller audio chunks more frequently
- Check server load in `SessionReady` message
