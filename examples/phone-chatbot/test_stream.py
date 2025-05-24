#!/usr/bin/env python3
import asyncio
import json
import base64
import wave
import websockets

# adjust if your server is on a different host/port
WS_URI = "ws://localhost:8000/ws/twilio"

# arbitrary IDs for testing
STREAM_SID = "testStreamSid"
CALL_SID   = "testCallSid"

async def run_test():
    # open the incoming and outgoing WAVs
    in_wav  = wave.open("user.wav", "rb")
    out_wav = wave.open("agent.wav", "wb")
    out_wav.setnchannels(1)
    out_wav.setsampwidth(1)    # μ-law is 8 bits
    out_wav.setframerate(8000) # Twilio expects 8 kHz μ-law

    async with websockets.connect(WS_URI) as ws:
        # 1) handshake: tell the server “connected” then “start”
        await ws.send(json.dumps({"event":"connected"}))
        await ws.send(json.dumps({
            "event":"start",
            "start": {"streamSid": STREAM_SID, "callSid": CALL_SID}
        }))

        async def send_audio():
            """Read 20 ms chunks (160 samples → 320 bytes) and send as base64."""
            while True:
                pcm = in_wav.readframes(160)
                if not pcm:
                    break
                b64 = base64.b64encode(pcm).decode("ascii")
                msg = {
                    "event": "media",
                    "media": {"payload": b64},
                    "streamSid": STREAM_SID
                }
                await ws.send(json.dumps(msg))
                await asyncio.sleep(0.02)  # 20 ms per frame

        async def recv_audio():
            """Listen for the bot’s μ-law chunks and write them to agent.wav"""
            while True:
                raw = await ws.recv()
                data = json.loads(raw)
                # TwilioFrameSerializer will send back “media” events for TTS
                if data.get("event") == "media" and data.get("media", {}).get("payload"):
                    chunk = base64.b64decode(data["media"]["payload"])
                    out_wav.writeframes(chunk)

        # run sender & receiver in parallel
        await asyncio.gather(send_audio(), recv_audio())

    in_wav.close()
    out_wav.close()

if __name__ == "__main__":
    asyncio.run(run_test())
