"""Inbound RTVI over LiveKitTransport never reaches RTVIProcessor.

  A. does RTVIProcessor fire on_client_ready?          expected False
  B. does the frame instead reach the OUTPUT transport
     and get re-sent to the client?                    expected True
  C. is that re-send addressed by SID or identity?     control: SID is undeliverable

Run:  LK_URL=wss://... LK_KEY=... LK_SECRET=... python repro.py
"""

import asyncio
import json
import os
import sys

from livekit import api, rtc
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport
from pipecat.workers.runner import WorkerRunner

URL, KEY, SECRET = os.environ["LK_URL"], os.environ["LK_KEY"], os.environ["LK_SECRET"]
ROOM = os.environ.get("LK_ROOM", "repro-rtvi-inbound")

logger.remove()
logger.add(sys.stderr, level="WARNING")

# The simplest well-formed RTVI message. Any type would do; client-ready is the one
# whose absence is most visible, since a session cannot become ready without it.
CLIENT_READY = {"id": "repro-1", "label": "rtvi-ai", "type": "client-ready", "data": {}}


def token(identity: str) -> str:
    return (
        api.AccessToken(KEY, SECRET)
        .with_identity(identity)
        .with_grants(
            api.VideoGrants(room_join=True, room=ROOM, can_publish=True, can_subscribe=True)
        )
        .to_jwt()
    )


async def main() -> int:
    rtvi_handled = asyncio.Event()  # check A
    received: list[dict] = []  # everything the client gets back
    resent: list[tuple[str, str]] = []  # check B: (participant_id, message) at output.send_message

    # Bot side: a normal pipeline. No STT/LLM/TTS — the bug is in message routing,
    # so nothing between input and output is needed to show it.
    transport = LiveKitTransport(
        url=URL,
        token=token("repro-bot"),
        room_name=ROOM,
        params=LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    rtvi = RTVIProcessor()

    @rtvi.event_handler("on_client_ready")
    async def _on_client_ready(_p):
        rtvi_handled.set()

    # Wrap the output transport's send_message. Without this, a message that is misrouted
    # and then dropped is indistinguishable from one that was never received at all —
    # which is what makes this bug hard to see from the outside.
    out = transport.output()
    _orig_send = out.send_message

    async def _traced_send(frame):
        resent.append((getattr(frame, "participant_id", None), frame.message))
        return await _orig_send(frame)

    out.send_message = _traced_send

    worker = PipelineWorker(
        Pipeline([transport.input(), rtvi, out]), observers=[RTVIObserver(rtvi)]
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    runner_task = asyncio.create_task(runner.run())
    await asyncio.sleep(6)  # let the bot finish joining before the client arrives

    # Client side: a bare LiveKit room, standing in for a client-side transport
    # (pipecat does not publish one for LiveKit).
    client = rtc.Room()

    @client.on("data_received")
    def _on_data(packet: rtc.DataPacket):
        try:
            received.append(json.loads(bytes(packet.data).decode()))
        except Exception:
            received.append({"undecodable": len(packet.data)})

    await client.connect(URL, token("repro-client"))
    me = client.local_participant
    await asyncio.sleep(2)

    # The one inbound message under test.
    await client.local_participant.publish_data(json.dumps(CLIENT_READY).encode(), reliable=True)
    await asyncio.sleep(6)

    # Controls for check C. Same directed send, same recipient, two different keys —
    # isolating the addressing defect from everything else above.
    await transport.send_message(json.dumps({"probe": "by-sid"}), me.sid)
    await asyncio.sleep(3)
    by_sid = any(m.get("probe") == "by-sid" for m in received)

    await transport.send_message(json.dumps({"probe": "by-identity"}), me.identity)
    await asyncio.sleep(3)
    by_identity = any(m.get("probe") == "by-identity" for m in received)

    # Did the output transport try to send the client its own message back?
    echoed = [(pid, msg) for pid, msg in resent if json.loads(msg) == CLIENT_READY]

    print(f"A. RTVIProcessor fired on_client_ready    : {rtvi_handled.is_set()}   (expected False)")
    print(f"B. client's own message re-sent by OUTPUT : {bool(echoed)}   (expected True)")
    for pid, _ in echoed:
        print(f"     addressed to {pid!r}; sid={me.sid!r} identity={me.identity!r}")
    print(f"C. directed send delivered by SID         : {by_sid}   (expected False)")
    print(f"   directed send delivered by IDENTITY    : {by_identity}   (expected True)")

    ok = (not rtvi_handled.is_set()) and bool(echoed) and (not by_sid) and by_identity

    await client.disconnect()
    await runner.cancel()
    try:
        await asyncio.wait_for(runner_task, timeout=10)
    except (TimeoutError, asyncio.CancelledError):
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
