#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A SIP bot that registers to a SIP server and greets inbound callers.

The bot is itself a SIP user agent (no WebRTC or WebSocket gateway in
between): it registers to a SIP server, answers inbound calls, speaks a
greeting through local TTS, and hangs up. Any registrar that routes calls
to registered endpoints works — the setup below uses FreeSWITCH, a common
open-source PBX.

Requirements::

    uv add "pipecat-ai[sip,kokoro]"

Running against FreeSWITCH:

1. Create a directory user for the bot (replace the password)::

       <!-- conf/directory/default/5001.xml -->
       <include>
         <user id="5001">
           <params>
             <param name="password" value="bot-password"/>
           </params>
           <variables>
             <variable name="user_context" value="default"/>
           </variables>
         </user>
       </include>

2. Route calls to the bot in the dialplan (this example bridges any
   inbound PSTN or internal call whose destination is 5001)::

       <extension name="ai_bot">
         <condition field="destination_number" expression="^5001$">
           <action application="bridge" data="user/5001@$${domain}"/>
         </condition>
       </extension>

3. Reload and confirm the registration once the bot is up::

       fs_cli -x "reloadxml"
       fs_cli -x "sofia status profile internal reg"

4. Call 5001 from a registered phone. The bot answers, speaks its
   greeting, and hangs up.

NAT notes: when the bot runs behind NAT, keep the registration alive so
the server's inbound INVITEs follow a path the NAT still forwards —
``reg_interval`` below is deliberately short for that reason. When
FreeSWITCH itself runs behind NAT (e.g. in a container), set its
``ext-sip-ip`` / ``ext-rtp-ip`` to the address endpoints can reach, so its
SDP advertises a routable media address.

For a bot that converses (STT + LLM + TTS) instead of only greeting, see
``getting-started/06-voice-agent.py`` — the SIP transport slots into the
same pipeline in place of the WebRTC transport.
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.transports.sip.connection import SIPConnection
from pipecat.transports.sip.transport import SIPParams, SIPTransport
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# The bot's SIP account: AOR is sip:<user>@<domain>.
SIP_USER = os.getenv("SIP_USER", "5001")
SIP_DOMAIN = os.getenv("SIP_DOMAIN", "192.168.1.10")  # FreeSWITCH address
SIP_PASSWORD = os.getenv("SIP_PASSWORD", "bot-password")
SIP_TRANSPORT = os.getenv("SIP_TRANSPORT", "udp")  # udp / tcp / tls

connection = SIPConnection(
    user=SIP_USER,
    domain=SIP_DOMAIN,
    password=SIP_PASSWORD,
    transport=SIP_TRANSPORT,
    # RFC 2833 telephone events, the common PBX default.
    dtmf_mode="rtpevent",
    # Telephony audio; PCMA ("alaw") is the default in some regions.
    audio_codecs=("PCMU", "PCMA"),
    # Re-REGISTER before typical NAT UDP flow timeouts; also replaces a
    # stale registration binding after a restart (RFC 5626 +sip.instance).
    # Must be a canonical lowercase UUID.
    reg_interval=25,
    # Change per deployment: two bots sharing one AOR must not share an
    # instance_id, or each registration replaces the other's binding.
    instance_id="90c6bfb4-13f1-4b29-a8a4-f00a57dbc8d1",
)

transport = SIPTransport(
    connection,
    SIPParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
)

tts = KokoroTTSService(
    # First run downloads the model (~350MB) to the local Kokoro cache.
    settings=KokoroTTSService.Settings(voice="af_heart"),
)

worker = PipelineWorker(
    Pipeline([transport.input(), tts, transport.output()]),
    # A SIP bot spends most of its life waiting for a call; keep the
    # pipeline up instead of timing out after the default idle period.
    idle_timeout_secs=None,
)


async def main():
    # Handlers must be registered before the runner starts: the transport
    # sets up (and fires on_dialin_ready) when the worker starts.
    @transport.event_handler("on_dialin_ready")
    async def on_dialin_ready(transport, address):
        logger.info(f"Registered to SIP server, AOR = {address}")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await worker.queue_frames([TTSSpeakFrame("Hello from your SIP bot!"), EndFrame()])

    runner = WorkerRunner(handle_sigint=True)
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
