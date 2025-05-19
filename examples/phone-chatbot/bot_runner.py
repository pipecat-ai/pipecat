# bot_runner.py

import os
import json
import io
import wave
import asyncio
import logging
from datetime import datetime

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger("bot_runner")

# Load environment variables from .env
load_dotenv(override=True)

# Twilio REST client
from twilio.rest import Client as TwilioClient

# Pipecat transports & pipeline
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import (
    LLMMessagesFrame,
    LLMTextFrame,
    EndFrame,
    TTSStartedFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)

# Azure Speech-to-Text
from pipecat.services.azure.stt import AzureSTTService

class LoggingAzureSTTService(AzureSTTService):
    async def run(self, input_frames):
        async for frame in super().run(input_frames):
            if getattr(frame, "text", "").strip():
                logger.info(f"{datetime.now().isoformat()} USER: {frame.text}")
            yield frame

# OpenAI LLM (GPT-4o-mini)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

class LoggingLLMService(OpenAILLMService):
    async def run(self, input_frames):
        async for frame in super().run(input_frames):
            if isinstance(frame, LLMTextFrame) and frame.text.strip():
                logger.info(f"{datetime.now().isoformat()} LLM: {frame.text}")
            yield frame

# -----------------------------------------------------------------------------
# Smallest.ai Lightning v2 TTS adapter
# -----------------------------------------------------------------------------
from pipecat.services.tts_service import TTSService
import requests

class SmallestTTSService(TTSService):
    def __init__(self, api_key=None, voice_id=None, sample_rate=24000, speed=1.0):
        # Lightning v2 supports up to 24 kHz for higher fidelity
        super().__init__(sample_rate=sample_rate)
        self.api_key = api_key or os.getenv("SMALLEST_API_KEY")
        self.voice_id = voice_id or os.getenv("SMALLEST_VOICE_ID", "arman")
        self.speed = speed
        # note: sample_rate is read-only in base class

    def _chunk_text(self, text: str, max_len: int = 500):
        chunks = []
        while text:
            if len(text) <= max_len:
                chunks.append(text); break
            cut = text.rfind(".", 0, max_len)
            cut = (cut + 1) if cut != -1 else max_len
            chunks.append(text[:cut])
            text = text[cut:].lstrip()
        return chunks

    def _synthesize(self, text: str) -> bytes:
        logger.info(f"{datetime.now().isoformat()} AGENT → TTS: {text}")
        url = "https://waves-api.smallest.ai/api/v1/lightning-v2/get_speech"  # :contentReference[oaicite:1]{index=1}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        pcm_parts = []
        for chunk in self._chunk_text(text):
            payload = {
                "text": chunk,
                "voice_id": self.voice_id,
                "add_wav_header": True,
                "sample_rate": self.sample_rate,
                "speed": self.speed,
                "language": "en",        # adjust for multi-language support
                "consistency": 0.5,      # voice consistency control
                "similarity": 0.0,       # voice similarity (0=none,1=clone)
                "enhancement": 1,        # 0=no enhancement,1=default enhancement
            }
            resp = requests.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            buf = io.BytesIO(resp.content)
            with wave.open(buf, "rb") as wf:
                pcm_parts.append(wf.readframes(wf.getnframes()))
        return b"".join(pcm_parts)

    async def run_tts(self, text: str):
        audio = await asyncio.get_running_loop().run_in_executor(None, self._synthesize, text)
        yield TTSStartedFrame()
        yield TTSAudioRawFrame(audio=audio, sample_rate=self.sample_rate, num_channels=1)
        yield TTSStoppedFrame()

# -----------------------------------------------------------------------------
# FastAPI app & Twilio setup
# -----------------------------------------------------------------------------
app = FastAPI()
logger.info("Starting Pipecat phone chatbot runner")

TWILIO = TwilioClient(
    os.getenv("TWILIO_ACCOUNT_SID"),
    os.getenv("TWILIO_AUTH_TOKEN"),
)
CALLER_ID = os.getenv("TWILIO_CALLER_ID")
PUBLIC_HOST = os.getenv("PUBLIC_HOSTNAME")

@app.post("/start_call")
async def start_call(req: Request):
    data = await req.json()
    to_number = data.get("to")
    logger.info("start_call to=%s", to_number)
    if not to_number:
        return PlainTextResponse("Missing 'to' in JSON body", status_code=400)
    call = TWILIO.calls.create(
        to=to_number,
        from_=CALLER_ID,
        url=f"https://{PUBLIC_HOST}/twilio_call",
    )
    logger.info("Twilio call SID=%s", call.sid)
    return {"status": "calling", "call_sid": call.sid}

@app.post("/start")
async def start_alias(req: Request):
    return await start_call(req)

@app.api_route("/twilio_call", methods=["GET", "POST"])
async def twilio_call(request: Request):
    logger.info("Twilio webhook /twilio_call")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Start>
    <Stream url="wss://{PUBLIC_HOST}/ws/twilio" track="both"/>
  </Start>
  <Pause length="3600"/>
</Response>"""
    return PlainTextResponse(content=twiml, media_type="application/xml")

# -----------------------------------------------------------------------------
# WebSocket handler
# -----------------------------------------------------------------------------
@app.websocket("/ws/twilio")
async def ws_twilio(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket accepted")
    try:
        # Handshake
        msg = await websocket.receive_text()
        ev = json.loads(msg)
        event = ev.get("event")
        if event == "connected":
            sid, csid = ev.get("streamSid"), ev.get("callSid")
        elif event == "start":
            sid, csid = ev["start"]["streamSid"], ev["start"].get("callSid")
        else:
            logger.warning("Unexpected event %r", event); return
        logger.info("Handshake OK sid=%s csid=%s", sid, csid)

        serializer = TwilioFrameSerializer(
            stream_sid=sid,
            call_sid=csid,
            account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
            auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
        )
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                serializer=serializer,
            ),
        )

        stt = LoggingAzureSTTService(
            api_key=os.getenv("AZURE_SPEECH_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
        )
        llm = LoggingLLMService(
            name="llm",
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        )
        tts = SmallestTTSService(
            api_key=os.getenv("SMALLEST_API_KEY"),
            voice_id=os.getenv("SMALLEST_VOICE_ID", "priya"),
            sample_rate=24000,          # match Lightning v2 default
            speed=float(os.getenv("SMALLEST_VOICE_SPEED", "1.0")),
        )

        system_prompt = os.getenv("BOT_SYSTEM_PROMPT", "You are a helpful phone agent.")
        ctx = OpenAILLMContext(messages=[{"role": "system", "content": system_prompt}])
        agg = llm.create_context_aggregator(ctx)

        pipeline = Pipeline([
            transport.input(),
            stt,
            agg.user(),
            llm,
            tts,
            transport.output(),
            agg.assistant(),
        ])
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=16000,   # or match your call codec
                audio_out_sample_rate=24000,  # Lightning v2 sample rate
                allow_interruptions=True,
            ),
        )

        @transport.event_handler("on_client_connected")
        async def on_connect(_, client):
            logger.info("Client connected → greeting")
            await task.queue_frames([
                LLMMessagesFrame([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "<call started>"},
                ])
            ])

        @transport.event_handler("on_client_disconnected")
        async def on_disconnect(_, client):
            logger.info("Client disconnected → end")
            await task.queue_frames([EndFrame()])

        runner = PipelineRunner(handle_sigint=False)
        await runner.run(task)

    except Exception:
        logger.exception("WebSocket error:")
    finally:
        logger.info("Closing WebSocket")
        try:
            await websocket.close()
        except RuntimeError:
            pass

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "bot_runner:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level="debug",
    )
