#!/usr/bin/env python3
"""
Phone chatbot using Daily's SIP trunk capabilities with pipecat.
This implementation uses Daily instead of Twilio for audio streaming.
"""

import os
import json
import logging
import asyncio
from datetime import datetime

from fastapi import FastAPI, Request, WebSocket, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv

# Daily API client
import httpx

# Pipecat imports
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport, FastAPIWebsocketParams
)
from pipecat.serializers.daily import DailyFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import (
    LLMMessagesFrame, LLMTextFrame, TTSStartedFrame, TTSAudioRawFrame,
    TTSStoppedFrame, EndFrame
)

from pipecat.services.azure.stt import AzureSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger("daily_bot")
load_dotenv(override=True)

# Initialize FastAPI app
app = FastAPI()
logger.info("Daily phone-bot runner started")

# Daily API configuration
DAILY_API_KEY = os.getenv("DAILY_API_KEY")
DAILY_API_URL = "https://api.daily.co/v1"
PUBLIC_HOST = os.getenv("PUBLIC_HOSTNAME")  # e.g. xyz.ngrok-free.app

# Daily Room and SIP configuration cache
daily_rooms = {}
sip_configs = {}

# -------------------------------------------------------------------------
# Helper functions for Daily API
# -------------------------------------------------------------------------

async def create_daily_room(room_name=None):
    """Create a Daily room for the call."""
    if not room_name:
        room_name = f"phone-call-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
    url = f"{DAILY_API_URL}/rooms"
    headers = {"Authorization": f"Bearer {DAILY_API_KEY}"}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            headers=headers,
            json={
                "name": room_name,
                "properties": {
                    "start_audio_off": False,
                    "start_video_off": True,
                }
            }
        )
    
    if response.status_code != 200:
        logger.error(f"Failed to create Daily room: {response.text}")
        raise HTTPException(status_code=500, detail="Failed to create Daily room")
    
    room_data = response.json()
    daily_rooms[room_name] = room_data
    logger.info(f"Created Daily room: {room_name}")
    return room_data

async def create_sip_config(room_name):
    """Create a SIP configuration for the Daily room."""
    url = f"{DAILY_API_URL}/meeting-tokens"
    headers = {"Authorization": f"Bearer {DAILY_API_KEY}"}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            headers=headers,
            json={
                "properties": {
                    "room_name": room_name,
                    "is_owner": True,
                }
            }
        )
    
    if response.status_code != 200:
        logger.error(f"Failed to create meeting token: {response.text}")
        raise HTTPException(status_code=500, detail="Failed to create meeting token")
    
    token_data = response.json()
    
    # Create SIP configuration
    url = f"{DAILY_API_URL}/rooms/{room_name}/sip-trunk-configs"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            headers=headers,
            json={
                "token": token_data["token"],
                "alias": f"phone-call-{room_name}",
                "webhook_url": f"https://{PUBLIC_HOST}/daily_webhook",
            }
        )
    
    if response.status_code != 200:
        logger.error(f"Failed to create SIP config: {response.text}")
        raise HTTPException(status_code=500, detail="Failed to create SIP configuration")
    
    sip_config = response.json()
    sip_configs[room_name] = sip_config
    logger.info(f"Created SIP config for room: {room_name}")
    return sip_config

async def dial_number(to_number, room_name):
    """Dial a phone number using Daily's SIP trunk."""
    if room_name not in sip_configs:
        logger.error(f"No SIP config found for room: {room_name}")
        raise HTTPException(status_code=400, detail="Room not configured for SIP")
    
    sip_config = sip_configs[room_name]
    url = f"{DAILY_API_URL}/sip-trunks/{sip_config['sip_trunk_id']}/dial-out"
    headers = {"Authorization": f"Bearer {DAILY_API_KEY}"}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            headers=headers,
            json={
                "to_number": to_number,
                "from_number": os.getenv("DAILY_SIP_FROM_NUMBER"),
                "room_name": room_name,
                "alias": sip_config["alias"],
            }
        )
    
    if response.status_code != 200:
        logger.error(f"Failed to dial number: {response.text}")
        raise HTTPException(status_code=500, detail="Failed to dial number")
    
    call_data = response.json()
    logger.info(f"Dialed {to_number} for room {room_name}")
    return call_data

# -------------------------------------------------------------------------
# API endpoints
# -------------------------------------------------------------------------

@app.post("/start_call")
async def start_call(req: Request):
    """Start a phone call using Daily's SIP trunk."""
    body = await req.json()
    to_number = body.get("to")
    if not to_number:
        return JSONResponse({"error": "Missing 'to' parameter"}, status_code=400)
    
    # Create a Daily room
    room_data = await create_daily_room()
    room_name = room_data["name"]
    
    # Create SIP configuration
    sip_config = await create_sip_config(room_name)
    
    # Dial the number
    call_data = await dial_number(to_number, room_name)
    
    return {
        "status": "success",
        "call_id": call_data["call_id"],
        "room_name": room_name
    }

@app.post("/daily_webhook")
async def daily_webhook(req: Request):
    """Webhook for Daily call events."""
    body = await req.json()
    logger.info(f"Daily webhook event: {body.get('event_type')}")
    return {"status": "ok"}

@app.websocket("/ws/daily/{room_name}")
async def ws_daily(ws: WebSocket, room_name: str):
    """WebSocket endpoint for Daily room."""
    await ws.accept()
    logger.info(f"WebSocket accepted for room: {room_name}")
    
    try:
        # Configure transport with Daily serializer
        serializer = DailyFrameSerializer()
        transport = FastAPIWebsocketTransport(
            websocket=ws,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                serializer=serializer,
            ),
        )
        
        # Configure services
        stt = AzureSTTService(
            api_key=os.getenv("AZURE_SPEECH_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
        )
        
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        )
        
        # Configure TTS with ElevenLabs
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", "Priya"),
            model=os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2"),
            sample_rate=16000,  # Daily supports higher quality audio
        )
        
        # Configure LLM context
        system_prompt = os.getenv("BOT_SYSTEM_PROMPT", "You are a helpful phone agent.")
        ctx = OpenAILLMContext([{"role": "system", "content": system_prompt}])
        agg = llm.create_context_aggregator(ctx)
        
        # Configure pipeline
        pipeline = Pipeline([
            transport.input(),
            stt,
            agg.user(),
            llm,
            agg.assistant(),
            tts,
            transport.output(),
        ])
        
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=16000,  # Daily supports higher quality
                audio_out_sample_rate=16000,
                allow_interruptions=True,
            ),
        )
        
        # Configure event handlers
        @transport.event_handler("on_client_connected")
        async def on_connect(_, client):
            logger.info("Client connected → sending greeting")
            await task.queue_frames([
                LLMTextFrame(text="Hello! This is a test. Can you hear me clearly? Please respond if you can hear this message."),
            ])
        
        @transport.event_handler("on_client_disconnected")
        async def on_disconnect(_, client):
            logger.info("Client disconnected")
            await task.queue_frames([EndFrame()])
        
        # Run the pipeline
        await PipelineRunner(handle_sigint=False).run(task)
        
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
    finally:
        await ws.close()
        logger.info("WebSocket closed")

# -------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "daily_bot:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True
    ) 