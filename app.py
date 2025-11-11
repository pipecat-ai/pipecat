import os
import argparse
import asyncio
from dotenv import load_dotenv
from loguru import logger

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Pipecat imports
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner

from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from pipecat.transcriptions.language import Language
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.frames.frames import EndFrame

# -----------------------------------------------------
# Load environment variables
# -----------------------------------------------------
load_dotenv(override=True)

# -----------------------------------------------------
# FastAPI Server Setup
# -----------------------------------------------------
app = FastAPI(title="Pipecat Speech2Speech", version="1.0")

# ✅ Correct way to add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "🎙️ Pipecat Realtime Speech2Speech is running",
        "websocket": "/ws",
        "health": "/health"
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

# -----------------------------------------------------
# Speech2Speech Pipeline
# -----------------------------------------------------
async def run_pipeline(transport, handle_sigint: bool = False):
    logger.info("🎤 Starting Pipecat Realtime Speech2Speech pipeline")

    # --- Services setup ---
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
    )

    stt = SpeechmaticsSTTService(
        api_key=os.getenv("SPEECHMATICS_API_KEY"),
        language=Language.EN,
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        params=BaseOpenAILLMService.InputParams(temperature=0.7),
    )

    # --- Context & aggregator ---
    messages = [
        {"role": "system", "content": "You are Julia, a warm, conversational AI voice assistant."}
    ]
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # --- Define processing pipeline ---
    task = PipelineTask(
        Pipeline(
            [
                transport.input(),               # Mic input
                stt,                             # Speech → Text
                context_aggregator.user(),       # User context
                llm,                             # LLM response
                tts,                             # Text → Speech
                transport.output(),              # Audio back to user
                context_aggregator.assistant(),  # Assistant context
            ]
        )
    )

    # --- Event hooks ---
    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info("✅ Client connected to Pipecat stream")
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("❌ Client disconnected")
        await task.queue_frames([EndFrame()])

    # --- Run pipeline ---
    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)

# -----------------------------------------------------
# WebSocket endpoint (Render will expose this as /ws)
# -----------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("🔌 WebSocket client connected")

    params = FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    )

    # ✅ Correct: pass WebSocket, not app
    transport = FastAPIWebsocketTransport(websocket, params)

    try:
        await run_pipeline(transport, handle_sigint=False)
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket client disconnected")
    except Exception as e:
        logger.exception(f"❗ Error in WebSocket pipeline: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
