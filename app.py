import os
import asyncio
from dotenv import load_dotenv
from loguru import logger
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Pipecat imports
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from pipecat.transcriptions.language import Language
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.frames.frames import EndFrame, LLMRunFrame

# -----------------------------------------------------
# Load environment variables
# -----------------------------------------------------
load_dotenv(override=True)

# -----------------------------------------------------
# FastAPI Server Setup
# -----------------------------------------------------
app = FastAPI(title="Pipecat Speech2Speech", version="2.0")

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
        "message": "🎙️ Nova Voice Assistant is running!",
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
        {
            "role": "system",
            "content": (
                "You are Julia, a warm, conversational AI voice assistant. "
                "Keep your responses short and natural for real-time speech."
            ),
        }
    ]
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # --- Audio Buffer (records all user audio) ---
    audiobuffer = AudioBufferProcessor(user_continuous_stream=True)

    # --- Define processing pipeline ---
    pipeline = Pipeline([
        transport.input(),               # Mic input
        stt,                             # Speech → Text
        context_aggregator.user(),       # Update user context
        llm,                             # Generate response
        tts,                             # Text → Speech
        transport.output(),              # Send audio back to user
        audiobuffer,                     # Record audio (optional)
        context_aggregator.assistant(),  # Update assistant context
    ])

    # --- Pipeline task parameters ---
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            allow_interruptions=True,
        ),
    )

    # --- Event Hooks ---
    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info("✅ Client connected to Pipecat stream")
        # Trigger LLM startup message
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("❌ Client disconnected")
        await task.queue_frames([EndFrame()])

    # --- Run the pipeline ---
    runner = PipelineRunner(handle_sigint=handle_sigint, force_gc=True)
    await runner.run(task)

# -----------------------------------------------------
# WebSocket endpoint
# -----------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("🔌 WebSocket client connected")

    params = FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        add_wav_header=False,
        vad_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
        vad_audio_passthrough=True,
        serializer=ProtobufFrameSerializer(),
    )

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
