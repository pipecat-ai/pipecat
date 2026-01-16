## SSL certificate verification issue in mac -> If you face the same issue then uncomment the code and run the server again.
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# _original_create_default_context = ssl.create_default_context
# def _create_unverified_context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
#     context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
#     context.check_hostname = False
#     context.verify_mode = ssl.CERT_NONE
#     return context
# ssl.create_default_context = _create_unverified_context
# _original_SSLContext = ssl.SSLContext
# class _UnverifiedSSLContext(ssl.SSLContext):
#     def __init__(self, protocol=ssl.PROTOCOL_TLS_CLIENT):
#         super().__init__(protocol)
#         self.check_hostname = False
#         self.verify_mode = ssl.CERT_NONE
# import os
# os.environ['PYTHONHTTPSVERIFY'] = '0'

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

root_dir = Path(__file__).resolve().parent.parent.parent
env_path = root_dir / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(override=True)

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from processors.freeze_simulator import FreezeSimulatorProcessor
from processors.transcript_collector import TranscriptCollectorProcessor
from processors.bot_text_collector import BotTextCollectorProcessor
from session_store import session_store
from raw_audio_serializer import RawPCM16Serializer

# Global references
freeze_simulator: Optional[FreezeSimulatorProcessor] = None
current_task: Optional[PipelineTask] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting voice agent server...")
    yield
    logger.info("Shutting down voice agent server...")
    if session_store.get_session():
        session_store.save_audio_to_file()


app = FastAPI(
    title="infer assigment api",
    description="API for voice agent with recording, transcripts, and freeze detection",
    version="1.0.0",
    lifespan=lifespan
)

#  CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket endpoint for voice communication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time voice communication."""
    global freeze_simulator, current_task
    
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    session_id = str(uuid.uuid4())[:8]
    sample_rate = 16000
    
    # Initialize session storage
    session_store.create_session(session_id, sample_rate)
    
    try:
        # Create transport with raw audio serializer for browser WebSocket
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_sample_rate=sample_rate,
                audio_out_sample_rate=sample_rate,
                serializer=RawPCM16Serializer(sample_rate=sample_rate),
                vad_analyzer=SileroVADAnalyzer(params=VADParams(
                    confidence=0.7,
                    start_secs=0.2,
                    stop_secs=0.5,
                    min_volume=0.4,  # Higher threshold to reduce false positives
                )),
            )
        )
        
        # Initialize services
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            audio_passthrough=True
        )
        
        llm = GroqLLMService(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
        )
        
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            model="eleven_turbo_v2_5",
        )
        
        # Create bot text collector (placed after LLM to capture bot responses)
        bot_text_collector = BotTextCollectorProcessor(
            on_bot_transcript=lambda text, start: session_store.add_transcript("assistant", text, start),
            on_latency=lambda user_stop, bot_start: session_store.add_latency(user_stop, bot_start),
            on_transcript_end=lambda end_time: session_store.update_transcript_end_time(end_time),
        )
        
        # Create user transcript collector (placed after STT to capture user speech)
        transcript_collector = TranscriptCollectorProcessor(
            on_user_transcript=lambda text, start: session_store.add_transcript("user", text, start),
            on_transcript_end=lambda end_time: session_store.update_transcript_end_time(end_time),
            bot_collector=bot_text_collector,  # Link for latency coordination
        )
        
        # Create freeze simulator
        freeze_simulator = FreezeSimulatorProcessor(
            freeze_duration_seconds=3.0,
            freeze_callback=lambda start, end: session_store.add_freeze_event(start, end)
        )
        
        # Create audio buffer for recording
        # buffer_size controls when on_audio_data is triggered (in bytes)
        # 16000 samples * 2 bytes = 32000 bytes = 1 second at 16kHz mono
        audio_buffer = AudioBufferProcessor(
            sample_rate=sample_rate,
            num_channels=1,
            buffer_size=32000,  # Flush every ~1 second
        )
        
        # System prompt
        messages = [
            {
                "role": "system",
                "content": """You are a helpful AI assistant in a voice conversation. 
Keep your responses conversational and concise (1 sentence max).
Your output will be spoken aloud, so avoid special characters, bullet points, or formatting.
Be friendly and engaging.""",
            },
        ]
        
        context = LLMContext(messages)
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)
        
        # Register audio recording handler BEFORE starting recording
        @audio_buffer.event_handler("on_audio_data")
        async def on_audio_data(buffer, audio, sr, channels):
            session_store.append_audio(audio)
        
        # Build the pipeline
        # - transcript_collector after STT to capture user transcripts
        # - bot_text_collector after LLM to capture bot responses
        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                transcript_collector,  # After STT for user transcripts
                user_aggregator,
                llm,
                bot_text_collector,    # After LLM for bot transcripts
                tts,
                freeze_simulator,
                transport.output(),
                audio_buffer,
                assistant_aggregator,
            ]
        )
        
        current_task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )
        
        # Start recording after pipeline is set up
        await audio_buffer.start_recording()
        
        # Start with a greeting
        messages.append({
            "role": "system",
            "content": "Greet the user briefly and ask how you can help in 1 sentence max."
        })
        await current_task.queue_frames([LLMRunFrame()])
        
        # Run the pipeline
        runner = PipelineRunner(handle_sigint=False)
        await runner.run(current_task)
        
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if current_task:
            try:
                await audio_buffer.stop_recording()
            except:
                pass
        logger.info(f"Session {session_id} ended")


# Request/Response models
class FreezeRequest(BaseModel):
    duration_seconds: Optional[float] = 3.0


# Session Data Endpoints
@app.get("/session")
async def get_session():
    """Get current session metadata."""
    session = session_store.get_session()
    if not session:
        raise HTTPException(status_code=404, detail="No active session")
    return session.to_dict()


@app.head("/session/audio")
async def head_session_audio():
    """Check if session audio exists."""
    audio_bytes = session_store.get_audio_wav_bytes()
    if not audio_bytes:
        raise HTTPException(status_code=404, detail="No audio recorded")
    return Response(
        status_code=200,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
        }
    )


@app.get("/session/audio")
async def get_session_audio():
    """Get session audio as WAV file."""
    audio_bytes = session_store.get_audio_wav_bytes()
    if not audio_bytes:
        raise HTTPException(status_code=404, detail="No audio recorded")
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=recording.wav"}
    )


@app.post("/session/freeze")
async def trigger_freeze(request: FreezeRequest = FreezeRequest()):
    """Trigger a freeze simulation."""
    global freeze_simulator
    if not freeze_simulator:
        raise HTTPException(status_code=400, detail="No active session")
    if freeze_simulator.is_frozen:
        raise HTTPException(status_code=400, detail="Freeze already in progress")
    
    freeze_simulator._freeze_duration = request.duration_seconds
    await freeze_simulator.trigger_freeze()
    
    return {"status": "freeze_triggered", "duration_seconds": request.duration_seconds}


@app.get("/session/freeze/status")
async def get_freeze_status():
    """Get current freeze status."""
    global freeze_simulator
    return {
        "is_frozen": freeze_simulator.is_frozen if freeze_simulator else False,
        "active_session": freeze_simulator is not None
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
