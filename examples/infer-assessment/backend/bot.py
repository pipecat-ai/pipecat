#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Quickstart Example.

The example runs a simple voice AI bot that you can connect to using your
browser and speak with it. You can also deploy this bot to Pipecat Cloud.

Required AI services:
- Deepgram (Speech-to-Text)
- OpenAI (LLM)
- Cartesia (Text-to-Speech)

Run the bot using::

    uv run bot.py
"""

import os
import uuid
import wave
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
import os
print(f"DEBUG: CWD = {os.getcwd()}")
print(f"DEBUG: DB Path = {os.path.abspath('database.db')}")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger.info("✅ Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from pipecat.frames.frames import AudioRawFrame, TTSStartedFrame, TTSStoppedFrame, UserStoppedSpeakingFrame
import random
import asyncio

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")
    
    # Ensure DB exists
    from app.database import create_db_and_tables
    from app.schema import Session
    create_db_and_tables()

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": "You are a friendly AI assistant. Respond naturally and keep your answers conversational.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    # Create a session recorder to manage recording and state
    session_id = uuid.uuid4()
    run_id = uuid.uuid4()
    
    # Ensure recordings directory exists
    recordings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
    os.makedirs(recordings_dir, exist_ok=True)
    recording_path = os.path.join(recordings_dir, f"{session_id}.wav")

    # Audio recording processor
    audiobuffer = AudioBufferProcessor(
        user_continuous_stream=False, # We want turns
    )
    
    # Transcript processor
    transcript = TranscriptProcessor()
    
    # initialize session data
    session_data = {
        "id": session_id,
        "created_at": datetime.utcnow(),
        "transcript": [],
        "freeze_events": [],
        "latency_metrics": {},
        "audio_url": f"/recordings/{session_id}.wav"
    }

    # Helper to save session to DB
    def save_session_to_db():
        try:
             # We need to use the sync engine context here or create a new session
             from app.database import engine, create_db_and_tables
             from sqlmodel import Session as SQLSession
             from app.schema import Session as SessionModel
             
             with SQLSession(engine) as db:
                 # Check if exists, else create
                 existing = db.get(SessionModel, session_id)
                 if existing:
                     existing.transcript = session_data["transcript"]
                     existing.freeze_events = session_data["freeze_events"]
                     existing.latency_metrics = session_data["latency_metrics"]
                     existing.audio_url = session_data["audio_url"]
                     db.add(existing)
                 else:
                     new_session = SessionModel(**session_data)
                     db.add(new_session)
                 db.commit()
                 logger.info(f"Saved session {session_id} to database")
        except Exception as e:
            logger.error(f"Failed to save session to DB: {e}")

    # Freeze Simulator
    class FreezeSimulator(FrameProcessor):
        def __init__(self):
            super().__init__()
            self._frozen = False
            self._freeze_end_time = 0
            self._is_bot_speaking = False
            self._current_event_idx = -1
            # Check env var to enable/disable
            self._enabled = os.getenv("ENABLE_FREEZE_SIMULATION", "false").lower() == "true"
            if self._enabled:
                logger.info("Freeze Simulation ENABLED")
        
        def _end_freeze(self):
            if self._frozen and self._current_event_idx >= 0:
                # Update the actual end time and duration
                actual_end_time = datetime.utcnow().timestamp()
                event = session_data["freeze_events"][self._current_event_idx]
                event["end_time"] = actual_end_time
                event["duration"] = actual_end_time - event["start_time"]
                
                self._frozen = False
                logger.info(f"Unfreezing bot. Actual duration: {event['duration']:.2f}s")
                save_session_to_db()

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            
            # Track TTS state
            if isinstance(frame, TTSStartedFrame):
                self._is_bot_speaking = True
                # Calculate latency at the START of speech
                if "last_user_end_time" in session_data:
                     # Calculate latency (Voice Latency)
                     start_time = datetime.utcnow().timestamp()
                     latency = start_time - session_data["last_user_end_time"]
                     session_data["current_turn_latency"] = latency
                     logger.info(f"Bot started speaking. Latency: {latency:.2f}s")
            elif isinstance(frame, TTSStoppedFrame):
                self._is_bot_speaking = False
                # If we were frozen, stop freezing now as speech ended
                if self._frozen:
                    self._end_freeze()
            
            # Track User state for latency
            if isinstance(frame, UserStoppedSpeakingFrame):
                 session_data["last_user_end_time"] = datetime.utcnow().timestamp()

            if not self._enabled:
                await self.push_frame(frame, direction)
                return

            # Check if we should freeze on AudioRawFrame (output audio)
            if isinstance(frame, AudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
                current_time = datetime.utcnow().timestamp()
                
                # If currently frozen
                if self._frozen:
                    if current_time < self._freeze_end_time:
                         # Send silence instead of dropping
                         silence_audio = b'\x00' * len(frame.audio)
                         silence_frame = AudioRawFrame(audio=silence_audio, sample_rate=frame.sample_rate, num_channels=frame.num_channels)
                         await self.push_frame(silence_frame, direction)
                         return 
                    else:
                        # Timeout reached
                        self._end_freeze()
                
                # If not frozen, small chance to freeze BUT ONLY IF SPEAKING
                elif self._is_bot_speaking and random.random() < 0.05: # 5% chance per frame while speaking
                     duration = random.uniform(1.0, 3.0) # 1-3s freeze (shorter for realism)
                     self._frozen = True
                     self._freeze_end_time = current_time + duration
                     logger.info(f"Freezing bot for max {duration:.2f}s")
                     
                     # Create new event
                     event = {
                         "start_time": current_time,
                         "end_time": self._freeze_end_time, # Placeholder
                         "duration": duration # Placeholder
                     }
                     session_data["freeze_events"].append(event)
                     self._current_event_idx = len(session_data["freeze_events"]) - 1
                     save_session_to_db()
                     
                     # Drop this frame too (send silence)
                     silence_audio = b'\x00' * len(frame.audio)
                     silence_frame = AudioRawFrame(audio=silence_audio, sample_rate=frame.sample_rate, num_channels=frame.num_channels)
                     await self.push_frame(silence_frame, direction)
                     return 
            
            await self.push_frame(frame, direction)
            
    freeze_simulator = FreezeSimulator()

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        # Save the full recording
        with wave.open(recording_path, "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)
        logger.info(f"Saved recording to {recording_path}")
        
        # After saving audio, ensure final DB update
        save_session_to_db()

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            transcript.user(), # Capture user transcript
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            freeze_simulator, # Simulate freeze (drop frames before audio output)
            transport.output(),  # Transport bot output
            audiobuffer, # Capture audio
            transcript.assistant(), # Capture bot transcript
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    # Track turns for transcript
    async def log_user_turn(turn):
        # turn is a dictionary or object with text, timestamp?
        # The context aggregator might give us messages.
        # Let's rely on processors if possible or hooking into the assistants.
        pass

    # We need to capture transcript items. 
    # A simple way is to observe the context.
    
    @transcript.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        # We process transcript updates
        
        # frame is a TranscriptionUpdateFrame? Or similar?
        # The docs say: emit events with transcript updates
        # Check docs/transcript-processor.md again or inspect code if unsure
        # But assuming it passes 'frame' which has 'messages'.
        
        # Actually docs say:
        # for message in frame.messages:
        #    print(f"[{message.timestamp}] {message.role}: {message.content}")
        
        for message in frame.messages:
             role = message.role
             content = message.content
             timestamp = datetime.utcnow().timestamp() # message.timestamp might be relative or string
             
             latency = 0.0
             
             if role == 'assistant':
                 # Use the latency calculated at the START of the turn
                 if "current_turn_latency" in session_data:
                      latency = session_data["current_turn_latency"]
                 elif "last_user_end_time" in session_data:
                      # Fallback if TTSStartedFrame wasn't caught for some reason
                      last_user_time = session_data["last_user_end_time"]
                      latency = timestamp - last_user_time
                 elif session_data["transcript"] and session_data["transcript"][-1]["role"] == "user":
                      # Fallback
                      last_user_time = session_data["transcript"][-1]["timestamp"]
                      latency = timestamp - last_user_time
                 
                 # Update average latency
                 latencies = [t["latency"] for t in session_data["transcript"] if t["role"] == "assistant"]
                 # Add current latency to calc average
                 latencies.append(latency)
                 if latencies:
                      session_data["latency_metrics"]["average_latency"] = sum(latencies) / len(latencies)

             session_data["transcript"].append({
                "role": role,
                "content": content,
                "timestamp": timestamp,
                "latency": latency
             })
        
        save_session_to_db()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Start recording
        await audiobuffer.start_recording()
        
        # Create initial DB entry
        save_session_to_db()
        
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        # Stop recording is implicit when pipeline stops usually, but audiobuffer might need a nudge to flush?
        # audiobuffer processor stops when pipeline stops.
        # But we need to ensure we save the file. 
        # The on_audio_data is called when the processor stops or we stop it.
        await audiobuffer.stop_recording()
        
        await task.cancel()
    
    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()