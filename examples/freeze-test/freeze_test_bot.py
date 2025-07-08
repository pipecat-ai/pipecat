#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import os
import random
from contextlib import asynccontextmanager
from typing import Any, Dict

import sentry_sdk
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMMessagesFrame,
    StartFrame,
    StartInterruptionFrame,
    StopFrame,
    StopInterruptionFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.loggers.debug_log_observer import DebugLogObserver
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIProcessor
from pipecat.processors.metrics.sentry import SentryMetrics
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.utils.time import time_now_iso8601

load_dotenv(override=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles FastAPI startup and shutdown."""
    yield  # Run app


# Initialize FastAPI app with lifespan manager
app = FastAPI(lifespan=lifespan)

# Configure CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulateFreezeInput(FrameProcessor):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Whether we have seen a StartFrame already.
        self._initialized = False
        self._send_frames_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            # Push StartFrame before start(), because we want StartFrame to be
            # processed by every processor before any other frame is processed.
            await self.push_frame(frame, direction)
            await self._start(frame)
        elif isinstance(frame, CancelFrame):
            logger.info("SimulateFreezeInput: Received cancel frame")
            await self._stop()
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            logger.info("SimulateFreezeInput: Received end frame")
            await self.push_frame(frame, direction)
            await self._stop()
        elif isinstance(frame, StopFrame):
            logger.info("SimulateFreezeInput: Received stop frame")
            await self.push_frame(frame, direction)
            await self._stop()

    async def _start(self, frame: StartFrame):
        if self._initialized:
            return
        logger.info(f"Starting SimulateFreezeInput")
        self._initialized = True
        if not self._send_frames_task:
            self._send_frames_task = self.create_task(self._send_frames())

    async def _stop(self):
        logger.info(f"Stopping SimulateFreezeInput")
        self._initialized = False
        if self._send_frames_task:
            await self.cancel_task(self._send_frames_task)
            self._send_frames_task = None

    async def _send_user_text(self, text: str):
        self.reset_watchdog()
        # Emulation as if the user has spoken and the stt transcribed
        await self.push_frame(UserStartedSpeakingFrame())
        await self.push_frame(StartInterruptionFrame())
        await self.push_frame(
            TranscriptionFrame(
                text,
                "",
                time_now_iso8601(),
            )
        )
        # Need to wait before sending the UserStoppedSpeakingFrame,
        # otherwise TranscriptionFrame will be processed
        # later than the UserStoppedSpeakingFrame
        await asyncio.sleep(0.1)
        await self.push_frame(UserStoppedSpeakingFrame())
        await self.push_frame(StopInterruptionFrame())

    async def _send_frames(self):
        try:
            i = 0
            while True:
                logger.debug("SimulateFreezeInput _send_frames")
                await self._send_user_text("Tell me a brief history of Brazil!")
                await asyncio.sleep(3)
                await self._send_user_text("and who has discovered it")
                i += 1
                if i >= 20:
                    break
                # sleeping 1s before interrupting
                wait_time = random.uniform(1, 10)
                await asyncio.sleep(wait_time)
        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")


async def run_example(websocket_client):
    logger.info(f"Starting bot")

    # Create a transport using the WebRTC connection
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=ProtobufFrameSerializer(),
        ),
    )

    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        traces_sample_rate=1.0,
    )

    freeze = SimulateFreezeInput()

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    async def handle_user_idle(user_idle: UserIdleProcessor, retry_count: int) -> bool:
        if retry_count == 1:
            # First attempt: Add a gentle prompt to the conversation
            messages.append(
                {
                    "role": "system",
                    "content": "The user has been quiet. Politely and briefly ask if they're still there.",
                }
            )
            await user_idle.push_frame(LLMMessagesFrame(messages))
            return True
        elif retry_count == 2:
            # Second attempt: More direct prompt
            messages.append(
                {
                    "role": "system",
                    "content": "The user is still inactive. Ask if they'd like to continue our conversation.",
                }
            )
            await user_idle.push_frame(LLMMessagesFrame(messages))
            return True
        else:
            # Third attempt: End the conversation
            await user_idle.push_frame(
                TTSSpeakFrame("It seems like you're busy right now. Have a nice day!")
            )
            await task.queue_frame(EndFrame())
            return False

    user_idle = UserIdleProcessor(callback=handle_user_idle, timeout=10.0)

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        metrics=SentryMetrics(),
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        metrics=SentryMetrics(),
    )

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            ParallelPipeline(
                [
                    freeze,
                ],
                [
                    transport.input(),
                    stt,
                ],
            ),
            user_idle,
            rtvi,
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
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
        ),
        idle_timeout_secs=120,
        observers=[
            DebugLogObserver(
                frame_types={
                    InterimTranscriptionFrame: None,
                    TranscriptionFrame: None,
                    # TTSTextFrame: None,
                    # LLMTextFrame: None,
                    OpenAILLMContextFrame: None,
                    LLMFullResponseEndFrame: None,
                    UserStartedSpeakingFrame: None,
                    UserStoppedSpeakingFrame: None,
                    StartInterruptionFrame: None,
                    StopInterruptionFrame: None,
                },
                exclude_fields={
                    "result",
                    "metadata",
                    "audio",
                    "image",
                    "images",
                },
            ),
        ],
        enable_watchdog_timers=True,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info(f"Client ready")
        await rtvi.set_bot_ready()
        # Kick off the conversation.
        # messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        # await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/client/")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted")
    try:
        await run_example(websocket)
    except Exception as e:
        print(f"Exception in run_bot: {e}")


@app.post("/connect")
async def bot_connect(request: Request) -> Dict[Any, Any]:
    server_mode = os.getenv("WEBSOCKET_SERVER", "fast_api")
    if server_mode == "websocket_server":
        ws_url = "ws://localhost:8765"
    else:
        ws_url = "ws://localhost:7860/ws"
    return {"ws_url": ws_url}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
