#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotInterruptionFrame,
    CancelFrame,
    EndFrame,
    Frame,
    LLMTextFrame,
    StartFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.sync.base_notifier import BaseNotifier
from pipecat.sync.event_notifier import EventNotifier
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


class VoicemailDetector(ParallelPipeline):
    def __init__(self, llm: LLMService):
        # Initialize LLM
        self._classifier_llm = llm
        self._messages = [
            {
                "role": "system",
                "content": """You are a voicemail detection classifier. Your job is to determine if the caller is leaving a voicemail message or trying to have a live conversation.

VOICEMAIL INDICATORS (respond "YES"):
- One-way communication (caller talks without expecting immediate responses)
- Messages like "Hi, this is [name], please call me back"
- "I'm calling about..." followed by details without pausing for response
- "Leave me a message" or "call me when you get this"
- Monologue-style speech patterns
- Mentions of time/date when they're calling
- Business-like messages with contact information

CONVERSATION INDICATORS (respond "NO"):
- Interactive speech ("Hello?", "Are you there?", "Can you hear me?")
- Questions directed at the recipient expecting immediate answers
- Responses to prompts or questions
- Back-and-forth dialogue patterns
- Greetings expecting responses ("Hi, how are you?")
- Real-time problem solving or discussion

Respond with ONLY "YES" if it's a voicemail, or "NO" if it's a conversation attempt. Do not explain your reasoning.""",
            },
        ]
        self._context = OpenAILLMContext(self._messages)
        self._context_aggregator = llm.create_context_aggregator(self._context)
        self._conversation_notifier = EventNotifier()
        self._classifier_gate = self.ClassifierGate(self._conversation_notifier)
        self._voicemail_processor = self.VoicemailProcessor(self._conversation_notifier)
        self._passthrough_processor = self.PassThroughProcessor()

        super().__init__(
            # Conversation branch
            [self._passthrough_processor],
            # Classifer branch
            [
                self._classifier_gate,
                self._context_aggregator.user(),
                self._classifier_llm,
                self._voicemail_processor,
                self._context_aggregator.assistant(),
            ],
        )

    class ClassifierGate(FrameProcessor):
        def __init__(self, notifier: BaseNotifier):
            super().__init__()
            self._notifier = notifier
            self._gate_opened = True
            self._gate_task: Optional[asyncio.Task] = None

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, StartFrame):
                # Start the task immediately, don't wait for other conditions
                self._gate_task = self.create_task(self._wait_for_notification())
                logger.info(f"{self}: Gate task started, waiting for notification")

            elif isinstance(frame, (EndFrame, CancelFrame)):
                if self._gate_task:
                    await self.cancel_task(self._gate_task)
                    self._gate_task = None

            if self._gate_opened:
                await self.push_frame(frame, direction)
            elif not self._gate_opened and isinstance(frame, BotInterruptionFrame):
                await self.push_frame(frame, direction)

        async def _wait_for_notification(self):
            try:
                logger.info(f"{self}: Waiting for notification...")
                await self._notifier.wait()
                logger.info(f"{self}: Received notification!")

                if self._gate_opened:
                    self._gate_opened = False
                    logger.info(f"{self}: Gate closed")
            except asyncio.CancelledError:
                logger.debug(f"{self}: Gate task was cancelled")
                raise
            except Exception as e:
                logger.exception(f"{self}: Error in gate task: {e}")
                raise

    class VoicemailProcessor(FrameProcessor):
        def __init__(self, notifier: BaseNotifier):
            super().__init__()
            self._notifier = notifier

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMTextFrame):
                # Check if the frame is a NO response, notify the notifier
                response = frame.text.strip().upper()
                print(f"Response from LLM: {response}")
                if "NO" in response:
                    logger.info(f"{self}: User conversation, notifying to close gate")
                    await self._notifier.notify()
                elif "YES" in response:
                    logger.info(f"{self}: User is leaving a voicemail, push BotInterruptionFrame")
                    # If the user is leaving a voicemail, we push a BotInterruptionFrame
                    await self._notifier.notify()
                    await self.push_frame(BotInterruptionFrame(), FrameDirection.UPSTREAM)
                    # How do we know when to send this?!
                    await asyncio.sleep(3)
                    await self.push_frame(TTSSpeakFrame("This is Mark. Call me back later."))

            else:
                # Push the frame
                await self.push_frame(frame, direction)

    class PassThroughProcessor(FrameProcessor):
        def __init__(self):
            super().__init__()

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    voicemail_detector_llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    voicemail_detector = VoicemailDetector(voicemail_detector_llm)

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
            transport.input(),  # Transport user input
            stt,
            voicemail_detector,
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
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # # Kick off the conversation.
        # messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        # await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
