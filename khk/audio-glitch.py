#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import time
import statistics

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import Frame, TTSStartedFrame, TTSStoppedFrame, TTSAudioRawFrame
from pipecat.pipeline.parallel_pipeline import ParallelPipeline


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


class AudioTimingProcessor(FrameProcessor):
    def __init__(self, print_interval=False):
        super().__init__()
        self.print_interval = print_interval
        self.tts_started_time = None
        self.tts_stopped_time = None
        self.tts_last_frame_time = None
        self.tts_audio_frame_intervals = []
        self.tts_audio_frame_count = 0
        self.dummy_sum_of_intervals = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            self.tts_started_time = time.time()
        elif isinstance(frame, TTSAudioRawFrame):
            self.tts_audio_frame_count += 1
            if self.tts_last_frame_time is not None:
                self.tts_audio_frame_intervals.append(time.time() - self.tts_last_frame_time)
                # tiny but pointless amount of computation
                self.dummy_sum_of_intervals += time.time() - self.tts_audio_frame_intervals[-1] + sum(i * i for i in range(10000))

            self.tts_last_frame_time = time.time()
        elif isinstance(frame, TTSStoppedFrame):
            self.print_intervals()
            self.tts_stopped_time = time.time()
            self.tts_audio_frame_count = 0
            self.tts_audio_frame_intervals = []

        await self.push_frame(frame, direction)

    def print_intervals(self):
        if not self.print_interval:
            return
        
        # print max, min, median, audio frame count. 
        if self.tts_audio_frame_intervals:
            logger.info(f"TTS audio frame intervals: max={max(self.tts_audio_frame_intervals):.2f}, min={min(self.tts_audio_frame_intervals):.2f}, median={statistics.median(self.tts_audio_frame_intervals):.2f}, audio frame count={self.tts_audio_frame_count}")
        else:
            logger.info(f"TTS audio frame intervals: no data available, audio frame count={self.tts_audio_frame_count}")


async def run_bot(transport: BaseTransport):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

 
    # Create a bunch of the above simple processors to test audio frame delay glitching.
    # On my machine, 200 processors causes a big problem. 100 shows just occasional very small glitches.
    # Commit 061f2086b278f8df11cef73a6170d8413ef6334a is worse than current main (which makes sense).
    NUM_PROCESSORS_IN_PARALLEL_PIPELINE = 200
    silent_timing_processors = [AudioTimingProcessor() for _ in range(NUM_PROCESSORS_IN_PARALLEL_PIPELINE-1)]
    extra_processors = ParallelPipeline(
        [AudioTimingProcessor(print_interval=True)], 
        [*silent_timing_processors, AudioTimingProcessor(print_interval=True)]
    )


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
            rtvi,  # RTVI processor
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            extra_processors,
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

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
