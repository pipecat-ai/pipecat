#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.inworld.tts import InworldTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams


class LoggingInworldTTSService(InworldTTSService):
    """InworldTTSService with timestamp logging enabled."""

    async def _process_streaming_response(self, response):
        """Override to log timestamp info from Inworld API responses."""
        import base64
        import json

        from pipecat.services.inworld.tts import calculate_word_times_from_inworld

        buffer = ""
        async for chunk in response.content.iter_chunked(1024):
            if not chunk:
                continue
            buffer += chunk.decode("utf-8")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    chunk_data = json.loads(line_str)
                    if "result" in chunk_data and "audioContent" in chunk_data["result"]:
                        await self.stop_ttfb_metrics()
                        async for frame in self._process_audio_chunk(
                            base64.b64decode(chunk_data["result"]["audioContent"])
                        ):
                            yield frame
                    # Log and process timestamps
                    if "result" in chunk_data and "timestampInfo" in chunk_data["result"]:
                        timestamp_info = chunk_data["result"]["timestampInfo"]
                        # Log the raw timestamp info from Inworld API
                        logger.info(f"Inworld timestampInfo: {timestamp_info}")
                        word_times, new_cumulative = calculate_word_times_from_inworld(
                            timestamp_info, self._cumulative_time, self._timestamp_type
                        )
                        if word_times:
                            await self.add_word_timestamps(word_times)
                            self._cumulative_time = new_cumulative
                except json.JSONDecodeError:
                    continue

load_dotenv(override=True)

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    # Create an HTTP session
    async with aiohttp.ClientSession() as session:
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        # Inworld TTS Service - Unified streaming and non-streaming
        # Set streaming=True for real-time audio, streaming=False for complete audio generation
        # Character-level timestamps are enabled below to show word timing in RTVI events
        streaming = True  # Toggle this to switch between modes

        tts = LoggingInworldTTSService(
            api_key=os.getenv("INWORLD_API_KEY", ""),
            aiohttp_session=session,
            voice_id="Ashley",
            model="inworld-tts-1",
            streaming=streaming,  # True: real-time chunks, False: complete audio then playback
            params=InworldTTSService.InputParams(
                timestamp_type="CHARACTER",  # Enable character-level timestamps
            ),
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        messages = [
            {
                "role": "system",
                "content": "You are very knowledgable about dogs. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(context)

        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                rtvi,  # RTVI processor
                stt,  # STT
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
            idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected")
            # Kick off the conversation.
            messages.append({"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames([LLMRunFrame()])

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
