#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import EndFrame, LLMMessagesAppendFrame, LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.google.llm_vertex import (
    GoogleVertexLLMService,
)

from pipecat.services.gemini_multimodal_live.gemini import (
    GeminiMultimodalLiveLLMService,
    GeminiMultimodalModalities,
    InputParams,
)
from pipecat.services.gemini_multimodal_live.gemini_vertex import (
    GoogleVertexMultimodalLiveLLMService,
)

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

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    ##### temp using this as test bed for gemini/vertex refactor #####
    ##### 1. vanilla google llm
    # llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))
    # # llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.0-flash-live-preview-04-09") ## can't use this model here

    # #### 2. google llm on vertex ai
    # llm = GoogleVertexLLMService(
    #     credentials=os.getenv("GOOGLE_TEST_CREDENTIALS"),
    #     params=GoogleVertexLLMService.InputParams(
    #         project_id=os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
    #     ),
    #     # model="google/gemini-2.0-flash-live-preview-04-09", ## can't use this model here
    # )

    # ##### 3. vanilla live llm
    # llm = GeminiMultimodalLiveLLMService(
    #     api_key=os.getenv("GOOGLE_API_KEY"),
    #     # params=InputParams(modalities=GeminiMultimodalModalities.AUDIO),
    #     params=InputParams(modalities=GeminiMultimodalModalities.TEXT),
    #     # model="models/gemini-2.0-flash-live-001"
    #     # model="gemini-2.0-flash-live-preview-04-09", ## can't use this model here
    #     # model="models/gemini-2.0-flash-live-preview-04-09"
    # )

    #### 4. live llm on vertex ai
    llm = GoogleVertexMultimodalLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        params=GoogleVertexMultimodalLiveLLMService.InputParams(
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
            # modalities="TEXT", 
            # modalities=GeminiMultimodalModalities.TEXT, #ug, figure out why this isn't a string later
        ),
        # model="models/gemini-2.0-flash-live-001"
        # model="gemini-2.0-flash-live-preview-04-09"
        model="models/gemini-2.0-flash-live-preview-04-09"
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    async def handle_user_idle(user_idle: UserIdleProcessor, retry_count: int) -> bool:
        if retry_count == 1:
            # First attempt: Add a gentle prompt to the conversation
            print(f"_____17-detect-user-idle.py * handle_user_idle:::::::::::::::")
            message = {
                "role": "system",
                "content": "The user has been quiet. Politely and briefly ask if they're still there.",
            }
            await user_idle.push_frame(LLMMessagesAppendFrame([message], run_llm=True))
            return True
        elif retry_count == 2:
            # Second attempt: More direct prompt
            print(f"_____17-detect-user-idle.py * retry:::::::::::::")
            message = {
                "role": "system",
                "content": "The user is still inactive. Ask if they'd like to continue our conversation.",
            }
            await user_idle.push_frame(LLMMessagesAppendFrame([message], run_llm=True))
            return True
        else:
            # Third attempt: End the conversation
            await user_idle.push_frame(
                TTSSpeakFrame("It seems like you're busy right now. Have a nice day!")
            )
            await task.queue_frame(EndFrame())
            return False

    user_idle = UserIdleProcessor(callback=handle_user_idle, timeout=5.0)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            # stt,
            # user_idle,  # Idle user check-in
            context_aggregator.user(),
            llm,  # LLM
            # tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),
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
