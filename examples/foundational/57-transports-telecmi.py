"""
TeleCMI Transport Example

This example demonstrates how to build a voice agent that connects to phone calls
using the TeleCMI (PSTN) transport.

For instructions on how to install the `pipecat-ai[telecmi]` dependencies, buy a number,
and configure this example, please see the TeleCMI Transport README:
https://github.com/pipecat-ai/pipecat/tree/main/src/pipecat/transports/telecmi
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.telecmi import (
    TelecmiTransport,
    TelecmiParams,
    TelecmiAPI,
)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def create_session(url: str, token: str, room_name: str, metadata: dict, **kwargs):
    """
    Fired dynamically by the TelecmiAPI when a new call connects.
    """
    logger.info(f"Incoming call. Room: {room_name}")

    # Optional metadata passed via TeleCMI
    metadata = metadata or {}
    customer_name = metadata.get("customer_name", "there")

    params = TelecmiParams()

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",  # British Reading Lady
    )

   
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    transport = TelecmiTransport(
        url=url,
        token=token,
        room_name=room_name,
        params=params,
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # Speech to text
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # Text to speech
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"First participant joined TeleCMI room: {participant}")
        context.add_message(
            {"role": "system", "content": f"Please introduce yourself to {customer_name}."}
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"Participant left TeleCMI room {room_name}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    logger.info(f"Starting Pipecat Pipeline runner over TeleCMI for room {room_name}...")

    await runner.run(task)


async def main():
    agent_id = os.getenv("AGENT_ID")
    agent_token = os.getenv("AGENT_TOKEN")

    if not agent_id or not agent_token:
        logger.error("Missing AGENT_ID or AGENT_TOKEN inside .env")
        sys.exit(1)

    # The TelecmiAPI handles the Socket.IO connection to receive inbound calls
    # and spawns 'create_session' continuously as calls arrive.
    agent = TelecmiAPI(
        agent_id=agent_id,
        agent_token=agent_token,
        create_session=create_session,
        debug=True,
    )

    logger.info(f"🚀 TeleCMI API connecting for ID: {agent_id}")
    await agent.connect()


if __name__ == "__main__":
    asyncio.run(main())
