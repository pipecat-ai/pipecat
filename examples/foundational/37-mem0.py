#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Mem0 Personalized Voice Agent Example with Pipecat.

This example demonstrates how to create a conversational AI assistant with memory capabilities
using Mem0 integration. It shows how to build an agent that remembers previous interactions
and personalizes responses based on conversation history.

The example:
    1. Sets up a video/audio conversation between a user and an AI assistant
    2. Uses Mem0 to store and retrieve memories from conversations
    3. Creates personalized greetings based on previous interactions
    4. Handles multi-modal interaction through audio
    5. Demonstrates two approaches for memory management:
       - Using Mem0 API (cloud-based memory storage)
       - Using local configuration with custom LLM (self-hosted memory)

Example usage (run from pipecat root directory):
    $ pip install "pipecat-ai[daily,openai,elevenlabs,silero,mem0]"
    $ python examples/foundational/37-mem0.py

Requirements:
    - OpenAI API key (for GPT-4o-mini)
    - ElevenLabs API key (for text-to-speech)
    - Daily API key (for video/audio transport)
    - Mem0 API key (for cloud-based memory storage)
    - [Optional] Anthropic API key (if using Claude with local config)

    Environment variables (set in .env or in your terminal using `export`):
        DAILY_SAMPLE_ROOM_URL=daily_sample_room_url
        DAILY_API_KEY=daily_api_key
        OPENAI_API_KEY=openai_api_key
        ELEVENLABS_API_KEY=elevenlabs_api_key
        MEM0_API_KEY=mem0_api_key
        ANTHROPIC_API_KEY=anthropic_api_key (if using Claude with local config)

The bot runs as part of a pipeline that processes audio frames and manages the conversation flow.
"""

import os
from typing import Union

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
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.mem0.memory import Mem0MemoryService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

try:
    from mem0 import Memory, MemoryClient  # noqa: F401
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Mem0, you need to `pip install mem0ai`. Also, set the environment variable MEM0_API_KEY."
    )
    raise Exception(f"Missing module: {e}")


async def get_initial_greeting(
    memory_client: Union[MemoryClient, Memory], user_id: str, agent_id: str, run_id: str
) -> str:
    """Fetch all memories for the user and create a personalized greeting.

    Returns:
        A personalized greeting based on user memories
    """
    try:
        if isinstance(memory_client, Memory):
            filters = {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}
            filters = {k: v for k, v in filters.items() if v is not None}
            memories = memory_client.get_all(**filters)
        else:
            # Create filters based on available IDs
            id_pairs = [("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)]
            clauses = [{name: value} for name, value in id_pairs if value is not None]
            filters = {"AND": clauses} if clauses else {}

            # Get all memories for this user
            memories = memory_client.get_all(filters=filters, version="v2", output_format="v1.1")

        if not memories or len(memories) == 0:
            logger.debug(f"!!! No memories found for this user. {memories}")
            return "Hello! It's nice to meet you. How can I help you today?"

        # Create a personalized greeting based on memories
        greeting = "Hello! It's great to see you again. "

        # Add some personalization based on memories (limit to 3 memories for brevity)
        if len(memories) > 0:
            greeting += "Based on our previous conversations, I remember: "
            for i, memory in enumerate(memories["results"][:3], 1):
                memory_content = memory.get("memory", "")
                # Keep memory references brief
                if len(memory_content) > 100:
                    memory_content = memory_content[:97] + "..."
                greeting += f"{memory_content} "

            greeting += "How can I help you today?"

        logger.debug(f"Created personalized greeting from {len(memories)} memories")
        return greeting

    except Exception as e:
        logger.error(f"Error retrieving initial memories from Mem0: {e}")
        return "Hello! How can I help you today?"


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
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Daily video transport
    - Speech-to-text and text-to-speech services
    - Language model integration
    - Mem0 memory service (using either API or local configuration)
    - RTVI event handling
    """
    # Note: You can pass the user_id as a parameter in API call
    USER_ID = "pipecat-demo-user"

    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # Initialize text-to-speech service
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="pNInz6obpgDQGcFmaJgB",
    )

    # =====================================================================
    # OPTION 1: Using Mem0 API (cloud-based approach)
    # This approach uses Mem0's cloud service for memory management
    # Requires: MEM0_API_KEY set in your environment
    # =====================================================================
    memory = Mem0MemoryService(
        api_key=os.getenv("MEM0_API_KEY"),  # Your Mem0 API key
        user_id=USER_ID,  # Unique identifier for the user
        agent_id="agent1",  # Optional identifier for the agent
        run_id="session1",  # Optional identifier for the run
        params=Mem0MemoryService.InputParams(
            search_limit=10,
            search_threshold=0.3,
            api_version="v2",
            system_prompt="Based on previous conversations, I recall: \n\n",
            add_as_system_message=True,
            position=1,
        ),
    )

    # =====================================================================
    # OPTION 2: Using Mem0 with local configuration (self-hosted approach)
    # This approach uses a local LLM configuration for memory management
    # Requires: Anthropic API key if using Claude model
    # =====================================================================
    # Uncomment the following code and comment out the previous memory initialization to use local config

    # local_config = {
    #     "llm": {
    #         "provider": "anthropic",
    #         "config": {
    #             "model": "claude-3-5-sonnet-20240620",
    #             "api_key": os.getenv("ANTHROPIC_API_KEY"),  # Make sure to set this in your .env
    #         }
    #     },
    #     "embedder": {
    #         "provider": "openai",
    #         "config": {
    #             "model": "text-embedding-3-large"
    #         }
    #     }
    # }

    # # Initialize Mem0 memory service with local configuration
    # memory = Mem0MemoryService(
    #     local_config=local_config,  # Use local LLM for memory processing
    #     user_id=USER_ID,            # Unique identifier for the user
    #     # agent_id="agent1",        # Optional identifier for the agent
    #     # run_id="session1",        # Optional identifier for the run
    # )

    # Initialize LLM service
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

    messages = [
        {
            "role": "system",
            "content": """You are a personal assistant. You can remember things about the person you are talking to.
                        Some Guidelines:
                        - Make sure your responses are friendly yet short and concise.
                        - If the user asks you to remember something, make sure to remember it.
                        - Greet the user by their name if you know about it.
                    """,
        },
    ]

    # Set up conversation context and management
    # The context_aggregator will automatically collect conversation context
    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            stt,
            context_aggregator.user(),
            memory,
            llm,
            tts,
            transport.output(),
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
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        # Get personalized greeting based on user memories. Can pass agent_id and run_id as per requirement of the application to manage short term memory or agent specific memory.
        greeting = await get_initial_greeting(
            memory_client=memory.memory_client, user_id=USER_ID, agent_id=None, run_id=None
        )

        # Add the greeting as an assistant message to start the conversation
        context.add_message({"role": "assistant", "content": greeting})

        # Queue the context frame to start the conversation
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")

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
