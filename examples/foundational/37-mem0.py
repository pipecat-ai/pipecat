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

Example usage (run from pipecat root directory):
    $ pip install "pipecat-ai[daily,openai,elevenlabs,silero,mem0]"
    $ python examples/foundational/35-mem0.py

Requirements:
    - OpenAI API key (for GPT-4o-mini)
    - ElevenLabs API key (for text-to-speech)
    - Daily API key (for video/audio transport)
    - Mem0 API key (for memory storage and retrieval)

    Environment variables (set in .env or in your terminal using `export`):
        DAILY_SAMPLE_ROOM_URL=daily_sample_room_url
        DAILY_API_KEY=daily_api_key
        OPENAI_API_KEY=openai_api_key
        ELEVENLABS_API_KEY=elevenlabs_api_key
        MEM0_API_KEY=mem0_api_key

The bot runs as part of a pipeline that processes audio frames and manages the conversation flow.
"""

import os

from dotenv import load_dotenv
from loguru import logger
from openai import audio

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.mem0.memory import Mem0MemoryService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

load_dotenv(override=True)

try:
    from mem0 import MemoryClient
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Mem0, you need to `pip install mem0ai`. Also, set the environment variable MEM0_API_KEY."
    )
    raise Exception(f"Missing module: {e}")


async def get_initial_greeting(
    memory_client: MemoryClient, user_id: str, agent_id: str, run_id: str
) -> str:
    """Fetch all memories for the user and create a personalized greeting.

    Returns:
        A personalized greeting based on user memories
    """
    try:
        # Create filters based on available IDs
        id_pairs = [("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)]
        clauses = [{name: value} for name, value in id_pairs if value is not None]
        filters = {"AND": clauses} if clauses else {}

        # Get all memories for this user
        memories = memory_client.get_all(filters=filters, version="v2")

        if not memories or len(memories) == 0:
            logger.debug(f"!!! No memories found for this user. {memories}")
            return "Hello! It's nice to meet you. How can I help you today?"

        # Create a personalized greeting based on memories
        greeting = "Hello! It's great to see you again. "

        # Add some personalization based on memories (limit to 3 memories for brevity)
        if len(memories) > 0:
            greeting += "Based on our previous conversations, I remember: "
            for i, memory in enumerate(memories[:3], 1):
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


async def run_bot(webrtc_connection: SmallWebRTCConnection):
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Daily video transport
    - Speech-to-text and text-to-speech services
    - Language model integration
    - Mem0 memory service
    - RTVI event handling
    """
    # Note: You can pass the user_id as a parameter in API call
    USER_ID = "pipecat-demo-user"

    logger.info(f"Starting bot")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # Initialize text-to-speech service
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="pNInz6obpgDQGcFmaJgB",
    )

    # Initialize Mem0 memory service
    memory = Mem0MemoryService(
        api_key=os.getenv("MEM0_API_KEY"),
        user_id=USER_ID,  # Unique identifier for the user
        # agent_id="agent1",  # Optional identifier for the agent
        # run_id="session1", # Optional identifier for the run
        params=Mem0MemoryService.InputParams(
            search_limit=10,
            search_threshold=0.3,
            api_version="v2",
            system_prompt="Based on previous conversations, I recall: \n\n",
            add_as_system_message=True,
            position=1,
        ),
    )

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
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)
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
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Get personalized greeting based on user memories. Can pass agent_id and run_id as per requirement of the application to manage short term memory or agent specific memory.
        greeting = await get_initial_greeting(
            memory_client=memory.memory_client, user_id=USER_ID, agent_id=None, run_id=None
        )

        # Add the greeting as an assistant message to start the conversation
        context.add_message({"role": "assistant", "content": greeting})

        # Queue the context frame to start the conversation
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")

    @transport.event_handler("on_client_closed")
    async def on_client_closed(transport, client):
        logger.info(f"Client closed connection")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


if __name__ == "__main__":
    from run import main

    main()
