#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pattern Pair Voice Switching Example with Pipecat.

This example demonstrates how to use the PatternPairAggregator to dynamically switch
between different voices in a storytelling application. It showcases how pattern matching
can be used to control TTS behavior in streaming text from an LLM.

The example:
    1. Sets up a storytelling bot with three distinct voices (narrator, male, female)
    2. Uses pattern pairs (<voice>name</voice>) to trigger voice switching
    3. Processes the patterns in real-time as text streams from the LLM
    4. Removes the pattern tags before sending text to TTS

The PatternPairAggregator:
    - Buffers text until complete patterns are detected
    - Identifies content between start/end pattern pairs
    - Triggers callbacks when patterns are matched
    - Processes patterns that may span across multiple text chunks
    - Returns processed text at sentence boundaries

Requirements:
    - OpenAI API key (for GPT-4o)
    - Cartesia API key (for text-to-speech)
    - Daily API key (for video/audio transport)

    Environment variables (.env file):
        OPENAI_API_KEY=your_openai_key
        CARTESIA_API_KEY=your_cartesia_key
        DAILY_API_KEY=your_daily_key

Note:
    This example shows one application of PatternPairAggregator (voice switching),
    but the same approach can be used for various pattern-based text processing needs,
    such as formatting instructions, command recognition, or structured data extraction.
"""

import os

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
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.utils.text.pattern_pair_aggregator import PatternMatch, PatternPairAggregator

load_dotenv(override=True)


# Define voice IDs
VOICE_IDS = {
    "narrator": "c45bc5ec-dc68-4feb-8829-6e6b2748095d",  # Narrator voice
    "female": "71a7ad14-091c-4e8e-a314-022ece01c121",  # Female character voice
    "male": "7cf0e2b1-8daf-4fe4-89ad-f6039398f359",  # Male character voice
}

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

    # Create pattern pair aggregator for voice switching
    pattern_aggregator = PatternPairAggregator()

    # Add pattern for voice switching
    pattern_aggregator.add_pattern_pair(
        pattern_id="voice_tag",
        start_pattern="<voice>",
        end_pattern="</voice>",
        remove_match=True,
    )

    # Register handler for voice switching
    async def on_voice_tag(match: PatternMatch):
        voice_name = match.content.strip().lower()
        if voice_name in VOICE_IDS:
            # First flush any existing audio to finish the current context
            await tts.flush_audio()
            # Then set the new voice
            tts.set_voice(VOICE_IDS[voice_name])
            logger.info(f"Switched to {voice_name} voice")
        else:
            logger.warning(f"Unknown voice: {voice_name}")

    pattern_aggregator.on_pattern_match("voice_tag", on_voice_tag)

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # Initialize TTS with narrator voice as default
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=VOICE_IDS["narrator"],
        text_aggregator=pattern_aggregator,
    )

    # Initialize LLM
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    # System prompt for storytelling with voice switching
    system_prompt = """You are an engaging storyteller that uses different voices to bring stories to life.

You have three voices to use, but each has a specific purpose:

<voice>narrator</voice>
This is the default narrator voice. Use this for all narration, descriptions, and non-dialogue text.

<voice>female</voice>
Use this ONLY for direct speech by female characters (just the quoted text).

<voice>male</voice>
Use this ONLY for direct speech by male characters (just the quoted text).

IMPORTANT: Switch back to narrator voice immediately after character dialogue.

Here's an EXAMPLE of correct voice usage:

<voice>narrator</voice>
Sarah spotted her old friend across the café. She couldn't believe her eyes.

<voice>female</voice>
"Jacob! It's been so long!"

<voice>narrator</voice>
Sarah exclaimed, jumping up from her seat with a radiant smile.

<voice>male</voice>
"Sarah, is it really you? I can't believe it!"

<voice>narrator</voice>
Jacob replied, grinning widely as he walked over to her. The two friends embraced warmly, as if trying to make up for all the years spent apart.

<voice>female</voice>
"What are you doing in town? Last I heard you were in Seattle."

<voice>narrator</voice>
She asked, gesturing for him to join her at the table.

FOLLOW THESE RULES:
1. Always begin with the narrator voice
2. Only use character voices for the EXACT words they speak (in quotes)
3. SWITCH BACK to narrator voice for speech tags and all other text
4. Begin by asking what kind of story the user would like to hear
5. Create engaging dialogue with distinct characters

Remember: Use narrator voice for EVERYTHING except the actual quoted dialogue."""

    # Set up LLM context
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    # Create pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,  # TTS with pattern aggregator
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
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Start conversation - empty prompt to let LLM follow system instructions
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
