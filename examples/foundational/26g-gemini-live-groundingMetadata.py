import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.frames import LLMSearchResponseFrame
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=False,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=False,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=False,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
    ),
}

SYSTEM_INSTRUCTION = """
You are a helpful AI assistant that actively uses Google Search to provide up-to-date, accurate information.

IMPORTANT: For ANY question about current events, news, recent developments, real-time information, or anything that might have changed recently, you MUST use the google_search tool to get the latest information.

You should use Google Search for:
- Current news and events
- Recent developments in any field
- Today's weather, stock prices, or other real-time data
- Any question that starts with "what's happening", "latest", "recent", "current", "today", etc.
- When you're not certain about recent information

Always be proactive about using search when the user asks about anything that could benefit from real-time information.

Your output will be converted to audio so don't include special characters in your answers.

Respond to what the user said in a creative and helpful way, always using search for current information.
"""


class GroundingMetadataProcessor(FrameProcessor):
    """Processor to capture and display grounding metadata from Gemini Live API."""

    def __init__(self):
        super().__init__()
        self._grounding_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMSearchResponseFrame):
            self._grounding_count += 1
            logger.info(f"\n\n🔍 GROUNDING METADATA RECEIVED #{self._grounding_count}\n")
            logger.info(f"📝 Search Result Text: {frame.search_result[:200]}...")

            if frame.rendered_content:
                logger.info(f"🔗 Rendered Content: {frame.rendered_content}")

            if frame.origins:
                logger.info(f"📍 Number of Origins: {len(frame.origins)}")
                for i, origin in enumerate(frame.origins):
                    logger.info(f"  Origin {i + 1}: {origin.site_title} - {origin.site_uri}")
                    if origin.results:
                        logger.info(f"    Results: {len(origin.results)} items")

        # Always push the frame downstream
        await self.push_frame(frame, direction)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting Gemini Live Grounding Metadata Test Bot")

    # Create tools using ToolsSchema with custom tools for Gemini
    tools = ToolsSchema(
        standard_tools=[],  # No standard function declarations needed
        custom_tools={AdapterType.GEMINI: [{"google_search": {}}, {"code_execution": {}}]},
    )

    llm = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=SYSTEM_INSTRUCTION,
        voice_id="Charon",  # Aoede, Charon, Fenrir, Kore, Puck
        transcribe_user_audio=True,
        tools=tools,
    )

    # Create a processor to capture grounding metadata
    grounding_processor = GroundingMetadataProcessor()

    messages = [
        {
            "role": "user",
            "content": "Please introduce yourself and let me know that you can help with current information by searching the web. Ask me what current information I'd like to know about.",
        },
    ]

    # Set up conversation context and management
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            llm,
            grounding_processor,  # Add our grounding processor here
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
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
