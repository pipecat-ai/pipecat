#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import tempfile

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
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


sample_file_path = ""


async def create_sample_file():
    if sample_file_path:
        return sample_file_path
    else:
        """Create a sample text file for testing the File API."""
        content = """# Sample Document for Gemini File API Test

    This is a test document to demonstrate the Gemini File API functionality.

    ## Key Information:
    - This document was created for testing purposes
    - It contains information about AI assistants
    - The document should be analyzed by Gemini
    - The secret phrase for the test is "Pineapple Pizza"

    ## AI Assistant Capabilities:
    1. Natural language processing
    2. File analysis and understanding
    3. Context-aware conversations
    4. Multi-modal interactions

    ## Conclusion:
    This document serves as a test case for the Gemini File API integration with Pipecat.
    The AI should be able to reference and discuss the contents of this file.
    """

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            return f.name


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting File API bot")

    # Create a sample file to upload
    sample_file_path = await create_sample_file()
    logger.info(f"Created sample file: {sample_file_path}")

    system_instruction = """
    You are a helpful AI assistant with access to a document that has been uploaded for analysis.

    The document contains test information.
    You should be able to:
    - Reference and discuss the contents of the uploaded document
    - Answer questions about what's in the document
    - Use the information from the document in our conversation

    Your output will be converted to audio so don't include special characters in your answers.
    Be friendly and demonstrate your ability to work with the uploaded file.
    """

    # Initialize Gemini service with File API support
    llm = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=system_instruction,
        voice_id="Charon",  # Aoede, Charon, Fenrir, Kore, Puck
        transcribe_user_audio=True,
    )

    # Upload the sample file to Gemini File API
    logger.info("Uploading file to Gemini File API...")
    file_info = None
    try:
        file_info = await llm.file_api.upload_file(
            sample_file_path, display_name="Sample Test Document"
        )
        logger.info(f"File uploaded successfully: {file_info['file']['name']}")

        # Get file URI and mime type
        file_uri = file_info["file"]["uri"]
        mime_type = "text/plain"

        # Create context with file reference
        context = OpenAILLMContext(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Greet the user and let them know you have access to a document they can ask you about. Mention that you can discuss its contents.",
                        },
                        {
                            "type": "file_data",
                            "file_data": {"mime_type": mime_type, "file_uri": file_uri},
                        },
                    ],
                }
            ]
        )

        logger.info("File reference added to conversation context")

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        # Continue with a basic context if file upload fails
        context = OpenAILLMContext(
            [
                {
                    "role": "user",
                    "content": "Greet the user and explain that there was an issue with file upload, but you're ready to help with other tasks.",
                }
            ]
        )

    # Create context aggregator
    context_aggregator = llm.create_context_aggregator(context)

    # Build the pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            llm,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    # Configure the pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # Handle client connection event
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation using standard context frame
        await task.queue_frames([LLMRunFrame()])

    # Handle client disconnection events
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    # Run the pipeline
    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)

    # Clean up: delete the uploaded file and temporary file
    if file_info:
        try:
            await llm.file_api.delete_file(file_info["file"]["name"])
            logger.info("Cleaned up uploaded file from Gemini")
        except Exception as e:
            logger.error(f"Error cleaning up file: {e}")

    # Remove temporary file
    try:
        os.unlink(sample_file_path)
        logger.info("Cleaned up temporary file")
    except Exception as e:
        logger.error(f"Error removing temporary file: {e}")


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    upload_example_file = input("""

        Please pass in a TEXT filepath to test upload.
        NOTE: Files are stored on Google's servers for 48 hours.

        Press Enter to use a default test file.

        text filepath : """)
    if upload_example_file:
        print(f"Uploading file: {upload_example_file}")
        sample_file_path = upload_example_file.strip()
    else:
        print(f"Using default file")

    main()
