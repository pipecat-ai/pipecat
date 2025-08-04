import asyncio
import os
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.chunked_audio_sender import ChunkedAudioSenderProcessor
from pipecat.serializers.vonage import VonageFrameSerializer
from pipecat.services.openai import OpenAISTTService, OpenAITTSService, OpenAILLMService
from pipecat.transports.network.websocket_server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)

# Set your OpenAI key
os.environ["OPENAI_API_KEY"] = "<OPENAI KEY>"

SYSTEM_INSTRUCTION = """
You are OpenAI Chatbot, a friendly, helpful robot.
Your goal is to demonstrate your capabilities in a succinct way.
Your output will be converted to audio so don't include special characters in your answers.
Respond to what the user said in a creative and helpful way. Keep your responses brief. One or two sentences at most.
"""

async def run_bot_websocket_server():
    vonage_frame_serializer = VonageFrameSerializer()
    ws_transport = WebsocketServerTransport(
        host="0.0.0.0",
        port=8005,
        params=WebsocketServerParams(
            serializer=vonage_frame_serializer,
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=True,
            vad_analyzer=SileroVADAnalyzer(),
            session_timeout=60 * 3,
        ),
    )

    stt = OpenAISTTService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-transcribe",
        prompt="Expect words based on questions with various topics, such as technology, science, and culture.",
    )

    tts = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        voice="coral",
        instructions="There could be new line characters in text like \\n which you can ignore while conversion to speech audio",
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            ws_transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            ChunkedAudioSenderProcessor(
                chunk_duration_ms=20,
                sample_rate=16000,
                channels=1,
                sample_width=2,
                delay=0.01
            ),
            ws_transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @ws_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @ws_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    @ws_transport.event_handler("on_websocket_ready")
    async def on_websocket_ready(client):
        logger.info("Server WebSocket ready")

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(run_bot_websocket_server())
