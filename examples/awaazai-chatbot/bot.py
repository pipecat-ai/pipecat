import os
import sys

from dotenv import load_dotenv
from loguru import logger
from deepgram import LiveOptions
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.logger import FrameLogger
from pipecat.serializers.awaazai import AwaazAIFrameSerializer

from pipecat.services.elevenlabs import ElevenLabsTTSService, Language
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)


load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def run_bot(websocket_client, stream_id ):
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_out_enabled=True,
            add_wav_header=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(sample_rate=8000),
            vad_audio_passthrough=True,
            serializer=AwaazAIFrameSerializer(stream_id),
            audio_out_sample_rate=8000
        ),
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"),live_options=LiveOptions(sample_rate=8000,))

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="CwhRBWXzGAHq8TQ4Fs17",
        output_format="pcm_8000",
        params=ElevenLabsTTSService.InputParams(language=Language.EN),
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in an audio call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)
    fl = FrameLogger("LLM Output")

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text
            context_aggregator.user(),
            llm,  # LLM
            fl,
            tts,  # Text-To-Speech
            transport.output(),  # Websocket output to client
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.queue_frames([EndFrame()])

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)