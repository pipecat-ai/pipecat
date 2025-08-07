import os
import sys

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.services.deepgram import DeepgramSTTService
from deepgram.clients.listen.v1.websocket.options import LiveOptions
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.serializers.twilio import TwilioFrameSerializer
# from exotel import ExotelFrameSerializer
from pipecat.serializers.livekit import LivekitFrameSerializer
from pipecat.serializers.protobuf import ProtobufFrameSerializer

from loguru import logger

from dotenv import load_dotenv
from custom.transport.ExotelWebsocketTransport import ExotelWebsocketTransport, ExotelWebsocketParams
from custom.serializers.exotel import ExotelFrameSerializer

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def run_bot(websocket_client, stream_sid):
    transport = FastAPIWebsocketTransport(
        # transport = ExotelWebsocketTransport(
        websocket=websocket_client,
        input_name="input.pcm",
        output_name="output.wav",
        params=FastAPIWebsocketParams(
            # params=ExotelWebsocketParams(
            audio_out_enabled=True,
            add_wav_header=True,
            audio_in_enabled=True,
            audio_in_channels=1,
            audio_out_sample_rate=8000,
            audio_out_bitrate=128000,
            camera_out_enabled=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            serializer=TwilioFrameSerializer(stream_sid=stream_sid, params=TwilioFrameSerializer.InputParams(
                sample_rate=8000,
            )),
            # serializer=ExotelFrameSerializer(stream_sid=stream_sid, params=ExotelFrameSerializer.InputParams(
            #     sample_rate=8000,
            # )),
        ),
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            encoding="linear16",
            channels=1,
            model='nova-2-general',
            punctuate=True,
            interim_results=True,
            endpointing=500,
            utterance_end_ms=1000,
        )
    )
# | {'event': 'start', 'stream_sid': '3357459ca698d8c765bac10b0ec418bl', 'sequence_number': '1', 'start': {'stream_sid': '3357459ca698d8c765bac10b0ec418bl', 'call_sid': 'b0c8aea60aceb0007fc5a1bb75c218bl', 'account_sid': 'kritibudh1', 'from': '09992750105', 'to': '04446972319', 'media_format': {'encoding': 'base64', 'sample_rate': '8000', 'bit_rate': '128kbps'}}}

    # tts = CartesiaTTSService(
    #     api_key=os.getenv("CARTESIA_API_KEY"),
    #     voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
    #     sample_rate=8000,
    #     encoding="pcm_s16le",
    #     container='raw',
    # )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        model="eleven_multilingual_v2",
        output_format="mp3_44100_64",
    )
    # 'mp3_22050_32', 'mp3_44100_32', 'mp3_44100_64',

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in an audio call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text
            context_aggregator.user(),
            llm,  # LLM
            tts,  # Text-To-Speech
            transport.output(),  # Websocket output to client
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        messages.append(
            {"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMMessagesFrame(messages)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.queue_frames([EndFrame()])

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
