#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import datetime
import io
import json
import os
import sys
import wave
from pathlib import Path

import aiofiles
from dotenv import load_dotenv
from fastapi import WebSocket
from loguru import logger
from openai.types.chat import ChatCompletionAssistantMessageParam
from smart_endpointing import (
    CLASSIFIER_MODEL,
    AudioAccumulator,
    CompletenessCheck,
    OutputGate,
    StatementJudgeContextFilter,
    classifier_system_instruction,
)

from pipecat.audio.filters.krisp_filter import KrispFilter
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    TTSSpeakFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.filters.custom_mute_filter import TransportInputFilter
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.filters.stt_mute_filter import STTMuteConfig, STTMuteFilter, STTMuteStrategy
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.atoms.agent import (
    AtomsAgentContext,
    CallData,
    initialize_conversational_agent,
)
from pipecat.services.atoms.manager import AgentActionProcessor
from pipecat.services.atoms.prompts import FT_RESPONSE_MODEL_SYSTEM_PROMPT
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.waves.tts import WavesHttpTTSService, WavesSSETTSService
from pipecat.sync.event_notifier import EventNotifier
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def save_audio(server_name: str, audio: bytes, sample_rate: int, num_channels: int):
    if len(audio) > 0:
        filename = (
            f"{server_name}_recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"Merged audio saved to {filename}")
    else:
        logger.info("No audio data to save")


async def run_bot(websocket_client: WebSocket, stream_sid: str, call_sid: str, testing: bool):
    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
            audio_in_filter=KrispFilter(
                model_path=os.getenv("KRISP_MODEL_PATH"),
                suppression_level=90,
            ),
        ),
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    statement_llm = GoogleLLMService(
        name="StatementJudger",
        api_key=os.getenv("GOOGLE_API_KEY"),
        model=CLASSIFIER_MODEL,
        temperature=0.0,
        system_instruction=classifier_system_instruction,
    )

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        audio_passthrough=True,
    )

    # tts = CartesiaTTSService(
    #     api_key=os.getenv("CARTESIA_API_KEY"),
    #     voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    #     push_silence_after_stop=testing,
    # )

    tts = WavesHttpTTSService(
        api_key=os.getenv("WAVES_API_KEY"),
        voice_id="deepika",
    )

    # tts = WavesSSETTSService(
    #     api_key=os.getenv("WAVES_API_KEY"),
    #     voice_id="nyah",
    # )

    transport_input_filter = TransportInputFilter()
    agent_flow_processor = await initialize_conversational_agent(
        agent_id="6834620651876e6afe20256a",
        call_id="CALL-1748080011312-b252f8",
        call_data=CallData(
            variables={
                "call_id": "CALL-1748263569023-2029e3",
                "user_number": "+918815141212",
                "agent_number": "+918815141212",
            }
        ),
        transport_input_filter=transport_input_filter,
    )

    turn_tracking_observer = TurnTrackingObserver()
    agent_action_processor = AgentActionProcessor(turn_tracking_observer=turn_tracking_observer)

    messages = [
        {
            "role": "system",
            "content": FT_RESPONSE_MODEL_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps({"transcript": ""}),
        },
    ]

    context = AtomsAgentContext(messages=messages)
    context_aggregator = llm.create_context_aggregator(
        context,
        user_params=LLMUserAggregatorParams(aggregation_timeout=0.0),
        assistant_params=LLMAssistantAggregatorParams(expect_stripped_words=False),
    )
    chunks = [chunk async for chunk in (agent_flow_processor.get_response(context=context))]
    initial_agent_response = "".join(chunks)
    context.add_message(
        ChatCompletionAssistantMessageParam(role="assistant", content=initial_agent_response)
    )
    logger.info(f"Initial agent response: {initial_agent_response}")

    # NOTE: Watch out! This will save all the conversation in memory. You can
    # pass `buffer_size` to get periodic callbacks.
    audiobuffer = AudioBufferProcessor(user_continuous_stream=not testing)

    notifier = EventNotifier()
    audio_accumulator = AudioAccumulator()
    statement_judge_context_filter = StatementJudgeContextFilter(notifier=notifier)
    completeness_check = CompletenessCheck(notifier=notifier, audio_accumulator=audio_accumulator)
    bot_output_gate = OutputGate(notifier=notifier, start_open=True)

    stt_mute_filter = STTMuteFilter(
        config=STTMuteConfig(
            strategies={STTMuteStrategy.MUTE_UNTIL_FIRST_BOT_COMPLETE, STTMuteStrategy.CUSTOM},
            should_mute_callback=transport_input_filter.should_mute,
        )
    )

    async def user_idle_notifier(frame):
        await notifier.notify()

    user_idle = UserIdleProcessor(callback=user_idle_notifier, timeout=10.0)

    async def block_user_stopped_speaking(frame):
        return not isinstance(frame, UserStoppedSpeakingFrame)

    async def pass_only_llm_trigger_frames(frame: Frame):
        return (
            isinstance(frame, OpenAILLMContextFrame)
            or isinstance(frame, LLMMessagesFrame)
            or isinstance(frame, StartInterruptionFrame)
            or isinstance(frame, StopInterruptionFrame)
            or isinstance(frame, FunctionCallInProgressFrame)
            or isinstance(frame, FunctionCallResultFrame)
        )

    # pipeline = Pipeline(
    #     [
    #         transport.input(),  # Websocket input from client
    #         stt,  # Speech-To-Text
    #         context_aggregator.user(),
    #         llm,  # LLM
    #         tts,  # Text-To-Speech
    #         transport.output(),  # Websocket output to client
    #         audiobuffer,  # Used to buffer the audio in the pipeline
    #         context_aggregator.assistant(),
    #     ]
    # )
    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text
            stt_mute_filter,
            context_aggregator.user(),  # Aggregates user input into context
            ParallelPipeline(
                [
                    # Branch 1: Main flow continuation (blocks original UserStoppedSpeaking)
                    FunctionFilter(filter=block_user_stopped_speaking),
                ],
                [
                    # Branch 2: Statement LLM for end-of-turn detection
                    statement_judge_context_filter,  # Prepares input for statement_llm
                    statement_llm,  # Google LLM for "YES"/"NO"
                    completeness_check,  # Processes "YES"/"NO", may output UserStoppedSpeakingFrame
                ],
                [
                    # Branch 3: Main conversational LLM and TTS (gated)
                    FunctionFilter(
                        filter=pass_only_llm_trigger_frames
                    ),  # Ensure only relevant frames go to main LLM
                    # llm,  # Main LLM (OpenAI)
                    agent_flow_processor,
                    agent_action_processor,
                    bot_output_gate,  # Gates the output of the main LLM
                ],
            ),
            # agent_flow_processor,
            # agent_action_processor,
            # llm,
            audiobuffer,
            tts,  # Text-To-Speech (receives from gated main LLM)
            user_idle,
            transport.output(),  # Websocket output to client
            context_aggregator.assistant(),  # Aggregates assistant responses into context
        ]
    )

    task = PipelineTask(
        pipeline,
        observers=[turn_tracking_observer],
        enable_turn_tracking=False,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            allow_interruptions=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Start recording.
        await audiobuffer.start_recording()
        # Kick off the conversation.
        # await task.queue_frames([context_aggregator.user().get_context_frame()])
        await task.queue_frames([TTSSpeakFrame(text=initial_agent_response)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        server_name = f"server_{websocket_client.client.port}"
        await save_audio(server_name, audio, sample_rate, num_channels)

    # We use `handle_sigint=False` because `uvicorn` is controlling keyboard
    # interruptions. We use `force_gc=True` to force garbage collection after
    # the runner finishes running a task which could be useful for long running
    # applications with multiple clients connecting.
    runner = PipelineRunner(handle_sigint=False, force_gc=True)

    await runner.run(task)
