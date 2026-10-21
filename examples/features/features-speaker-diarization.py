#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A bot that tells apart several people talking on one audio stream.

The STT service's speaker diarization labels every transcript with the speaker
who said it: each frame carries the provider's label in ``speaker_label``, and
``speaker_format`` writes it into the text, so the LLM sees who said what. The
bot listens until the conversation pauses, then says how many people spoke and
what each of them said.

Pick the STT service with the runner body, e.g. ``{"stt": "soniox"}``; it
defaults to Deepgram. Streaming services keep a speaker's label for the whole
session, which is what this bot needs. Segmented (batch) services diarize each
speech segment on its own, so their labels don't carry across segments.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame, TranscriptionFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.stt_service import STTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# Written into each transcript so the LLM sees which speaker said what.
# ``{speaker_number}`` numbers speakers in the order they first spoke, so the
# tags are the same whichever provider's labels ("0", "A", "S1", ...) are behind them.
SPEAKER_FORMAT = "[Speaker {speaker_number}] {text}"


transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "livekit": lambda: LiveKitParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


def create_stt(name: str) -> STTService:
    """Create a streaming STT service with speaker diarization enabled."""
    if name == "deepgram":
        from pipecat.services.deepgram.stt import DeepgramSTTService

        return DeepgramSTTService(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            settings=DeepgramSTTService.Settings(diarize=True, speaker_format=SPEAKER_FORMAT),
        )
    if name == "assemblyai":
        from pipecat.services.assemblyai.stt import AssemblyAISTTService

        return AssemblyAISTTService(
            api_key=os.environ["ASSEMBLYAI_API_KEY"],
            settings=AssemblyAISTTService.Settings(
                speaker_labels=True, speaker_format=SPEAKER_FORMAT
            ),
        )
    if name == "aws":
        from pipecat.services.aws.stt import AWSTranscribeSTTService

        return AWSTranscribeSTTService(
            settings=AWSTranscribeSTTService.Settings(
                show_speaker_label=True, speaker_format=SPEAKER_FORMAT
            ),
        )
    if name == "nvidia":
        from pipecat.services.nvidia.stt import NvidiaSTTService

        return NvidiaSTTService(
            api_key=os.environ["NVIDIA_API_KEY"],
            settings=NvidiaSTTService.Settings(
                speaker_diarization=True, speaker_format=SPEAKER_FORMAT
            ),
        )
    if name == "smallest":
        from pipecat.services.smallest.stt import SmallestSTTService

        return SmallestSTTService(
            api_key=os.environ["SMALLEST_API_KEY"],
            settings=SmallestSTTService.Settings(diarize=True, speaker_format=SPEAKER_FORMAT),
        )
    if name == "soniox":
        from pipecat.services.soniox.stt import SonioxSTTService

        return SonioxSTTService(
            api_key=os.environ["SONIOX_API_KEY"],
            settings=SonioxSTTService.Settings(
                enable_speaker_diarization=True, speaker_format=SPEAKER_FORMAT
            ),
        )
    if name == "speechmatics":
        from pipecat.services.speechmatics.stt import SpeechmaticsSTTService

        return SpeechmaticsSTTService(
            api_key=os.environ["SPEECHMATICS_API_KEY"],
            settings=SpeechmaticsSTTService.Settings(
                enable_diarization=True, speaker_format=SPEAKER_FORMAT
            ),
        )
    if name == "xai":
        from pipecat.services.xai.stt import XAISTTService

        return XAISTTService(
            api_key=os.environ["XAI_API_KEY"],
            settings=XAISTTService.Settings(diarize=True, speaker_format=SPEAKER_FORMAT),
        )
    raise ValueError(f"Unknown STT service: {name}")


class SpeakerLogger(BaseObserver):
    """Log every final transcript with the speaker the STT service assigned it."""

    async def on_push_frame(self, data: FramePushed):
        if isinstance(data.frame, TranscriptionFrame) and isinstance(data.source, STTService):
            logger.info(f"Speaker {data.frame.speaker_label}: {data.frame.text}")


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    body = runner_args.body or {}
    stt_name = body.get("stt", "deepgram")
    logger.info(f"Starting bot with {stt_name} STT")

    stt = create_stt(stt_name)

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You listen to conversations between several people and report on them. "
                "The user's messages are transcripts in which each part starts with its "
                "speaker's tag in square brackets; parts with the same tag were said by the "
                "same person. Diarization sometimes gives a word or two the wrong tag, so "
                "don't count a tag that only appears on a word or two as another person. "
                "When a conversation comes in, say how many people are talking, then, in one "
                "short sentence per person, everything that person said, calling them the "
                "first speaker, the second speaker and so on. Never read the tags aloud. "
                "Your responses will be spoken aloud, so avoid emojis, bullet points, or "
                "other formatting that can't be spoken."
            ),
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            # Wait for a real pause before replying, so the short gaps between
            # one speaker and the next don't end the user's turn.
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=2.0)],
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT, with speaker diarization
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[SpeakerLogger()],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation.
        context.add_message(
            {
                "role": "developer",
                "content": "Briefly introduce yourself and say you are ready to listen.",
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
