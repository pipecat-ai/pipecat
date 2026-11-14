#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A bot for the text flow capture tests: a real pipeline that is never run."""

from collections.abc import AsyncGenerator

from pipecat.frames.frames import Frame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.frameworks.rtvi.observer import RTVIObserverParams
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.utils.text.pattern_pair_aggregator import MatchAction, PatternPairAggregator
from pipecat.utils.text.transforms import replace_text
from pipecat.workers.runner import WorkerRunner

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


class SilentTTS(TTSService):
    """A TTS service with no provider behind it: every request gets a short silence."""

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        yield TTSStartedFrame()
        yield TTSAudioRawFrame(
            audio=b"\x00\x00" * (self.sample_rate // 10),
            sample_rate=self.sample_rate,
            num_channels=1,
            context_id=context_id,
        )
        yield TTSStoppedFrame()


class PassThrough(FrameProcessor):
    """A processor between the LLM and the TTS that the trace does not run."""


async def shout(text: str, text_type) -> str:
    return text.upper()


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    llm = OpenAILLMService(api_key="test")

    aggregator = PatternPairAggregator()
    aggregator.add_pattern(
        type="code", start_pattern="<code>", end_pattern="</code>", action=MatchAction.AGGREGATE
    )
    llm_text_processor = LLMTextProcessor(
        text_aggregator=aggregator,
        text_transforms=[("*", replace_text([(r"\bDr\.", "Doctor")]))],
    )

    tts = SilentTTS(skip_text_types=["code"], text_transforms=[("*", shout)])

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
            llm_text_processor,
            PassThrough(),
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        rtvi_observer_params=RTVIObserverParams(skip_text_types=["code"]),
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        pass

    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)
