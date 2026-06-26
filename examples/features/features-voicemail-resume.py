#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voicemail classify-and-route that can resume into a live conversation.

This is a "roll your own" alternative to the built-in ``VoicemailDetector``. It
uses only public Pipecat APIs and one small custom ``FrameProcessor`` that you
fully own, so there is no private state to reach into and nothing that blocks the
call permanently.

How it works:

- The custom ``VoicemailClassifier`` sits after STT and before the user context
  aggregator. It buffers the very first caller turn, classifies it with its own
  lightweight LLM call, and then routes:
    - voicemail: it drops that first turn (so the main LLM never replies to a
      machine greeting), leaves a canned message, and from then on lets later
      speech flow through, so a human who picks up after the message can keep
      talking (this is the resume).
    - human: it releases the buffered turn so the main LLM answers normally.
- Everything after the first decision flows straight through, so the call is
  never permanently gated.

Because the classifier decides once per call, run the eval as a suite (a fresh
bot per scenario).
"""

import os

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import Frame, TranscriptionFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}

# The message the bot leaves when it reaches a voicemail system. It invites the
# called party to jump in, which is what the resume path handles.
VOICEMAIL_MESSAGE = (
    "Hi, this is Jamie calling about your appointment tomorrow. "
    "Please call me back at 555-0123. If you're there, feel free to jump in."
)

# A tiny, fast classifier. It only needs to answer with one word.
CLASSIFIER_PROMPT = (
    "You decide whether the FIRST thing heard on an outbound phone call is a live "
    "person or a voicemail / answering machine greeting. Voicemail greetings say "
    "things like 'you've reached', 'leave a message', 'not available', or 'after the "
    "tone'. A live person says short things like 'hello?', 'who's this?', or gives "
    'their name. Answer with exactly one word: "VOICEMAIL" or "CONVERSATION".'
)


class VoicemailClassifier(FrameProcessor):
    """Classifies the first caller turn as voicemail vs. human and routes it.

    Place this after STT and before the user context aggregator. The aggregator
    starts a turn from the final transcription, so this processor holds the first
    transcription, classifies it, then either drops it and leaves a message
    (voicemail) or releases it to the main LLM (human). After the decision every
    frame flows straight through, so a human who picks up later is heard.
    """

    def __init__(
        self, *, classifier: AsyncOpenAI, voicemail_message: str, model: str = "gpt-4o-mini"
    ):
        super().__init__()
        self._client = classifier
        self._voicemail_message = voicemail_message
        self._model = model
        self._decided = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Once the first caller turn is classified, get out of the way: every
        # frame flows through, which is what lets the call resume with a person.
        if self._decided or direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        # The user context aggregator below starts a turn from the final
        # transcription, so holding that one frame keeps the main LLM from
        # replying until we have classified the first turn.
        if isinstance(frame, TranscriptionFrame):
            await self._classify_and_route(frame, direction)
            return

        # Audio, interim transcriptions, and control frames pass through.
        await self.push_frame(frame, direction)

    async def _classify_and_route(self, frame: TranscriptionFrame, direction: FrameDirection):
        text = (frame.text or "").strip()
        decision = await self._classify(text)
        self._decided = True

        if decision == "VOICEMAIL":
            logger.info(f"Voicemail detected from {text!r}. Leaving a message...")
            # Drop the greeting (so the main LLM never replies to a machine) and
            # leave the message. Later speech flows through (the resume).
            await self.push_frame(TTSSpeakFrame(self._voicemail_message))
            logger.info("Message left. Conversation stays open for a human pickup.")
        else:
            logger.info(f"Human detected from {text!r}. Continuing the conversation.")
            # Release the held turn so the main LLM answers the person.
            await self.push_frame(frame, direction)

    async def _classify(self, text: str) -> str:
        if not text:
            return "CONVERSATION"
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": CLASSIFIER_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=4,
                temperature=0,
            )
            answer = (response.choices[0].message.content or "").upper()
            return "VOICEMAIL" if "VOICEMAIL" in answer else "CONVERSATION"
        except Exception as e:
            # If the classifier call fails, fall back to a normal conversation
            # rather than leaving a message on a live person.
            logger.warning(f"Classification failed ({e}); assuming a live conversation.")
            return "CONVERSATION"


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

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
                "You are Jamie, a friendly assistant making an outbound phone call about "
                "the person's appointment tomorrow. Your responses will be spoken aloud, so "
                "avoid emojis, bullet points, or other formatting that can't be spoken. Talk "
                "with the person naturally and briefly about their appointment."
            ),
        ),
    )

    voicemail = VoicemailClassifier(
        classifier=AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]),
        voicemail_message=VOICEMAIL_MESSAGE,
    )

    # VAD lives on the transport (above) for interruption handling. The user
    # aggregator starts each turn from the STT transcription.
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            voicemail,  # Classify-and-route: between STT and the user context aggregator
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
