#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice agent that places an outbound SIP call.

The bot dials one destination as soon as its transport is ready — a softphone
on the same Daily domain, or a real phone through a Twilio trunk (see
``../util/provision.py``) — and runs the conversation when the far end answers.
Inbound INVITEs are left ringing.

Copy ``env.example`` to ``.env`` in ``examples/sip``, then dial from the
environment::

    SIP_DIALOUT_URI=sip:pipecat-harness-dev@example.sip.daily.co \\
        uv run python examples/sip/dial-out/bot.py -t sip -d

or from a runner body file, which ``provision.py call dial-out`` writes::

    uv run python examples/sip/dial-out/bot.py -t sip -d \\
        --runner-body examples/sip/dial-out/dialout-body.json

Progress through a call shows as numbered INFO lines, ``[1/6]`` to
``[6/6]`` — see ``../util/milestones.py`` for what each one means and
``../README.md`` for the topology they trace.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, LLMRunFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.sip.transport import SIPParams
from pipecat.workers.runner import WorkerRunner

SIP_EXAMPLES_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SIP_EXAMPLES_DIR / "util"))

from milestones import attach_milestones  # noqa: E402

load_dotenv(SIP_EXAMPLES_DIR / ".env", override=True)

STATE_PATH = SIP_EXAMPLES_DIR / ".state.json"
PROVISIONED_BOT_USER = "pipecat-harness-bot"

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant on a phone call. Your responses will be spoken "
    "aloud, so avoid emojis, bullet points, or other formatting that can't be "
    "spoken. Respond to what the user said in a creative, helpful, and brief way."
)


def get_dialout_settings(runner_args: RunnerArguments) -> dict | None:
    """The dial-out destination from the runner body, or ``SIP_DIALOUT_URI``.

    Args:
        runner_args: Arguments the runner started this session with.

    Returns:
        Dial-out settings to pass to ``start_dialout``, or ``None`` when
        neither source names a destination.
    """
    body = runner_args.body or {}
    if isinstance(body, dict) and body.get("dialout_settings"):
        return body["dialout_settings"]
    uri = os.getenv("SIP_DIALOUT_URI")
    return {"sipUri": uri} if uri else None


def use_provisioned_account() -> None:
    """Dial as the provisioned bot client by exporting its credentials.

    Fills ``SIP_USER``/``SIP_PASS``/``SIP_DOMAIN`` from ``.state.json``, the
    account ``provision.py`` created. Runs at import, since the runner
    resolves the account before it calls :func:`bot`. An account the
    environment already names stays as it is — ``provision.py call dial-out``
    uses that to dial through a Twilio trunk's credential list instead.
    """
    if os.getenv("SIP_USER") or not STATE_PATH.exists():
        return
    try:
        state = json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError as e:
        logger.warning(f"Ignoring {STATE_PATH.name}: not valid JSON ({e})")
        return

    client = state.get("daily", {}).get("sip_clients", {}).get(PROVISIONED_BOT_USER, {})
    if not (client.get("domain") and client.get("password")):
        return

    os.environ["SIP_USER"] = client["username"]
    os.environ["SIP_PASS"] = client["password"]
    os.environ["SIP_DOMAIN"] = client["domain"]
    # Daily's registrar takes TLS, the transport the runner picks for a client
    # it provisions itself.
    os.environ.setdefault("SIP_TRANSPORT", "tls")
    logger.info(f"Using provisioned account: sip:{client['username']}@{client['domain']}")


use_provisioned_account()


class HeardLogger(FrameProcessor):
    """Logs every final transcription at INFO as ``heard: "..."``.

    Shows in the bot terminal exactly what speech reached the LLM. The bot's
    own words appearing here about a second after it spoke them means the far
    end is echoing its audio back — see README.md troubleshooting.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Pass frame through, logging final transcriptions.

        Args:
            frame: Frame to process.
            direction: Direction the frame flows.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            logger.info(f'heard: "{frame.text}"')
        await self.push_frame(frame, direction)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments, dialout_settings: dict):
    """Run the conversation pipeline over an outbound SIP call.

    Args:
        transport: Transport the runner built.
        runner_args: Arguments the runner started this session with.
        dialout_settings: Destination to dial.
    """
    logger.info("Starting bot (dial-out)")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="86e30c1d-714b-4074-a1f2-1cb6b552fb49",
        ),
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(system_instruction=SYSTEM_INSTRUCTION),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            HeardLogger(),  # INFO line per final transcription
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
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    attach_milestones(transport, direction="dialout")

    @transport.event_handler("on_connected")
    async def on_connected(transport, data=None):
        logger.info(f"Dialing {dialout_settings}")
        _, error = await transport.start_dialout(dialout_settings)
        if error:
            logger.error(f"Dial-out failed: {error}")
            if "Address family" in str(error):
                logger.info(
                    "Hint: medianat=stun cannot run on a registration-less account. "
                    "Rerun with SIP_STUN_SERVER=off (provision.py call dial-out "
                    "prints it) and turn on the trunk's Symmetric RTP — see "
                    "README.md troubleshooting."
                )
            await runner.cancel()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("[4/6] pipeline started — bot is listening")
        # Kick off the conversation.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await runner.cancel()

    async def on_call_stopped(transport, data=None):
        await runner.cancel()

    transport.add_event_handler("on_dialout_stopped", on_call_stopped)
    transport.add_event_handler("on_error", on_call_stopped)

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud.

    Args:
        runner_args: Arguments the runner started this session with.
    """
    dialout_settings = get_dialout_settings(runner_args)
    if not dialout_settings:
        logger.error(
            "No dial-out destination. Set SIP_DIALOUT_URI, or pass "
            "--runner-body examples/sip/dial-out/dialout-body.json "
            "(provision.py call dial-out writes it)."
        )
        return

    transport_params = {
        "sip": lambda: SIPParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_sample_rate=8000,
            auto_answer=False,
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args, dialout_settings)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
