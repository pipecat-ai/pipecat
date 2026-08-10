#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A voice client for a coding agent, over the Agent Client Protocol.

Speech becomes an ACP prompt, the agent's stream becomes ACP frames, and
``ACPLogObserver`` logs those frames without speaking. The bot stays silent by
design: deciding what an agent's tool calls and reasoning should sound like is
the rendering problem, and it belongs in a processor between ``ACPService`` and
the TTS service.

Set ``ACP_AGENT_COMMAND`` to the agent to run and ``ACP_AGENT_CWD`` to the
directory it should work in. Permission requests are auto-approved, so point it
at a scratch repository.

Run it::

    export ACP_AGENT_COMMAND="npx @zed-industries/claude-code-acp"
    export ACP_AGENT_CWD=/path/to/scratch/repo
    python examples/acp/acp-voice-client.py
"""

import os
import shlex

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.observers.loggers.acp_log_observer import ACPLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.acp.aggregator import ACPUserAggregator
from pipecat.services.acp.permissions import ACPAutoPermission
from pipecat.services.acp.service import ACPService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

transport_params = {
    "daily": lambda: DailyParams(
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


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    command = shlex.split(os.environ["ACP_AGENT_COMMAND"])
    cwd = os.environ["ACP_AGENT_CWD"]
    logger.info(f"Running ACP agent {command} in {cwd}")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    acp = ACPService(command=command, cwd=cwd)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            ACPUserAggregator(),  # Speech -> ACPPromptFrame
            ACPAutoPermission(),  # Answers the agent's permission requests
            acp,  # The agent
            # A renderer goes here: ACP frames in, speakable text out.
            transport.output(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        observers=[ACPLogObserver()],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @acp.event_handler("on_agent_exited")
    async def on_agent_exited(service, returncode):
        logger.error(f"ACP agent exited with code {returncode}")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
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
