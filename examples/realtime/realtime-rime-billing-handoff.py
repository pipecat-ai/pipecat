#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Example: hand a Rime Prism conversation off to billing support mid-call.

A general support assistant can transfer the caller to billing. The transfer
tool updates the Prism session's instructions, tools, and tool fillers with an
:class:`~pipecat.frames.frames.LLMUpdateSettingsFrame`, so the same conversation
continues with the billing configuration. A Prism tool filler speaks while the
invoice lookup runs.
"""

import os

from dotenv import load_dotenv
from loguru import logger
from rimelabs_sdk.realtime import PrismFiller

from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame, LLMUpdateSettingsFrame
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.rime.realtime.llm import RimeRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


async def lookup_invoice(params: FunctionCallParams, invoice_id: str):
    """Read an invoice's payment status.

    Args:
        invoice_id: Invoice identifier.
    """
    await params.result_callback({"invoice_id": invoice_id, "status": "paid", "total": 42})


async def transfer_to_billing(params: FunctionCallParams):
    """Transfer the caller to billing support."""
    await params.pipeline_worker.queue_frame(
        LLMUpdateSettingsFrame(
            delta=RimeRealtimeLLMService.Settings(
                system_instruction=(
                    "You handle billing. Begin each answer with 'Billing support'. "
                    "Keep answers short. Use lookup_invoice for invoice status."
                ),
                tools=[lookup_invoice],
                tool_fillers={
                    "lookup_invoice": PrismFiller(type="text", text="Checking your invoice.")
                },
            )
        )
    )
    await params.result_callback({"department": "billing"})


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


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    llm = RimeRealtimeLLMService(
        endpoint=os.getenv("RIME_REALTIME_ENDPOINT"),
        settings=RimeRealtimeLLMService.Settings(
            system_instruction=(
                "You are a voice support assistant. Keep answers short. "
                "Transfer the caller to billing for any question about invoices or payments."
            ),
            tools=[transfer_to_billing],
        ),
    )

    context = LLMContext([{"role": "developer", "content": "Greet the caller and offer help."}])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
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
        observers=[TranscriptionLogObserver()],
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
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
