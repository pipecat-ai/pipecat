#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Prism voice, typed turns, tools, and a billing handoff. Requires RIME_API_KEY."""

import asyncio
import os

from dotenv import load_dotenv
from rimelabs_sdk.realtime import PrismFiller

from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame, LLMUpdateSettingsFrame
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.rime.realtime.llm import RimeRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

transport_params = {
    "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    async def lookup_invoice(params: FunctionCallParams, invoice_id: str):
        """Read an invoice's payment status.

        Args:
            invoice_id: Invoice identifier.
        """
        await asyncio.sleep(0.25)
        await params.result_callback({"invoice_id": invoice_id, "status": "paid", "total": 42})

    async def transfer_to_billing(params: FunctionCallParams):
        """Change the conversation to billing support."""
        await params.pipeline_worker.queue_frame(
            LLMUpdateSettingsFrame(
                delta=RimeRealtimeLLMService.Settings(
                    system_instruction="You handle billing. Begin each answer with 'Billing support'. Keep answers short.",
                    tools=[lookup_invoice],
                    tool_fillers={
                        "lookup_invoice": PrismFiller(type="text", text="Checking your invoice.")
                    },
                )
            )
        )
        await params.result_callback({"department": "billing"})

    llm = RimeRealtimeLLMService(
        endpoint=os.getenv("RIME_REALTIME_ENDPOINT"),
        settings=RimeRealtimeLLMService.Settings(
            system_instruction="You are a voice support assistant. Keep answers short. Use tools for invoice status and billing transfers.",
            tools=[lookup_invoice, transfer_to_billing],
            tool_fillers={
                "lookup_invoice": PrismFiller(type="text", text="Checking your invoice.")
            },
        ),
    )
    context = LLMContext([{"role": "system", "content": "Greet the caller and offer help."}])
    user, assistant = llm.create_context_aggregator(context)
    pipeline = Pipeline([transport.input(), user, llm, transport.output(), assistant])
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[TranscriptionLogObserver()],
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
