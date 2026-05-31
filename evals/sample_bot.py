"""Sample bot for the eval harness.

This is a minimal passthrough bot — input frames go straight to output, no LLM,
no TTS. It's enough to exercise the user-side evals
(``01_basic_user_input.yaml``, ``02_judge_user_text.yaml``, ``03_send_after.yaml``).

For evals that assert on bot-side events (``04_bot_greeting.yaml``,
``05_tool_call.yaml``, ``06_interruption.yaml``) you'll need to plug in real
LLM and TTS services — uncomment the example block below and configure it for
your provider.

Run with::

    python evals/sample_bot.py -t eval

Then in another terminal::

    python -m pipecat.evals evals/01_basic_user_input.yaml
"""

from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.eval.transport import EvalTransportParams
from pipecat.workers.runner import WorkerRunner

# Lambdas defer transport parameter creation until the transport type is
# selected at runtime — matches the pattern used by other pipecat examples.
transport_params = {
    "eval": lambda: EvalTransportParams(),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting sample eval bot (passthrough pipeline)")

    # Passthrough pipeline — input frames flow straight to output.
    # For evals that need bot-side events, replace with something like:
    #
    #     from pipecat.processors.aggregators.llm_context import LLMContextAggregatorPair
    #     from pipecat.services.openai.llm import OpenAILLMService
    #     from pipecat.services.openai.tts import OpenAITTSService
    #
    #     llm = OpenAILLMService(...)
    #     tts = OpenAITTSService(...)
    #     ctx_aggr = LLMContextAggregatorPair(LLMContext([{"role": "system", "content": "..."}]))
    #     pipeline = Pipeline([
    #         transport.input(),
    #         ctx_aggr.user(),
    #         llm,
    #         tts,
    #         transport.output(),
    #         ctx_aggr.assistant(),
    #     ])
    pipeline = Pipeline([transport.input(), transport.output()])

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(),
        # Generous timeout so the bot stays up while you iterate on evals.
        idle_timeout_secs=3600,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Eval harness connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Eval harness disconnected (bot stays up for next eval)")

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point — called by the dev runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
