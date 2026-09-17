#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Turn-completion eval bot: one LLM service behind the marker protocol, text in and out.

The suite spawns this bot once per model and scenario. The runner body (the
manifest entry's ``runner_body.data``) names the service class, its
constructor arguments, the model and its settings; the bot builds the service,
turns on ``filter_incomplete_user_turns`` through
``FilterIncompleteUserTurnStrategies``, and advertises the two tools the tool
scenarios expect. There is no STT or TTS: the harness sends text turns and
reads the LLM's text and marker events back.

Body fields, matching a ``models.yaml`` entry of the old in-process runner::

    service: pipecat.services.openai.llm.OpenAILLMService   # dotted class path
    kwargs: {api_key: $OPENAI_API_KEY}   # constructor arguments; $NAME reads the environment
    model: gpt-4.1
    settings: {extra: {reasoning_effort: none}}   # extra Settings(...) fields
    system_suffix: "/no_think"           # appended to the system instruction
    developer_role: false                # send developer messages as user
    tools: false                         # advertise no tools (an endpoint that rejects them)

``TURN_COMPLETION_PROMPT=<name>`` in the environment swaps the framework's
default instructions for ``prompts/<name>.txt``.
"""

import importlib
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.evals.transport import EvalTransportParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport
from pipecat.turns.user_turn_completion_mixin import (
    USER_TURN_COMPLETE_MARKER,
    USER_TURN_INCOMPLETE_LONG_MARKER,
    USER_TURN_INCOMPLETE_SHORT_MARKER,
    UserTurnCompletionConfig,
)
from pipecat.turns.user_turn_strategies import FilterIncompleteUserTurnStrategies
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

HERE = Path(__file__).resolve().parent

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant in a voice conversation. Your responses will be spoken "
    "aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. "
    "Respond to what the user said in a creative, helpful, and brief way."
)

transport_params = {
    "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def _tool_handler(params: FunctionCallParams):
    await params.result_callback({"ok": True, "note": "stubbed for the eval"})


TOOLS = ToolsSchema(
    standard_tools=[
        FunctionSchema(
            name="get_current_weather",
            description="Get the current weather for a location.",
            properties={
                "location": {"type": "string", "description": "City and state or country."}
            },
            required=["location"],
            handler=_tool_handler,
        ),
        FunctionSchema(
            name="book_appointment",
            description="Book an appointment on a given day and time.",
            properties={
                "day": {"type": "string", "description": "Day of the appointment."},
                "time": {"type": "string", "description": "Time of the appointment."},
            },
            required=["day", "time"],
            handler=_tool_handler,
        ),
    ]
)


def _env_subst(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        return os.environ[value[1:]]
    if isinstance(value, dict):
        return {k: _env_subst(v) for k, v in value.items()}
    return value


def _convert_nested_settings(cls: type, settings: dict[str, Any]) -> dict[str, Any]:
    """Turn the mapping forms of provider config objects into their classes.

    Google's and Anthropic's Settings accept a mapping for ``thinking``; the
    OpenAI Responses ``reasoning`` and DeepSeek ``thinking`` fields do not.
    """
    out = dict(settings)
    if isinstance(out.get("reasoning"), dict):
        from pipecat.services.openai.responses.llm import OpenAIResponsesReasoningConfig

        out["reasoning"] = OpenAIResponsesReasoningConfig(**out["reasoning"])
    if isinstance(out.get("thinking"), dict) and "DeepSeek" in cls.__name__:
        from pipecat.services.deepseek.llm import DeepSeekThinkingConfig

        out["thinking"] = DeepSeekThinkingConfig(**out["thinking"])
    return out


def build_llm(body: dict[str, Any]):
    """The LLM service the body describes, with the eval's system instruction."""
    module_name, _, attr = str(body["service"]).rpartition(".")
    cls = getattr(importlib.import_module(module_name), attr)
    system_instruction = SYSTEM_INSTRUCTION
    if body.get("system_suffix"):
        system_instruction = f"{system_instruction}\n\n{body['system_suffix']}"
    fields = _convert_nested_settings(cls, dict(body.get("settings") or {}))
    # `extra` is the pass-through bag of provider request parameters; keep it
    # out of from_mapping, which would nest it as an unknown field.
    extra = fields.pop("extra", {})
    settings = cls.Settings.from_mapping(
        {"model": _env_subst(body["model"]), "system_instruction": system_instruction, **fields}
    )
    if extra:
        settings.extra = {**(settings.extra or {}), **extra}
    llm = cls(settings=settings, **_env_subst(body.get("kwargs") or {}))
    if body.get("developer_role") is False:
        # A model behind a generic OpenAI-compatible service whose chat template
        # rejects the developer role; the adapter then sends those as user.
        llm.supports_developer_role = False
    return llm


def completion_config() -> UserTurnCompletionConfig:
    """The turn-completion config: the framework's default instructions, or a prompt variant."""
    variant = os.environ.get("TURN_COMPLETION_PROMPT")
    if not variant:
        return UserTurnCompletionConfig()
    template = (HERE / "prompts" / f"{variant}.txt").read_text()
    return UserTurnCompletionConfig(
        instructions=template.format(
            complete=USER_TURN_COMPLETE_MARKER,
            short=USER_TURN_INCOMPLETE_SHORT_MARKER,
            long=USER_TURN_INCOMPLETE_LONG_MARKER,
        )
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    body = runner_args.body
    if not body or "service" not in body:
        raise ValueError(
            "this bot needs a runner body naming the service and model (see module doc)"
        )
    logger.info(f"Starting turn-completion bot: {body.get('service')} {body.get('model')}")

    llm = build_llm(body)
    context = LLMContext(tools=TOOLS) if body.get("tools", True) else LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=FilterIncompleteUserTurnStrategies(config=completion_config()),
        ),
    )

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
        params=PipelineParams(enable_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
