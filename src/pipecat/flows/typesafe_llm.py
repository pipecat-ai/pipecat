#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A Flows LLM service that runs on TypeSafe judgments and pre-written lines.

:class:`TypeSafeFlowsLLMService` stands where the LLM normally goes in a Flows
pipeline, but it never generates text. When the flow enters a node it speaks
that node's written line. When the caller finishes a turn it asks TypeSafe's
Jev, in one request, which of the node's tools the turn calls for and which
value each tool argument takes, then runs the chosen tool exactly as an LLM
would, so Flows transitions and function-call events work unchanged. When the
turn matches no tool it speaks the node's reprompt line, and when a tool needs
an argument the caller has not given yet it asks for that argument with a
written line and remembers what the caller has said so far.

Everything it says goes out as LLM text (``LLMFullResponseStartFrame``,
``LLMTextFrame``, ``LLMFullResponseEndFrame``), so TTS, the assistant context
aggregator, RTVI clients, and text-mode evals all see the lines the way they
would see a real LLM's reply.

Requires the ``typesafe`` extra: ``uv add "pipecat-ai[typesafe]"``.
"""

import asyncio
import json
import re
import uuid
from collections.abc import Mapping
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.flows.exceptions import FlowError
from pipecat.flows.manager import FlowManager
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.typesafe_choice_router import default_state_builder
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.services.typesafe.judge import Choice, JudgeResult, TypeSafeError, TypeSafeJudge
from pipecat.utils.types import is_given

TOOL_QUESTION_ID = "tool"
"""Id of the ``Choice`` that picks the tool."""

NO_TOOL = "none"
"""The tool choice meaning the turn matches none of the node's tools."""

NOT_STATED = "not stated"
"""The argument choice meaning the caller did not give that argument."""

DEFAULT_INSTRUCTIONS = (
    "The bot said `bot_message` and the caller replied `user_reply`. The reply is a "
    "speech-to-text transcript of a phone call, so read it by sound. Which of these "
    "actions does the caller's reply ask for?"
)

DEFAULT_NO_TOOL_DESCRIPTION = (
    "None of the above: the caller says something unrelated, unclear, or only a filler"
)

_PLACEHOLDER = re.compile(
    r"(\\?)\{\{\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\}\}"
)


class NodeLines(BaseModel):
    """What the bot says in one node.

    Lines may use ``{{ key }}`` placeholders, filled from the flow manager's
    state the way Flows fills node prompts; ``{{ order.size }}`` walks into a
    stored mapping.

    Parameters:
        say: Spoken when the flow enters the node.
        reprompt: Spoken when the caller's turn matches none of the node's
            tools. Defaults to ``say``.
    """

    say: str | None = None
    reprompt: str | None = None


class ToolLines(BaseModel):
    """How one tool is judged and what the bot says around it.

    Parameters:
        description: What a caller's turn looks like when it calls for this
            tool. Defaults to the tool's schema description.
        options: The values each argument can take, keyed by argument name,
            for arguments whose schema has no ``enum``. An argument with
            neither is never filled in.
        ask: Spoken when the tool is chosen but the caller has not given that
            argument yet, keyed by argument name. Placeholders may use
            ``{{ args.<name> }}`` for arguments already given.
        result: Spoken after the tool ran and the flow stayed on the same
            node. Placeholders may use ``{{ result.<key> }}`` for the tool's
            result.
    """

    description: str | None = None
    options: dict[str, list[str | int | float]] = Field(default_factory=dict)
    ask: dict[str, str] = Field(default_factory=dict)
    result: str | None = None


class TypeSafeFlowsLLMService(LLMService):
    """An LLM service for Flows that picks tools with TypeSafe and speaks written lines.

    Give it the lines for each node and each tool, wire it into the pipeline
    where the LLM goes, pass it to the :class:`~pipecat.flows.FlowManager` as
    its ``llm``, and then set :attr:`flow_manager` so the service can read the
    current node and state.

    Each user turn costs one TypeSafe request that asks, in parallel, which
    tool the turn calls for and which value every option-bearing argument of
    every tool takes. Only the chosen tool's answers are used. The choice's
    confidence says how peaked its distribution is, not whether acting on it
    is safe; ``confidence_threshold`` turns low-confidence picks into a
    reprompt.

    Example::

        llm = TypeSafeFlowsLLMService(
            judge=TypeSafeJudge(),
            nodes={"initial": NodeLines(say="Pizza or sushi?")},
            tools={"choose_pizza": ToolLines(description="The caller wants pizza")},
        )
        flow_manager = FlowManager(worker=worker, llm=llm, ...)
        llm.flow_manager = flow_manager
    """

    def __init__(
        self,
        *,
        judge: TypeSafeJudge,
        nodes: Mapping[str, NodeLines],
        tools: Mapping[str, ToolLines] | None = None,
        instructions: str = DEFAULT_INSTRUCTIONS,
        no_tool_description: str = DEFAULT_NO_TOOL_DESCRIPTION,
        confidence_threshold: float = 0.0,
        flow_manager: FlowManager | None = None,
        warm_up: bool = True,
        **kwargs,
    ):
        """Initialize the service.

        Args:
            judge: The TypeSafe client wrapper. The service starts it, warms
                it up, and closes it.
            nodes: Lines for each node, keyed by node name.
            tools: Judging criteria and lines for each tool, keyed by tool
                name. A tool that is not listed is judged by its schema
                description and has no argument options.
            instructions: The tool question's instructions. Refer to the state
                fields `bot_message` and `user_reply`.
            no_tool_description: The criteria text of the choice that means
                the turn matches no tool.
            confidence_threshold: A tool choice below this confidence is
                treated as no match.
            flow_manager: The flow manager, when it already exists. It is
                normally set through :attr:`flow_manager` after construction.
            warm_up: Whether to open the TypeSafe connection on start.
            **kwargs: Passed to :class:`~pipecat.services.llm_service.LLMService`.
        """
        super().__init__(
            settings=LLMSettings(
                model=judge.model,
                system_instruction=None,
                temperature=None,
                max_tokens=None,
                top_p=None,
                top_k=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                filter_incomplete_user_turns=None,
                user_turn_completion_config=None,
            ),
            **kwargs,
        )
        self._judge = judge
        self._nodes = dict(nodes)
        self._tools = dict(tools or {})
        self._instructions = instructions
        self._no_tool_description = no_tool_description
        self._confidence_threshold = confidence_threshold
        self._flow_manager = flow_manager
        self._warm_up = warm_up
        self._task: asyncio.Task | None = None
        self._warm_up_task: asyncio.Task | None = None
        self._last_said: str | None = None
        # Arguments the caller has given so far for a tool, kept until the
        # tool runs or the flow leaves the node: (node, tool) -> {arg: value}.
        self._remembered: dict[tuple[str | None, str], dict[str, Any]] = {}

    @property
    def flow_manager(self) -> FlowManager | None:
        """The flow manager whose node and state the lines come from."""
        return self._flow_manager

    @flow_manager.setter
    def flow_manager(self, flow_manager: FlowManager) -> None:
        self._flow_manager = flow_manager

    def can_generate_metrics(self) -> bool:
        """Whether this service reports processing and usage metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Start the service and open the TypeSafe connection."""
        await super().start(frame)
        self._judge.start()
        if self._warm_up:
            self._warm_up_task = self.create_task(self._judge.warm_up())

    async def stop(self, frame: EndFrame):
        """Stop the service and close the TypeSafe connection."""
        await super().stop(frame)
        await self._cancel_turn()
        await self._judge.close()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close the TypeSafe connection."""
        await super().cancel(frame)
        await self._cancel_turn()
        await self._judge.close()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Answer context frames; pass everything else through."""
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self._cancel_turn()
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMContextFrame):
            await self._cancel_turn()
            self._task = self.create_task(self._respond(frame))
        else:
            await self.push_frame(frame, direction)

    async def _cancel_turn(self) -> None:
        if self._task is not None:
            task, self._task = self._task, None
            await self.cancel_task(task)
            # A judgment that never arrived has no first byte or processing
            # time to report, and must not be measured against the next turn.
            await self.cancel_ttfb_metrics()
            await self.stop_processing_metrics()

    # What to do with a context frame

    async def _respond(self, frame: LLMContextFrame) -> None:
        context = frame.context
        last = _last_message(context)
        role = last.get("role") if last is not None else None
        try:
            # Processing time covers the whole response: rendering a line,
            # or judging the turn and starting the tool it chose.
            await self.start_processing_metrics()
            if last is not None and role == "user":
                await self._respond_to_user(context, last)
            elif last is not None and role == "tool":
                await self._respond_to_result(context, last)
            else:
                await self._enter_node()
            await self.stop_processing_metrics()
        except FlowError as e:
            await self.stop_processing_metrics()
            await self.push_error(str(e), e)
        finally:
            self._task = None

    async def _enter_node(self) -> None:
        node = self._current_node()
        lines = self._nodes.get(node or "")
        if lines is None or lines.say is None:
            logger.warning(f"{self}: node '{node}' has no line to say on entry")
            return
        # Whatever the caller had given for tools on an earlier node no longer applies.
        self._remembered = {key: value for key, value in self._remembered.items() if key[0] == node}
        await self._say(self._render(lines.say))

    async def _respond_to_result(self, context: LLMContext, message: Mapping[str, Any]) -> None:
        tool = _function_for_result(context, message.get("tool_call_id"))
        lines = self._tools.get(tool or "")
        if lines is None or lines.result is None:
            logger.warning(f"{self}: tool '{tool}' has no line to say after it ran")
            return
        await self._say(self._render(lines.result, result=_parse_result(message.get("content"))))

    async def _respond_to_user(self, context: LLMContext, message: Mapping[str, Any]) -> None:
        node = self._current_node()
        schemas = _tool_schemas(context)
        if not schemas:
            await self._reprompt(node)
            return

        state = default_state_builder(context)
        # The context holds the bot's line only as far as TTS had spoken it
        # when the caller replied; the service knows the whole line.
        if self._last_said:
            state["bot_message"] = self._last_said
        questions = self._questions(schemas)
        try:
            # TTFB runs from the request to the judgment's arrival, which is
            # the whole response. TTFAT then runs on to the first answer
            # token: the spoken line or the tool call the judgment leads to.
            await self.start_ttfb_metrics()
            result = await self._judge.ask(state, questions)
            await self.stop_ttfb_metrics()
        except (TypeSafeError, TimeoutError, OSError) as e:
            await self.cancel_ttfb_metrics()
            await self.push_error(f"TypeSafe judgment failed: {e}", e)
            await self._reprompt(node)
            return

        await self._report_usage(result)
        tool_choice = result.choices[TOOL_QUESTION_ID]
        logger.debug(
            f"{self}: judged {state} in {result.latency_secs * 1000:.0f}ms: "
            f"{tool_choice.probabilities}"
        )
        if tool_choice.choice == NO_TOOL or tool_choice.confidence < self._confidence_threshold:
            await self._reprompt(node)
            return

        tool = tool_choice.choice
        schema = next(s for s in schemas if s.name == tool)
        lines = self._tools.get(tool, ToolLines())
        remembered = self._remembered.setdefault((node, tool), {})
        for name in schema.properties:
            answer = result.choices.get(_argument_question_id(tool, name))
            if answer is not None and answer.choice != NOT_STATED:
                remembered[name] = _coerce(answer.choice, schema.properties[name])

        missing = [name for name in schema.required if name not in remembered]
        if missing:
            line = lines.ask.get(missing[0])
            if line is None:
                logger.warning(f"{self}: tool '{tool}' has no line asking for '{missing[0]}'")
                await self._reprompt(node)
                return
            await self._say(self._render(line, args=dict(remembered)))
            return

        arguments = dict(remembered)
        self._remembered.pop((node, tool), None)
        # A turn answered with a tool call instead of text.
        await self.stop_ttfat_metrics()
        await self.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name=tool,
                    tool_call_id=f"typesafe-{uuid.uuid4().hex[:12]}",
                    arguments=arguments,
                    context=context,
                )
            ]
        )

    # Questions

    def _questions(self, schemas: list[FunctionSchema]) -> dict[str, Choice]:
        criteria: dict[str, str] = {}
        questions: dict[str, Choice] = {}
        for schema in schemas:
            lines = self._tools.get(schema.name, ToolLines())
            criteria[schema.name] = lines.description or schema.description
            for name, prop in schema.properties.items():
                options = prop.get("enum") or lines.options.get(name)
                if not options:
                    continue
                about = prop.get("description") or f"the {name}"
                questions[_argument_question_id(schema.name, name)] = Choice(
                    instructions=(
                        f"In `user_reply`, which value does the caller give for this: {about} "
                        f"Pick '{NOT_STATED}' when the reply does not give one."
                    ),
                    criteria={
                        **{str(option): f"The caller says {option}" for option in options},
                        NOT_STATED: "The reply does not give this value",
                    },
                )
        criteria[NO_TOOL] = self._no_tool_description
        questions[TOOL_QUESTION_ID] = Choice(instructions=self._instructions, criteria=criteria)
        return questions

    # Speaking

    async def _reprompt(self, node: str | None) -> None:
        lines = self._nodes.get(node or "")
        line = (lines.reprompt or lines.say) if lines else None
        if line is None:
            logger.warning(f"{self}: node '{node}' has no line to reprompt with")
            return
        await self._say(self._render(line))

    async def _say(self, text: str) -> None:
        self._last_said = text
        await self.push_frame(LLMFullResponseStartFrame())
        # Records TTFAT for the turn, when a judgment preceded the line.
        await self._push_llm_text(text)
        await self.push_frame(LLMFullResponseEndFrame())

    def _render(self, text: str, **extra: Any) -> str:
        values: dict[str, Any] = dict(self._flow_manager.state) if self._flow_manager else {}
        values.update(extra)

        def value(path: str) -> str:
            current: Any = values
            for part in path.split("."):
                if not isinstance(current, Mapping) or part not in current:
                    raise FlowError(f"line uses '{{{{ {path} }}}}', which is not in state")
                current = current[part]
            return str(current)

        def substitute(match: re.Match) -> str:
            if match.group(1):
                return match.group(0)[1:]
            return value(match.group(2))

        return _PLACEHOLDER.sub(substitute, text)

    def _current_node(self) -> str | None:
        return self._flow_manager.current_node if self._flow_manager else None

    async def _report_usage(self, result: JudgeResult) -> None:
        if result.input_tokens is None and result.output_tokens is None:
            return
        prompt = result.input_tokens or 0
        completion = result.output_tokens or 0
        await self.start_llm_usage_metrics(
            LLMTokenUsage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
            )
        )


def _argument_question_id(tool: str, argument: str) -> str:
    return f"{tool}.{argument}"


def _last_message(context: LLMContext) -> Mapping[str, Any] | None:
    for message in reversed(context.messages):
        if isinstance(message, Mapping):
            return message
    return None


def _function_for_result(context: LLMContext, tool_call_id: Any) -> str | None:
    for message in reversed(context.messages):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if call.get("id") == tool_call_id:
                return call.get("function", {}).get("name")
    return None


def _parse_result(content: Any) -> Any:
    if isinstance(content, str):
        try:
            return json.loads(content)
        except ValueError:
            return content
    return content


def _tool_schemas(context: LLMContext) -> list[FunctionSchema]:
    tools = context.tools
    if not is_given(tools) or not isinstance(tools, ToolsSchema):
        return []
    return list(tools.standard_tools)


def _coerce(value: str, prop: Mapping[str, Any]) -> Any:
    kind = prop.get("type")
    try:
        if kind == "integer":
            return int(value)
        if kind == "number":
            return float(value)
        if kind == "boolean":
            return value.lower() in ("true", "yes")
    except ValueError:
        pass
    return value
