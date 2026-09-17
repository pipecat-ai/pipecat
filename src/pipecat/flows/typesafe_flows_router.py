#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Drive a Flows conversation with TypeSafe, speaking a written line only when sure.

:class:`TypeSafeFlowsRouter` sits between the user context aggregator and a
text LLM in a Flows pipeline. At the end of each caller turn it asks
TypeSafe's Jev, in one request, which of the node's tools the turn calls
for, which value each argument takes, and whether the caller said anything
more than what the bot asked for. Then it picks one of three tiers:

- **Canned.** The tool, every argument value, and the "nothing more" answer
  are all at or above ``canned_threshold``. The router runs the tool and
  speaks the next node's written line itself, so the LLM never runs. A tool
  that still needs an argument gets its written ``ask`` line instead, when
  the caller plainly did not give that argument.
- **Routed.** The tool reaches ``routing_threshold`` but something is less
  sure: a value, or the caller also asked a question. The router runs the
  tool when its arguments are complete, and the LLM speaks the next node
  from its prompt with the tool result in context. When an argument is
  missing, the LLM asks for it in its own words.
- **Open.** No tool reaches the routing threshold, or the request failed.
  The turn passes to the LLM untouched, tools included, so the LLM can
  answer, ask, or call a tool itself.

Everything the router says goes out as LLM text, so TTS, the assistant
context aggregator, RTVI clients and text-mode evals see the lines the way
they see the LLM's replies. The context frame a tool result pushes back to
the LLM is not seen by the router, so tool results are always phrased by the
LLM; ``ToolLines.result`` is not used here.

Requires the ``typesafe`` extra: ``uv add "pipecat-ai[typesafe]"``.
"""

import asyncio
import uuid
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from loguru import logger

from pipecat.flows.exceptions import FlowError
from pipecat.flows.manager import FlowManager
from pipecat.flows.typesafe_llm import (
    DEFAULT_INSTRUCTIONS,
    DEFAULT_NO_TOOL_DESCRIPTION,
    NO_TOOL,
    TOOL_QUESTION_ID,
    NodeLines,
    ToolLines,
    argument_question_id,
    build_questions,
    given_arguments,
    last_message,
    render_line,
    stated_question_id,
    tool_schemas,
)
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    UserStartedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, MetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.typesafe_choice_router import default_state_builder
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.typesafe.judge import JudgeResult, Noul, NoulCriteria, TypeSafeJudge

MORE_QUESTION_ID = "more"
"""Id of the ``Noul`` asking whether the caller said more than what the bot asked for."""

MORE_INSTRUCTIONS = (
    "The bot said `bot_message` and the caller replied `user_reply`, a speech-to-text "
    "transcript of a phone call. Beyond answering what the bot asked, does the caller ask "
    "a question, raise a concern, or say something else the bot should respond to?"
)

MORE_CRITERIA: NoulCriteria = {
    "true": {
        "what": (
            "The reply asks something, objects, hesitates, or adds a request the bot "
            "should address, even if it also answers the question"
        ),
        "examples": [
            "A large pepperoni, and is it gluten free?",
            "Yes, but how long will it take?",
            "Hmm, what do you recommend?",
        ],
    },
    "false": {
        "what": "The reply only answers what the bot asked, with at most fillers or politeness",
        "examples": ["A large pepperoni", "Yes, that's right", "Two California rolls please"],
    },
}


class Tier(StrEnum):
    """How a caller's turn was handled."""

    CANNED = "canned"
    """TypeSafe was sure: the router acted and spoke a written line."""

    ROUTED = "routed"
    """TypeSafe picked the tool, but the LLM speaks."""

    OPEN = "open"
    """The turn went to the LLM untouched."""


class TypeSafeFlowsRouter(FrameProcessor):
    """Runs Flows tools on TypeSafe judgments and speaks written lines when sure.

    Place it between ``context_aggregator.user()`` and the LLM, give it the
    LLM (to run tool calls through the normal function-call machinery) and
    the lines for each node and tool, and set :attr:`flow_manager` once the
    manager exists.

    Each caller turn costs one TypeSafe request. The tool choice and each
    argument value have to reach ``routing_threshold`` for the router to act
    on them at all, and ``canned_threshold`` for the router to also speak
    the written line instead of the LLM. The request carries one more
    ``Noul``, whether the caller said anything beyond what was asked; the
    written line is used only when that probability is at or below one minus
    ``canned_threshold``, because a line written in advance cannot answer a
    question the caller slipped in. Confidence says how peaked a distribution
    is, not whether acting on it is safe, so give a tool with costly mistakes
    a higher threshold of its own through :class:`ToolLines`.

    The first node's written line is spoken on entry without a judgment,
    since nothing has been said yet. Every turn logs its tier.

    Example::

        router = TypeSafeFlowsRouter(
            judge=TypeSafeJudge(),
            llm=llm,
            nodes={"confirm": NodeLines(say="So that's {{ order.summary }}. Sound good?")},
            tools={"select_pizza_order": ToolLines(options={"size": ["small", "large"]})},
        )
        pipeline = Pipeline([..., context_aggregator.user(), router, llm, tts, ...])
        flow_manager = FlowManager(worker=worker, llm=llm, ...)
        router.flow_manager = flow_manager
    """

    def __init__(
        self,
        *,
        judge: TypeSafeJudge,
        llm: LLMService[Any],
        nodes: Mapping[str, NodeLines] | None = None,
        tools: Mapping[str, ToolLines] | None = None,
        instructions: str = DEFAULT_INSTRUCTIONS,
        no_tool_description: str = DEFAULT_NO_TOOL_DESCRIPTION,
        routing_threshold: float = 0.5,
        canned_threshold: float = 0.9,
        flow_manager: FlowManager | None = None,
        warm_up: bool = True,
        **kwargs,
    ):
        """Initialize the router.

        Args:
            judge: The TypeSafe client wrapper. The router starts it on
                ``StartFrame`` and closes it on ``EndFrame`` or ``CancelFrame``.
            llm: The LLM service the tools are registered with. The router
                runs chosen tools through it.
            nodes: Written lines per node name. A node without a line is
                always spoken by the LLM.
            tools: Judging criteria, argument options and ask lines per tool
                name. A tool that is not listed is judged by its schema
                description and has no argument options.
            instructions: The tool question's instructions. Refer to the state
                fields `bot_message` and `user_reply`.
            no_tool_description: The criteria text of the choice that means
                the turn matches no tool.
            routing_threshold: The confidence a tool choice and each argument
                value need for the router to act on them. Below it the turn
                goes to the LLM. A tool's ``ToolLines.confidence_threshold``
                overrides it.
            canned_threshold: The confidence the tool choice and each value
                need, and one minus the probability the "said more" ``Noul``
                may reach, for the router to speak the written line instead
                of the LLM. A tool's ``ToolLines.canned_threshold`` overrides
                it.
            flow_manager: The flow manager, when it already exists. It is
                normally set through :attr:`flow_manager` after construction.
            warm_up: Whether to open the TypeSafe connection on start.
            **kwargs: Additional arguments passed to :class:`FrameProcessor`.
        """
        super().__init__(**kwargs)
        self._judge = judge
        self._llm = llm
        self._nodes = dict(nodes or {})
        self._tools = dict(tools or {})
        self._instructions = instructions
        self._no_tool_description = no_tool_description
        self._routing_threshold = routing_threshold
        self._canned_threshold = canned_threshold
        self._flow_manager = flow_manager
        self._warm_up = warm_up

        self._judge_task: asyncio.Task | None = None
        self._held_frame: LLMContextFrame | None = None
        # Whether the next node entry is spoken from its written line. Set
        # when a canned-tier tool call caused the transition, and at start
        # for the first node.
        self._canned_entry = False
        # The written line spoken last, if the router spoke last. The context
        # holds it only as far as TTS had gone when the caller replied.
        self._last_said: str | None = None
        # Arguments the caller has given so far for a tool, kept until the
        # tool runs or the flow leaves the node: (node, tool) -> {arg: value}.
        self._remembered: dict[tuple[str | None, str], dict[str, Any]] = {}

        self.set_core_metrics_data(MetricsData(processor=self.name, model=judge.model))

    @property
    def flow_manager(self) -> FlowManager | None:
        """The flow manager whose node and state the lines come from."""
        return self._flow_manager

    @flow_manager.setter
    def flow_manager(self, flow_manager: FlowManager) -> None:
        self._flow_manager = flow_manager

    def can_generate_metrics(self) -> bool:
        """Processing time and token usage are reported per judgment."""
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Judge caller turns, speak canned node entries, pass everything else.

        Args:
            frame: The frame to process.
            direction: The direction the frame is moving in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._judge.start()
            if self._warm_up:
                self.create_task(self._judge.warm_up())
            self._canned_entry = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            await self._release_held_frame()
            await self.push_frame(frame, direction)
            await self._judge.close()
        elif isinstance(frame, CancelFrame):
            await self._drop_held_frame()
            await self.push_frame(frame, direction)
            await self._judge.close()
        elif isinstance(frame, (InterruptionFrame, UserStartedSpeakingFrame)):
            await self._drop_held_frame()
            await self.push_frame(frame, direction)
        elif (
            isinstance(frame, LLMContextFrame)
            and direction == FrameDirection.DOWNSTREAM
            and not frame.speculation
        ):
            await self._drop_held_frame()
            last = last_message(frame.context)
            if last is not None and last.get("role") == "user":
                self._held_frame = frame
                self._judge_task = self.create_task(self._judge_turn(frame))
            else:
                await self._enter_node(frame)
        else:
            await self.push_frame(frame, direction)

    async def _drop_held_frame(self):
        """Cancel a judgment in flight and forget the frame it was about."""
        self._held_frame = None
        if self._judge_task:
            await self.cancel_task(self._judge_task)
            self._judge_task = None

    async def _release_held_frame(self):
        """Cancel a judgment in flight and let its frame continue to the LLM."""
        frame = self._held_frame
        await self._drop_held_frame()
        if frame:
            await self._to_llm(frame)

    # Node entry

    async def _enter_node(self, frame: LLMContextFrame) -> None:
        node = self._current_node()
        # Whatever the caller had given for tools on an earlier node no longer applies.
        self._remembered = {key: value for key, value in self._remembered.items() if key[0] == node}
        canned, self._canned_entry = self._canned_entry, False
        lines = self._nodes.get(node or "")
        if canned and lines is not None and lines.say is not None:
            logger.info(f"{self}: entering node {node!r} with its written line")
            if await self._say(lines.say):
                return
        elif canned:
            logger.debug(f"{self}: node {node!r} has no written line; the LLM speaks")
        await self._to_llm(frame)

    # Caller turns

    async def _judge_turn(self, frame: LLMContextFrame) -> None:
        context = frame.context
        node = self._current_node()
        schemas = tool_schemas(context)
        if not schemas:
            await self._finish(frame, Tier.OPEN, "node has no tools")
            return

        state = default_state_builder(context)
        if self._last_said:
            state["bot_message"] = self._last_said
        questions: dict[str, Any] = build_questions(
            schemas,
            self._tools,
            instructions=self._instructions,
            no_tool_description=self._no_tool_description,
        )
        questions[MORE_QUESTION_ID] = Noul(instructions=MORE_INSTRUCTIONS, criteria=MORE_CRITERIA)

        try:
            await self.start_processing_metrics()
            result = await self._judge.ask(state, questions)
            await self.stop_processing_metrics()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self.stop_processing_metrics()
            await self.push_error(f"TypeSafe judgment failed: {e}", e)
            await self._finish(frame, Tier.OPEN, f"judgment failed: {e.__class__.__name__}")
            return

        await self._report_usage(result)
        tool_choice = result.choices.get(TOOL_QUESTION_ID)
        if tool_choice is None:
            await self._finish(frame, Tier.OPEN, "response had no tool answer")
            return
        more = result.nouls.get(MORE_QUESTION_ID)
        logger.debug(
            f"{self}: judged {state} in {result.latency_secs * 1000:.0f}ms: "
            f"{tool_choice.probabilities} more={more.probability if more else None}"
        )

        tool = tool_choice.choice
        lines = self._tools.get(tool, ToolLines())
        routing = (
            lines.confidence_threshold
            if lines.confidence_threshold is not None
            else self._routing_threshold
        )
        canned = (
            lines.canned_threshold if lines.canned_threshold is not None else self._canned_threshold
        )
        if tool == NO_TOOL or tool_choice.confidence < routing:
            await self._finish(
                frame, Tier.OPEN, f"{tool!r} at confidence {tool_choice.confidence:.2f}"
            )
            return

        schema = next(s for s in schemas if s.name == tool)
        given = given_arguments(result, schema, routing)
        remembered = self._remembered.setdefault((node, tool), {})
        remembered.update(given)

        # Sure enough for a written line: the tool, every value given this
        # turn, and nothing else in the reply for the line to miss.
        sure = (
            tool_choice.confidence >= canned
            and more is not None
            and more.probability <= 1 - canned
            and all(
                result.choices[argument_question_id(tool, name)].confidence >= canned
                for name in given
            )
        )

        missing = [name for name in schema.required if name not in remembered]
        if missing:
            first = missing[0]
            stated = result.nouls.get(stated_question_id(tool, first))
            line = lines.ask.get(first)
            if (
                sure
                and line is not None
                and stated is not None
                and stated.probability <= 1 - canned
            ):
                logger.info(f"{self}: {Tier.CANNED.value} tier: {tool} asks for {first!r}")
                if await self._say(line, args=dict(remembered)):
                    self._held_frame = None
                    self._judge_task = None
                    return
            await self._finish(frame, Tier.ROUTED, f"{tool} still needs {first!r}; the LLM asks")
            return

        arguments = dict(remembered)
        self._remembered.pop((node, tool), None)
        self._held_frame = None
        self._judge_task = None
        self._canned_entry = sure
        tier = Tier.CANNED if sure else Tier.ROUTED
        logger.info(f"{self}: {tier.value} tier: running {tool} with {arguments}")
        # The LLM learns the handlers Flows advertises from the context frames
        # it sees, and a canned turn never reaches it, so sync them here the
        # way a context frame would.
        self._llm._sync_registered_tool_handlers(context.tools)
        await self._llm.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name=tool,
                    tool_call_id=f"typesafe-{uuid.uuid4().hex[:12]}",
                    arguments=arguments,
                    context=context,
                )
            ]
        )

    async def _finish(self, frame: LLMContextFrame, tier: Tier, why: str) -> None:
        """Log the tier and hand the caller's turn to the LLM."""
        logger.info(f"{self}: {tier.value} tier: {why}")
        self._judge_task = None
        if self._held_frame is frame:
            self._held_frame = None
            await self._to_llm(frame)

    async def _to_llm(self, frame: LLMContextFrame) -> None:
        self._last_said = None
        self._canned_entry = False
        await self.push_frame(frame)

    # Speaking

    async def _say(self, text: str, **extra: Any) -> bool:
        """Speak a written line as LLM text. False when it could not be rendered."""
        state = self._flow_manager.state if self._flow_manager else {}
        try:
            line = render_line(text, state, **extra)
        except FlowError as e:
            await self.push_error(str(e), e)
            return False
        self._last_said = line
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(line))
        await self.push_frame(LLMFullResponseEndFrame())
        return True

    def _current_node(self) -> str | None:
        return self._flow_manager.current_node if self._flow_manager else None

    async def _report_usage(self, result: JudgeResult) -> None:
        if result.input_tokens is None and result.output_tokens is None:
            return
        prompt = result.input_tokens or 0
        completion = result.output_tokens or 0
        await self.start_llm_usage_metrics(
            LLMTokenUsage(
                prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
            )
        )
