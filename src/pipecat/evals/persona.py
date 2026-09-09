#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The simulated caller: who they are, what they want, and how they end the call.

An :class:`EvalPersona` is the instruction the persona LLM runs under, the
LLM service itself, which rides in the eval pipeline, and the context it
runs on, where the bot's turns are the ``user`` messages and its own the
``assistant`` ones. The context offers one tool, ``end_call``, which the
persona calls once its goal is achieved or clearly out of reach; its
``success`` claim is its own view, and the judge decides the outcome.
"""

from collections.abc import Awaitable, Callable

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallParams, LLMService

END_CALL_FUNCTION = "end_call"

END_CALL_SCHEMA = FunctionSchema(
    name=END_CALL_FUNCTION,
    description=(
        "End the call. Call this instead of speaking once your goal is achieved, "
        "or once it is clear the assistant cannot help you achieve it."
    ),
    properties={
        "success": {
            "type": "boolean",
            "description": "Whether you achieved your goal on this call.",
        },
        "reason": {
            "type": "string",
            "description": "One sentence on why you are ending the call.",
        },
    },
    required=["success", "reason"],
)

_INSTRUCTION_TEMPLATE = """\
You are playing a person on a phone call with a voice assistant. Stay in \
character throughout; never mention being simulated, a test, or an AI.

Who you are: {persona}

What you want from this call: {goal}

The assistant's words arrive as the user's messages. Reply with only what you \
would say next: one short spoken turn, in the first person, in plain sentences \
(no lists, markdown, or stage directions). Ask for or give one thing at a time, \
as a real caller would, and do not repeat what the assistant has already \
understood. When your goal is achieved, or it is clear the assistant cannot \
help, say nothing more and call the {end_call} tool with whether you succeeded \
and why."""


class EvalPersona:
    """The simulated caller: its instruction, its LLM, and the context it runs on."""

    def __init__(self, description: str, goal: str, llm: LLMService):
        """Initialize the persona.

        Args:
            description: Who the caller is, as free text.
            goal: What the caller wants from the call.
            llm: The LLM service that plays the caller, run inside the eval pipeline.
        """
        self._description = description
        self._goal = goal
        self._llm = llm
        self._context = LLMContext(tools=ToolsSchema(standard_tools=[END_CALL_SCHEMA]))

    @property
    def instruction(self) -> str:
        """The system instruction the persona LLM runs under."""
        return _INSTRUCTION_TEMPLATE.format(
            persona=self._description, goal=self._goal, end_call=END_CALL_FUNCTION
        )

    @property
    def llm(self) -> LLMService:
        """The LLM service that plays the caller, a processor in the eval pipeline."""
        return self._llm

    @property
    def context(self) -> LLMContext:
        """The context the persona runs on, with the ``end_call`` tool; empty until the call starts."""
        return self._context

    def on_end_call(self, handler: Callable[[FunctionCallParams], Awaitable[None]]) -> None:
        """Register what happens when the persona calls ``end_call``.

        Args:
            handler: Awaited with the call's params; its ``success`` and
                ``reason`` arguments are the persona's own claim.
        """
        self._llm.register_function(END_CALL_FUNCTION, handler)
