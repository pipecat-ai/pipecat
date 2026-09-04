#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The simulated caller: who they are, what they want, and how they end the call.

A :class:`Persona` turns a simulation's ``persona`` and ``goal`` into the
instruction the persona LLM runs under (set on the service as its system
instruction) and the
:class:`~pipecat.processors.aggregators.llm_context.LLMContext` it runs on
inside the eval pipeline. In that context the bot's turns are the ``user``
messages and the persona's own are the ``assistant`` messages: the persona LLM
answers the bot the way a bot answers a user.

The context advertises one tool, ``end_call``, which the persona calls instead
of speaking once its goal is achieved or clearly out of reach. Its ``success``
claim is the persona's own view; the goal judge decides the run's outcome.
"""

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext

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


class Persona:
    """The simulated caller behind a simulation's persona LLM."""

    def __init__(self, description: str, goal: str):
        """Initialize the persona.

        Args:
            description: Who the caller is, as free text.
            goal: What the caller wants from the call.
        """
        self._description = description
        self._goal = goal

    @property
    def instruction(self) -> str:
        """The system instruction the persona LLM runs under."""
        return _INSTRUCTION_TEMPLATE.format(
            persona=self._description, goal=self._goal, end_call=END_CALL_FUNCTION
        )

    def context(self) -> LLMContext:
        """A fresh context for one run: no messages yet, and the ``end_call`` tool."""
        return LLMContext(tools=ToolsSchema(standard_tools=[END_CALL_SCHEMA]))
