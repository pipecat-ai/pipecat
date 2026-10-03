#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Limit on LLM responses in a row that call functions."""

from dataclasses import dataclass

from loguru import logger

from pipecat.processors.aggregators.llm_context import (
    NOT_GIVEN,
    LLMContext,
    LLMContextToolChoice,
    NotGiven,
)

DEFAULT_FUNCTION_CALL_LIMIT_PROMPT = (
    "You have reached the limit of function calls for this turn. "
    "Answer the user now with the information you already have."
)


@dataclass
class FunctionCallLimitConfig:
    """How many LLM responses in a row may call functions.

    Each LLM response that calls functions is one iteration, and iterations in
    a row form a chain. After ``max_iterations`` iterations, the assistant
    aggregator runs the LLM on their results with the context tool choice set
    to ``"none"``, so the LLM answers instead of calling again. The tools stay
    in the context, so the request keeps its tool definitions and the context
    only grows at its end. A response with no function call ends the chain, as
    does an interruption, and restores the tool choice. Speech outside an LLM
    response, such as a ``TTSSpeakFrame``, does not end the chain. On an LLM
    service that ignores the context tool choice, only the prompt asks the
    LLM to answer, and the LLM can keep calling functions.

    Parameters:
        max_iterations: How many LLM responses in a row may call functions.
        prompt: Developer message appended to the context when the limit
            applies. ``None`` appends no message.
    """

    max_iterations: int
    prompt: str | None = DEFAULT_FUNCTION_CALL_LIMIT_PROMPT


class FunctionCallLimiter:
    """Counts the iterations of a chain and applies the limit to the context.

    With no config, it counts and never applies a limit.
    """

    def __init__(self, context: LLMContext, config: FunctionCallLimitConfig | None):
        """Initialize the limiter.

        Args:
            context: The context whose tool choice the limit sets.
            config: The limit. ``None`` sets no limit.
        """
        self._context = context
        self._config = config
        self._iterations = 0
        self._called_in_response = False
        self._limited = False
        self._tool_choice: LLMContextToolChoice | NotGiven = NOT_GIVEN

    def function_calls_started(self):
        """Count one iteration: an LLM response started function calls."""
        self._iterations += 1
        self._called_in_response = True

    def response_ended(self):
        """End the chain if the LLM response that ended called no function."""
        if not self._called_in_response:
            self.end_chain()
        self._called_in_response = False

    def apply(self):
        """Set the tool choice to ``"none"`` and add the prompt, once the chain is at the limit."""
        config = self._config
        if not config or self._limited or self._iterations < config.max_iterations:
            return
        logger.debug(f"Function call limit of {config.max_iterations} reached")
        self._limited = True
        self._tool_choice = self._context.tool_choice
        self._context.set_tool_choice("none")
        if config.prompt:
            self._context.add_message({"role": "developer", "content": config.prompt})

    def end_chain(self):
        """Reset the count and restore the tool choice the limit replaced."""
        self._iterations = 0
        self._called_in_response = False
        if not self._limited:
            return
        self._limited = False
        # A tool choice set while the limit applied is kept.
        if self._context.tool_choice == "none":
            self._context.set_tool_choice(self._tool_choice)
