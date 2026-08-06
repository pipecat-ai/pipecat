#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Configuration for announcing completed async tool results."""

from dataclasses import dataclass
from enum import StrEnum
from string import Formatter
from typing import Any

from pipecat.processors.aggregators.llm_context import LLMContextMessage

DEFAULT_SINGLE_RESULT_PROMPT = (
    "The background task '{name}' you started earlier has finished, and its result is "
    "already in the conversation. Tell the user what it found now — state the actual "
    "finding, briefly, rather than only saying the task is done. Refer to the task the "
    "way it came up in conversation, not by its function name."
)

DEFAULT_SINGLE_NOTIFY_PROMPT = (
    "The background task '{name}' you started earlier has finished, and its result is "
    "already in the conversation. Briefly tell the user the result is ready and ask "
    "whether they'd like to hear it now. Do not state the result itself yet. Refer to "
    "the task the way it came up in conversation, not by its function name."
)

DEFAULT_MULTIPLE_RESULT_PROMPT = (
    "{count} background tasks you started earlier have finished: {names}. Their results "
    "are already in the conversation. Tell the user what each one found now — state the "
    "actual findings, briefly, rather than only saying the tasks are done. Refer to the "
    "tasks the way they came up in conversation, not by their function names."
)

DEFAULT_MULTIPLE_NOTIFY_PROMPT = (
    "{count} background tasks you started earlier have finished: {names}. Their results "
    "are already in the conversation. Briefly tell the user the results are ready and ask "
    "which they'd like to hear. Do not state the results themselves yet. Refer to the "
    "tasks the way they came up in conversation, not by their function names."
)

# Placeholders a custom prompt may use, by cardinality.
SINGLE_PROMPT_FIELDS = frozenset({"name"})
MULTIPLE_PROMPT_FIELDS = frozenset({"count", "names"})


@dataclass(frozen=True)
class CompletedToolResult:
    """A completed async tool call whose announcement is scheduled by a response strategy.

    Queued by the ``LLMAssistantAggregator`` when an async function call
    (``cancel_on_interruption=False``) delivers its final result while a
    response strategy is configured (unless ``run_llm=False`` opted out of a
    response entirely). The result *content* is already in the LLM context by
    the time this is queued — only the announcement is scheduled.

    Classification is tool-call protocol, not tool implementation: a call
    that awaits a job group across workers routes identically to one that
    awaits an HTTP request. Mark a call async iff its *final result is the
    long-awaited content* (acknowledge via an intermediate result). A
    fire-and-forget dispatch tool whose final result is just "started" is a
    fast *reactive* tool — register it normally so its acknowledgment stays
    immediate, and queue a ``ResponseFrame`` from wherever the real
    completion arrives.

    Parameters:
        function_name: Name of the function that completed.
        tool_call_id: Unique identifier of the function call.
        arguments: Arguments the function was called with.
        result: The result the function returned.
    """

    function_name: str
    tool_call_id: str
    arguments: Any
    result: Any


class AnnouncementStyle(StrEnum):
    """How much of a completed tool result the bot volunteers.

    The result itself is in the LLM context either way, so the model can
    always use it once the user asks — the style only decides how much the
    bot says unprompted.

    Parameters:
        RESULT: State the result.
        NOTIFY: Say the result is ready and offer it, without stating it.
    """

    RESULT = "result"
    NOTIFY = "notify"


@dataclass
class AnnouncementConfig:
    """Configuration for announcing completed async tool results.

    Style is set per cardinality because the two cases call for different
    behavior: one finished task can be reported outright, while several at
    once become a wall of speech nobody asked for — so a batch defaults to
    offering rather than delivering.

    This configures what is said when a result *is* announced. To announce
    nothing at all, have the tool deliver its final result with
    ``FunctionCallResultProperties(run_llm=False)``: the result still lands
    in the LLM context, so it answers a later question, but it never
    interrupts. That decision belongs to the tool, which is the only place
    that knows whether a particular result was worth interrupting for.

    Parameters:
        single_style: How to announce one completed tool result.
        multiple_style: How to announce several completed together.
        single_prompt: Instruction for the single-result case, replacing the
            default for ``single_style``. May use ``{name}``.
        multiple_prompt: Instruction for the batch case, replacing the default
            for ``multiple_style``. May use ``{count}`` and ``{names}``.
    """

    single_style: AnnouncementStyle = AnnouncementStyle.RESULT
    multiple_style: AnnouncementStyle = AnnouncementStyle.NOTIFY
    single_prompt: str | None = None
    multiple_prompt: str | None = None

    def __post_init__(self):
        """Validate the placeholders in any custom prompt."""
        _validate_prompt_fields("single_prompt", self.single_prompt, SINGLE_PROMPT_FIELDS)
        _validate_prompt_fields("multiple_prompt", self.multiple_prompt, MULTIPLE_PROMPT_FIELDS)

    def compose(self, completed: "list[CompletedToolResult]") -> list[LLMContextMessage]:
        """Compose the instruction message announcing a released batch.

        Args:
            completed: The completed tool results being released together.

        Returns:
            A single developer-role instruction message, or no messages when
            nothing completed.
        """
        if not completed:
            return []
        if len(completed) == 1:
            style, prompt = self.single_style, self.single_prompt
        else:
            style, prompt = self.multiple_style, self.multiple_prompt
        content = (prompt or _default_prompt(len(completed), style)).format(
            name=completed[0].function_name,
            count=len(completed),
            names=", ".join(f"'{item.function_name}'" for item in completed),
        )
        return [{"role": "developer", "content": content}]


def _default_prompt(count: int, style: AnnouncementStyle) -> str:
    """Pick the shipped prompt for a cardinality and style."""
    if count == 1:
        if style is AnnouncementStyle.NOTIFY:
            return DEFAULT_SINGLE_NOTIFY_PROMPT
        return DEFAULT_SINGLE_RESULT_PROMPT
    if style is AnnouncementStyle.NOTIFY:
        return DEFAULT_MULTIPLE_NOTIFY_PROMPT
    return DEFAULT_MULTIPLE_RESULT_PROMPT


def _validate_prompt_fields(field: str, prompt: str | None, allowed: frozenset[str]):
    """Raise if a custom prompt uses a placeholder that won't be filled in."""
    if prompt is None:
        return
    used = {name for _, name, _, _ in Formatter().parse(prompt) if name}
    unknown = used - allowed
    if unknown:
        raise ValueError(
            f"{field} uses unknown placeholder(s) {sorted(unknown)}; available: {sorted(allowed)}"
        )
