#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Live LLM adapter for Pipecat."""

from typing import Any, cast

from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter
from pipecat.processors.aggregators.llm_context import LLMContextMessage, LLMSpecificMessage


class GeminiLiveLLMAdapter(GeminiLLMAdapter):
    """Gemini Live-specific adapter for Pipecat.

    Same as :class:`GeminiLLMAdapter`, except that tool calls and their results
    are converted to text, which is all Gemini Live's API accepts.
    """

    @property
    def id_for_llm_specific_messages(self) -> str:
        """Get the identifier used in LLMSpecificMessage instances for Gemini Live."""
        return "gemini-live"

    def _from_universal_context_messages(
        self,
        universal_context_messages: list[LLMContextMessage],
        *,
        system_instruction: str | None = None,
    ) -> GeminiLLMAdapter.ConvertedMessages:
        """Restructure messages to ensure proper Gemini Live format and message ordering.

        Args:
            universal_context_messages: Messages from the LLM context.
            system_instruction: Optional system instruction from service settings,
                used to decide whether to extract an initial "developer" message.
        """
        return super()._from_universal_context_messages(
            self._convert_tool_calls_to_text(universal_context_messages),
            system_instruction=system_instruction,
        )

    @staticmethod
    def _convert_tool_calls_to_text(messages: list[LLMContextMessage]) -> list[LLMContextMessage]:
        """Convert tool calls and their results to text messages.

        Seeding a Gemini Live conversation with tool calls — on a reconnect or a
        multi-worker handoff, say — fails with a 1007 error.

        Args:
            messages: Messages from the LLM context.

        Returns:
            The messages, with each tool call and result replaced by a text
            message describing it.
        """
        tool_call_names: dict[str, str] = {}
        converted: list[LLMContextMessage] = []

        for message in messages:
            if isinstance(message, LLMSpecificMessage):
                converted.append(message)
                continue

            # ChatCompletionMessageParam (a union of TypedDicts) doesn't allow
            # the dict-style key access used below; treat it as a plain dict.
            msg = cast(dict[str, Any], message)
            if msg.get("tool_calls"):
                summaries = []
                for tool_call in msg["tool_calls"]:
                    function = tool_call["function"]
                    tool_call_names[tool_call["id"]] = function["name"]
                    summaries.append(
                        f"[Called function {function['name']} with args {function['arguments']}]"
                    )
                converted.append({"role": "assistant", "content": " ".join(summaries)})
            elif msg.get("role") == "tool":
                # Same fallback name the Gemini adapter uses for a result whose
                # call isn't in the context.
                name = tool_call_names.get(msg.get("tool_call_id", ""), "tool_call_result")
                converted.append(
                    {"role": "user", "content": f"[Function {name} returned {msg['content']}]"}
                )
            else:
                converted.append(message)

        return converted
