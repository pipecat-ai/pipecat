#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""DeepSeek LLM adapter for Pipecat.

DeepSeek's API is OpenAI-compatible, but in thinking mode it requires every
assistant message of the current turn to carry a ``reasoning_content`` field
once a tool call is involved, and rejects the request with a 400 otherwise.
Pipecat does not keep a model's reasoning in the context, so this adapter
supplies an empty ``reasoning_content`` on assistant messages that lack one.
DeepSeek accepts the empty field on any assistant message and ignores it when
thinking is disabled, so it is applied uniformly rather than per turn.
"""

from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam

from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter, OpenAILLMInvocationParams
from pipecat.processors.aggregators.llm_context import LLMContext


class DeepSeekLLMAdapter(OpenAILLMAdapter):
    """Adapter that shapes messages to satisfy DeepSeek's thinking-mode rules.

    Extends ``OpenAILLMAdapter`` and adds an empty ``reasoning_content`` to
    every assistant message that has none, so requests are accepted in
    thinking mode after a tool call.
    """

    def get_llm_invocation_params(
        self,
        context: LLMContext,
        *,
        system_instruction: str | None = None,
        convert_developer_to_user: bool,
    ) -> OpenAILLMInvocationParams:
        """Get OpenAI-compatible invocation parameters with DeepSeek message fixes applied.

        Args:
            context: The LLM context containing messages, tools, etc.
            system_instruction: Optional system instruction from service settings
                or ``run_inference``. Forwarded to the parent adapter.
            convert_developer_to_user: If True, convert "developer"-role messages
                to "user"-role messages. Forwarded to the parent adapter.

        Returns:
            Dictionary of parameters for DeepSeek's ChatCompletion API, with
            ``reasoning_content`` present on every assistant message.
        """
        params = super().get_llm_invocation_params(
            context,
            system_instruction=system_instruction,
            convert_developer_to_user=convert_developer_to_user,
        )
        params["messages"] = self._add_reasoning_content(list(params["messages"]))
        return params

    def _add_reasoning_content(
        self, messages: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        """Return the messages with ``reasoning_content`` on every assistant message.

        Assistant messages that already carry the field are left as they are.
        Message dicts are shared with the source ``LLMContext``, so a stamped
        message is a copy rather than a mutation.

        Args:
            messages: List of OpenAI-shaped message dicts.

        Returns:
            The message list, with copies of the assistant messages that needed
            the field.
        """
        result: list[dict[str, Any]] = []
        for message in messages:
            msg = cast(dict[str, Any], message)
            if msg.get("role") == "assistant" and "reasoning_content" not in msg:
                msg = {**msg, "reasoning_content": ""}
            result.append(msg)
        return cast(list[ChatCompletionMessageParam], result)
