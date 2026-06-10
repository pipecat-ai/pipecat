#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Mistral LLM adapter for Pipecat.

Mistral's API uses an OpenAI-compatible interface but imposes three
conversation-history constraints that OpenAI does not:

1. **Tool messages must be followed by an assistant message.** A ``"tool"``
   role message that isn't followed by an ``"assistant"`` message is
   rejected.

2. **Only the initial contiguous system block is permitted.** A
   ``"system"`` message appearing after any non-system message must be
   converted to ``"user"``.

3. **A trailing assistant message requires ``prefix=True``.** When the
   conversation ends on an assistant message, Mistral expects the
   ``prefix`` flag set so it can continue from that partial reply.

This adapter extends ``OpenAILLMAdapter`` and applies those three fixups
before the messages reach ``build_chat_completion_params``.
"""

import copy
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam

from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter, OpenAILLMInvocationParams
from pipecat.processors.aggregators.llm_context import LLMContext


class MistralLLMAdapter(OpenAILLMAdapter):
    """Adapter that transforms messages to satisfy Mistral's API constraints.

    Mistral accepts the OpenAI chat-completions schema but enforces extra
    rules on conversation history. This adapter extends ``OpenAILLMAdapter``
    and rewrites the messages produced by the parent to comply with those
    rules before the request is built.
    """

    def get_llm_invocation_params(
        self,
        context: LLMContext,
        *,
        system_instruction: str | None = None,
        convert_developer_to_user: bool,
    ) -> OpenAILLMInvocationParams:
        """Get OpenAI-compatible invocation parameters with Mistral message fixes applied.

        Args:
            context: The LLM context containing messages, tools, etc.
            system_instruction: Optional system instruction from service settings
                or ``run_inference``. Forwarded to the parent adapter.
            convert_developer_to_user: If True, convert "developer"-role messages
                to "user"-role messages. Forwarded to the parent adapter.

        Returns:
            Dictionary of parameters for Mistral's ChatCompletion API, with
            messages transformed to satisfy Mistral's constraints.
        """
        params = super().get_llm_invocation_params(
            context,
            system_instruction=system_instruction,
            convert_developer_to_user=convert_developer_to_user,
        )
        params["messages"] = self._transform_messages(list(params["messages"]))
        return params

    def _transform_messages(
        self, messages: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        """Transform messages to satisfy Mistral's API constraints.

        Applies three transformation steps in order:

        1. **Insert assistant messages after tool messages** — Any ``"tool"``
           message not followed by an ``"assistant"`` message gets a minimal
           ``{"role": "assistant", "content": " "}`` inserted after it.

        2. **Convert non-initial system messages to user** — System messages
           after the initial contiguous system block are converted to
           ``"user"``, since Mistral only accepts system messages at the
           start of a conversation.

        3. **Set prefix on trailing assistant message** — If the final message
           is an assistant message without a ``prefix`` field, set
           ``prefix=True`` so Mistral will continue the partial reply.

        Args:
            messages: List of OpenAI-shaped message dicts.

        Returns:
            Transformed list of messages satisfying Mistral's constraints.
        """
        if not messages:
            return messages

        # Work on plain dicts: we need to mutate "role" (which OpenAI TypedDict
        # variants tag with fixed Literals) and to attach Mistral's non-standard
        # "prefix" field. Cast back on return — the outgoing list is valid for
        # Mistral's extended schema even though it doesn't fit OpenAI's.
        msgs: list[dict[str, Any]] = copy.deepcopy([dict(m) for m in messages])

        # Step 1: ensure every "tool" message is followed by an "assistant".
        insert_at: list[int] = []
        for i, msg in enumerate(msgs):
            if msg.get("role") == "tool":
                is_last = i == len(msgs) - 1
                if is_last or msgs[i + 1].get("role") != "assistant":
                    insert_at.append(i + 1)
        for idx in reversed(insert_at):
            msgs.insert(idx, {"role": "assistant", "content": " "})

        # Step 2: convert non-initial system messages to "user".
        # Mistral rejects system messages after any non-system message.
        first_non_system = next(
            (i for i, m in enumerate(msgs) if m.get("role") != "system"),
            len(msgs),
        )
        for i in range(first_non_system, len(msgs)):
            if msgs[i].get("role") == "system":
                msgs[i]["role"] = "user"

        # Step 3: set prefix on a trailing assistant message so Mistral will
        # continue it rather than rejecting the turn.
        last = msgs[-1]
        if last.get("role") == "assistant" and "prefix" not in last:
            last["prefix"] = True

        return cast(list[ChatCompletionMessageParam], msgs)
