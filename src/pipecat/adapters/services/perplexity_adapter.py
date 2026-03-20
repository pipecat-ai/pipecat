#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Perplexity LLM adapter for Pipecat.

Perplexity's API uses an OpenAI-compatible interface but enforces stricter
constraints on conversation history structure:

1. **Strict role alternation** — Messages must alternate between "user"/"tool"
   and "assistant" roles. Consecutive messages with the same role (e.g. two
   "user" messages in a row) are rejected with:
   ``"messages must be an alternating sequence of user/tool and assistant messages"``

2. **No non-initial system messages** — "system" messages are only allowed at
   the start of the conversation. A system message after a non-system message
   causes:
   ``"only the initial message can have the system role"``

3. **Last message must be user/tool** — The final message in the conversation
   must have role "user" or "tool". A trailing "assistant" message causes:
   ``"the last message must have the user or tool role"``

This adapter transforms the message list to satisfy all three constraints before
the messages are sent to Perplexity's API.
"""

import copy
from typing import List, Optional

from openai.types.chat import ChatCompletionMessageParam

from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter, OpenAILLMInvocationParams
from pipecat.processors.aggregators.llm_context import LLMContext


class PerplexityLLMAdapter(OpenAILLMAdapter):
    """Adapter that transforms messages to satisfy Perplexity's API constraints.

    Perplexity's API is stricter than OpenAI about message structure. This
    adapter extends ``OpenAILLMAdapter`` and applies message transformations
    to ensure compliance with Perplexity's constraints (role alternation,
    no non-initial system messages, last message must be user/tool).

    The transformations are applied in ``get_llm_invocation_params`` after the
    parent adapter extracts messages from the LLM context (including any
    ``system_instruction`` prepend).
    """

    def get_llm_invocation_params(
        self, context: LLMContext, *, system_instruction: Optional[str] = None
    ) -> OpenAILLMInvocationParams:
        """Get OpenAI-compatible invocation parameters with Perplexity message fixes applied.

        Args:
            context: The LLM context containing messages, tools, etc.
            system_instruction: Optional system instruction from service settings
                or ``run_inference``. Forwarded to the parent adapter.

        Returns:
            Dictionary of parameters for Perplexity's ChatCompletion API, with
            messages transformed to satisfy Perplexity's constraints.
        """
        params = super().get_llm_invocation_params(context, system_instruction=system_instruction)
        params["messages"] = self._transform_messages(list(params["messages"]))
        return params

    def _transform_messages(
        self, messages: List[ChatCompletionMessageParam]
    ) -> List[ChatCompletionMessageParam]:
        """Transform messages to satisfy Perplexity's API constraints.

        Applies three transformation steps in order:

        1. **Convert non-initial system messages to user** — Any system message
           after the initial system message block is converted to role "user",
           since Perplexity rejects system messages after a non-system message.

        2. **Merge consecutive same-role messages** — After the above
           conversions, adjacent messages with the same role are merged using
           list-of-dicts content format. This ensures strict role alternation
           (e.g. a converted system→user message adjacent to an existing user
           message gets merged).

        3. **Remove trailing assistant messages** — If the last message is
           "assistant", remove it. OpenAI appears to silently ignore trailing
           assistant messages server-side, so removing them preserves equivalent
           behavior while satisfying Perplexity's "last message must be
           user/tool" constraint.

        Note: we intentionally do *not* convert a trailing system message to
        "user". That would make the transformation unstable across calls —
        Perplexity appears to have statefulness/caching within a conversation,
        so a message that was sent as "user" in one call but becomes "system"
        in the next (once more messages are appended) causes errors. If the
        context consists entirely of system messages, the Perplexity API call
        will fail, but that mistake will be caught right away.

        Args:
            messages: List of message dicts with "role" and "content" keys.

        Returns:
            Transformed list of message dicts satisfying Perplexity's constraints.
        """
        if not messages:
            return messages

        messages = copy.deepcopy(messages)

        # Step 1: Convert non-initial system messages to "user".
        # Perplexity allows system messages at the start, but rejects them
        # after any non-system message.
        in_initial_system_block = True
        for i in range(len(messages)):
            if messages[i].get("role") == "system":
                if not in_initial_system_block:
                    messages[i]["role"] = "user"
            else:
                in_initial_system_block = False

        # Step 2: Merge consecutive same-role messages.
        # After system→user conversions above, we may have adjacent same-role
        # messages that violate Perplexity's strict alternation requirement.
        # Skip consecutive system messages at the start — Perplexity allows those.
        i = 0
        while i < len(messages) - 1:
            current = messages[i]
            next_msg = messages[i + 1]
            if current["role"] == next_msg["role"] == "system":
                # Perplexity allows multiple initial system messages, don't merge
                i += 1
            elif current["role"] == next_msg["role"]:
                # Convert string content to list-of-dicts format for merging
                if isinstance(current.get("content"), str):
                    current["content"] = [{"type": "text", "text": current["content"]}]
                if isinstance(next_msg.get("content"), str):
                    next_msg["content"] = [{"type": "text", "text": next_msg["content"]}]
                # Merge content from next message into current
                if isinstance(current.get("content"), list) and isinstance(
                    next_msg.get("content"), list
                ):
                    current["content"].extend(next_msg["content"])
                messages.pop(i + 1)
            else:
                i += 1

        # Step 3: Remove trailing assistant messages.
        # Perplexity requires the last message to be "user" or "tool".
        # OpenAI appears to silently ignore trailing assistant messages
        # server-side, so removing them preserves equivalent behavior.
        while messages and messages[-1].get("role") == "assistant":
            messages.pop()

        return messages
