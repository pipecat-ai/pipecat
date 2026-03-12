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

2. **No non-initial system messages** — "system" messages are only allowed as
   the very first message. A system message anywhere else causes:
   ``"only the initial message can have the system role"``

3. **Last message must be user/tool** — The final message in the conversation
   must have role "user" or "tool". A trailing "assistant" message causes:
   ``"the last message must have the user or tool role"``

This adapter transforms the message list to satisfy all three constraints before
the messages are sent to Perplexity's API.
"""

import copy
from typing import List

from openai.types.chat import ChatCompletionMessageParam

from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter, OpenAILLMInvocationParams
from pipecat.processors.aggregators.llm_context import LLMContext


class PerplexityLLMAdapter(OpenAILLMAdapter):
    """Adapter that transforms messages to satisfy Perplexity's API constraints.

    Perplexity's API is stricter than standard OpenAI about message structure.
    This adapter extends ``OpenAILLMAdapter`` and applies message transformations
    to ensure compliance with Perplexity's three constraints (role alternation,
    no non-initial system messages, last message must be user/tool).

    The transformations are applied in ``get_llm_invocation_params`` after the
    parent adapter extracts messages from the LLM context, and before
    ``build_chat_completion_params`` prepends ``system_instruction``.
    """

    def get_llm_invocation_params(self, context: LLMContext) -> OpenAILLMInvocationParams:
        """Get OpenAI-compatible invocation parameters with Perplexity message fixes applied.

        Args:
            context: The LLM context containing messages, tools, etc.

        Returns:
            Dictionary of parameters for Perplexity's ChatCompletion API, with
            messages transformed to satisfy Perplexity's constraints.
        """
        params = super().get_llm_invocation_params(context)
        params["messages"] = self._transform_messages(list(params["messages"]))
        return params

    def _transform_messages(
        self, messages: List[ChatCompletionMessageParam]
    ) -> List[ChatCompletionMessageParam]:
        """Transform messages to satisfy Perplexity's API constraints.

        Applies four transformation steps in order:

        1. **Merge consecutive initial system messages** — If the conversation
           starts with multiple system messages, merge them into a single system
           message using list-of-dicts content format. This addresses
           Perplexity's constraint that only the initial message can be system.

        2. **Convert non-initial system messages to user** — Any system message
           after the initial position is converted to role "user", since
           Perplexity rejects non-initial system messages.

        3. **Merge consecutive same-role messages** — After the above
           conversions, adjacent messages with the same role are merged using
           list-of-dicts content format. This ensures strict role alternation
           (e.g. a converted system→user message adjacent to an existing user
           message gets merged).

        4. **Remove trailing assistant messages** — If the last message is
           "assistant", remove it. OpenAI appears to silently ignore trailing
           assistant messages server-side, so removing them preserves equivalent
           behavior while satisfying Perplexity's "last message must be
           user/tool" constraint. If the only remaining message is "system"
           (possible when the context contains just a single system message),
           convert it to "user" since Perplexity requires the last message to
           be "user" or "tool".

        Args:
            messages: List of message dicts with "role" and "content" keys.

        Returns:
            Transformed list of message dicts satisfying Perplexity's constraints.
        """
        if not messages:
            return messages

        messages = copy.deepcopy(messages)

        # Step 1: Merge consecutive system messages at the start into one.
        # Perplexity only allows a single initial system message, so if there
        # are multiple consecutive system messages at the start, we merge them.
        if messages[0].get("role") == "system":
            system_end = 1
            while system_end < len(messages) and messages[system_end].get("role") == "system":
                system_end += 1

            if system_end > 1:
                # Merge all initial system messages into a single message using
                # list-of-dicts content format (same approach as Anthropic adapter).
                merged_content = []
                for msg in messages[:system_end]:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        merged_content.append({"type": "text", "text": content})
                    elif isinstance(content, list):
                        merged_content.extend(content)
                messages = [{"role": "system", "content": merged_content}] + messages[system_end:]

        # Step 2: Convert non-initial system messages to "user".
        # Perplexity only allows system role for the very first message.
        for i in range(1, len(messages)):
            if messages[i].get("role") == "system":
                messages[i]["role"] = "user"

        # Step 3: Merge consecutive same-role messages.
        # After system→user conversions above, we may have adjacent same-role
        # messages that violate Perplexity's strict alternation requirement.
        i = 0
        while i < len(messages) - 1:
            current = messages[i]
            next_msg = messages[i + 1]
            if current["role"] == next_msg["role"]:
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

        # Step 4: Handle trailing messages.
        # Perplexity requires the last message to be "user" or "tool".
        if messages:
            # Remove trailing assistant messages. OpenAI appears to silently
            # ignore trailing assistant messages server-side, so removing them
            # preserves equivalent behavior.
            while messages and messages[-1].get("role") == "assistant":
                messages.pop()

            # If the only remaining message is "system" (single system message
            # in the context), convert it to "user".
            if messages and len(messages) == 1 and messages[0].get("role") == "system":
                messages[0]["role"] = "user"

        return messages
