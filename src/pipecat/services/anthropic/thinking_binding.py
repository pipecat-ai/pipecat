#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Keeping Anthropic thinking blocks valid when a conversation changes.

Newer Claude models bind each thinking block to the system prompt, the tools
and the earlier messages it was created with. Where Anthropic enforces this, a
request that replays a block after any of them has changed is rejected.
Removing thinking blocks from the start of a conversation keeps the later ones
valid.
"""

import json
from collections.abc import Mapping
from typing import Any
from weakref import WeakKeyDictionary

from anthropic import BadRequestError

from pipecat.adapters.llm.services.anthropic_adapter import (
    AnthropicLLMInvocationParams,
    anthropic_is_given,
)
from pipecat.processors.aggregators.llm_context import LLMContext


class ThinkingBinding:
    """Tells when a context's thinking no longer matches its system prompt or tools.

    Thinking is left alone until Anthropic first rejects a block, since accounts
    it doesn't enforce binding on accept stale blocks.

    Parameters:
        enforced: Whether Anthropic has rejected a thinking block.
    """

    def __init__(self):
        """Initialize with nothing enforced or recorded."""
        self.enforced = False
        self._bindings: WeakKeyDictionary[LLMContext, str] = WeakKeyDictionary()

    def changed(self, context: LLMContext, params: AnthropicLLMInvocationParams) -> bool:
        """Check whether a request's thinking would be rejected for a changed binding.

        Args:
            context: The LLM context the request is built from.
            params: Invocation parameters of the request.

        Returns:
            True when binding is enforced and the system prompt or tools differ
            from the context's previous request.
        """
        previous = self._bindings.get(context)
        return self.enforced and previous is not None and previous != _binding(params)

    def record(self, context: LLMContext, params: AnthropicLLMInvocationParams):
        """Remember the system prompt and tools a context's request is sent with.

        Args:
            context: The LLM context the request is built from.
            params: Invocation parameters of the request.
        """
        self._bindings[context] = _binding(params)


def is_binding_error(error: BadRequestError) -> bool:
    """Check whether Anthropic rejected a thinking block made in a different conversation.

    Args:
        error: The rejection.

    Returns:
        True for a thinking block bound to a different conversation.
    """
    # It has no error code of its own. A tampered signature lacks this phrase.
    return "bound to a different conversation" in error.message


def _binding(params: AnthropicLLMInvocationParams) -> str:
    """Key the system prompt and tool set, ignoring cache markers and tool order."""
    system = params["system"] if anthropic_is_given(params["system"]) else []
    if isinstance(system, str):
        # Prompt caching sends the same prompt as a text block.
        system = [{"type": "text", "text": system}]
    tools = sorted(_dumps(_without_cache_control(tool)) for tool in params["tools"])
    return _dumps([[_without_cache_control(block) for block in system], tools])


def _without_cache_control(block: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in block.items() if k != "cache_control"}


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)
