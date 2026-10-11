#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Thinking blocks Anthropic rejects because the conversation changed after them.

Newer Claude models bind each thinking block to the system prompt, the tools
and the earlier messages it was created with, and reject a request that replays
it once any of them has changed. Removing thinking blocks from the start of a
conversation keeps the later ones valid.
"""

import json
from typing import Any

from anthropic import BadRequestError

from pipecat.adapters.services.anthropic_adapter import (
    AnthropicLLMInvocationParams,
    anthropic_is_given,
)


class StaleThinking:
    """The thinking blocks a conversation should no longer send.

    Parameters:
        signatures: Signatures of the thinking blocks to leave out.
    """

    def __init__(self):
        """Initialize with no stale thinking blocks."""
        self.signatures: set[str] = set()
        self._binding: str | None = None

    def binding_changed(self, params: AnthropicLLMInvocationParams) -> bool:
        """Record a request's system prompt and tools.

        Args:
            params: Invocation parameters of the request.

        Returns:
            Whether they differ from the previous request's.
        """
        binding = _binding(params)
        changed = self._binding is not None and binding != self._binding
        self._binding = binding
        return changed

    def add(self, signatures: set[str]) -> bool:
        """Leave thinking blocks out from now on.

        Args:
            signatures: Signatures of the thinking blocks.

        Returns:
            Whether any of them was not already left out.
        """
        new = signatures - self.signatures
        self.signatures |= new
        return bool(new)


def is_binding_error(error: BadRequestError) -> bool:
    """Whether Anthropic rejected a thinking block made in a different conversation.

    Args:
        error: The rejection.

    Returns:
        True for a thinking block bound to a different conversation.
    """
    # The rejection has no error code of its own.
    return "bound to a different conversation" in error.message


def _binding(params: AnthropicLLMInvocationParams) -> str:
    """Key the system prompt and tool set, ignoring cache markers and tool order."""
    system = params["system"] if anthropic_is_given(params["system"]) else None
    if isinstance(system, str):
        # Prompt caching sends the same prompt as text blocks.
        system = [{"type": "text", "text": system}]
    tools = sorted(_dumps(tool) for tool in params["tools"])
    return _dumps([system, tools])


def _dumps(value: Any) -> str:
    return json.dumps(_without_cache_control(value), sort_keys=True, default=str)


def _without_cache_control(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_cache_control(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_without_cache_control(v) for v in value]
    return value
