#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The LLM extension that pairs an ``LLMWithBackend`` frontend with its backend."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.services.llm_extensions.base_llm_extension import BaseLLMExtension, LLMExtensionTool

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallRegistryItem


class BackendPairingLLMExtension(BaseLLMExtension):
    """Pairs the frontend of an ``LLMWithBackend`` with its backend.

    The frontend's half of the pairing: the instruction that says when and
    how to delegate and what to do with what the backend sends back, and the
    tools delegation goes through, ``delegate`` first of all. The composite
    builds both from its :class:`~pipecat.pipeline.llm_with_backend.BackendConnector`
    and installs the extension on the frontend; the connector runs the
    session the tools feed.
    """

    installed_by = "LLMWithBackend"

    def __init__(self, *, instruction: str | None, tools: Sequence[FunctionSchema], **kwargs):
        """Initialize the extension.

        Args:
            instruction: What the pairing adds to the frontend's system
                instruction, or ``None`` for nothing.
            tools: The tools delegation goes through, each carrying its
                handler. They run as asynchronous tools: a delegation returns
                at once, and the backend's answer arrives later.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._instruction = instruction
        self._tools = [LLMExtensionTool(tool, cancel_on_interruption=False) for tool in tools]

    @property
    def system_instruction(self) -> str | None:
        """The pairing instruction."""
        return self._instruction

    def builtin_tools(
        self, registered: Mapping[str | None, "FunctionCallRegistryItem"]
    ) -> list[LLMExtensionTool]:
        """The delegation tools, whatever is registered.

        Args:
            registered: The service's registered tools by name.

        Returns:
            The tools.
        """
        return self._tools
