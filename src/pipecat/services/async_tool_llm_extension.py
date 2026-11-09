#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The LLM extension that teaches the model about its asynchronous tools and lets it cancel them."""

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from loguru import logger

from pipecat.frames.frames import FunctionCallResultProperties
from pipecat.processors.aggregators.async_tool_messages import ASYNC_TOOL_INSTRUCTIONS
from pipecat.services.base_llm_extension import BaseLLMExtension, LLMExtensionTool
from pipecat.utils.async_tool_cancellation import (
    ASYNC_TOOL_CANCELLATION_INSTRUCTIONS,
    CANCEL_TOOL_PREFIX,
    build_cancel_tool_schema,
    cancel_tool_name,
)

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams, FunctionCallRegistryItem


class AsyncToolLLMExtension(BaseLLMExtension):
    """Teaches the model about its asynchronous tools, and gives it a cancel tool for each it may stop.

    An asynchronous tool (``cancel_on_interruption=False``) keeps running after
    the model has replied, and its result arrives later. While any is
    registered, the extension adds the guidance on such tools to the system
    instruction. A tool registered with ``cancellable_by_llm=True`` gets a
    built-in cancel tool named for it (``cancel_<name>``), with the guidance on
    cancelling, so the model can drop work whose result is no longer wanted.

    Every :class:`~pipecat.services.llm_service.LLMService` has one, from
    construction; it contributes nothing until a tool is registered that
    calls for it, and follows the registered tools as they come and go.
    """

    installed_by = "LLMService"
    # After the extensions an application attaches, whose instructions come
    # first, as the tool guidance closes the system instruction.
    order = 1

    def __init__(self, **kwargs):
        """Initialize the extension.

        Args:
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._has_async_tools = False
        # The cancel tools advertised, by name.
        self._cancel_tool_names: set[str] = set()
        # Cancel-tool names that collide with a user's own tool, warned about once.
        self._warned_collisions: set[str] = set()

    @property
    def system_instruction(self) -> str | None:
        """The cancellation guidance while a cancel tool is advertised, and the async-tool guidance while an async tool is registered."""
        parts = []
        if self._cancel_tool_names:
            parts.append(ASYNC_TOOL_CANCELLATION_INSTRUCTIONS)
        if self._has_async_tools:
            parts.append(ASYNC_TOOL_INSTRUCTIONS)
        return "\n\n".join(parts) or None

    def builtin_tools(
        self, registered: Mapping[str | None, "FunctionCallRegistryItem"]
    ) -> list[LLMExtensionTool]:
        """One cancel tool per cancellable tool, named for it.

        Reconciled from a single reading of the registry, so the advertised set
        can never drift from what is actually cancellable. A tool that never
        opted in has no cancel tool at all, so there is nothing for the model
        to call against it.

        Args:
            registered: The service's registered tools by name.

        Returns:
            The cancel tools.
        """
        self._has_async_tools = any(
            name not in self._cancel_tool_names and not item.cancel_on_interruption
            for name, item in registered.items()
        )
        wanted = {
            cancel_tool_name(name): name
            for name, item in registered.items()
            if name and item.cancellable_by_llm
        }
        tools = []
        for name, target in wanted.items():
            if name in registered and name not in self._cancel_tool_names:
                # Once per name: this runs before every inference, and the
                # collision persists for as long as both tools are registered.
                if name not in self._warned_collisions:
                    self._warned_collisions.add(name)
                    logger.warning(
                        f"{self}: '{target}' is cancellable but its cancel tool can't be "
                        f"advertised — '{name}' is already registered as a tool of its own. "
                        f"Rename one of them, or the LLM has no way to stop {target}."
                    )
                continue
            tools.append(LLMExtensionTool(build_cancel_tool_schema(target, self._cancel)))
        self._cancel_tool_names = {tool.schema.name for tool in tools}
        return tools

    async def _cancel(self, params: "FunctionCallParams"):
        """Handle a ``cancel_<tool>`` call from the model.

        Which work to stop comes from the tool the model called, so the common
        case takes no arguments at all. ``tool_call_id`` only has to be given to
        choose between several running calls of the same tool.
        """
        target = (params.function_name or "")[len(CANCEL_TOOL_PREFIX) :]
        tool_call_id: str | None = params.arguments.get("tool_call_id")

        async def refuse(reason: str, **extra: Any) -> None:
            logger.debug(f"{self}: declining to cancel {target} — {reason}")
            await params.result_callback(
                {"cancelled": None, "reason": reason, **extra},
                properties=FunctionCallResultProperties(run_llm=True),
            )

        candidates = [
            call
            for call in params.llm.running_function_calls()
            if call.function_name == target and call.registry_item.cancellable_by_llm
        ]
        if not candidates:
            await refuse(f"no {target} call is running")
            return
        if tool_call_id:
            match = next((c for c in candidates if c.tool_call_id == tool_call_id), None)
            if match is None:
                await refuse(f"no running {target} call has that tool_call_id")
                return
        elif len(candidates) > 1:
            # Several calls of the one tool: the model has to say which, so the
            # refusal carries the ids to say it with. Spell the choices out in
            # the reason itself: a model reading a refusal acts on what it says
            # far more readily than on a field beside it.
            choices = "; ".join(
                f"{c.tool_call_id} (called with {json.dumps(c.arguments, default=str)})"
                for c in candidates
            )
            await refuse(
                f"{len(candidates)} {target} calls are running: {choices}. Call "
                f"{params.function_name} again with tool_call_id set to the one to stop.",
                running=[
                    {"tool_call_id": c.tool_call_id, "arguments": c.arguments} for c in candidates
                ],
            )
            return
        else:
            match = candidates[0]

        logger.debug(f"{self}: cancelling [{match.function_name}:{match.tool_call_id}]")
        await params.llm.cancel_function_call(match.tool_call_id, reason="User request")
        await params.result_callback(
            {"cancelled": match.tool_call_id, "function_name": match.function_name},
            properties=FunctionCallResultProperties(run_llm=True),
        )
