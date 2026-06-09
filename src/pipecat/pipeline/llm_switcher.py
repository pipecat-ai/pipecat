#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM switcher for switching between different LLMs at runtime, with different switching strategies."""

import warnings
from typing import Any, cast

from pipecat.adapters.schemas.direct_function import DirectFunction
from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.pipeline.service_switcher import (
    ServiceSwitcher,
    ServiceSwitcherStrategyManual,
    StrategyType,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService


class LLMSwitcher(ServiceSwitcher[StrategyType]):
    """A pipeline that switches between different LLMs at runtime.

    Example::

        llm_switcher = LLMSwitcher(llms=[openai_llm, anthropic_llm])
    """

    def __init__(
        self,
        llms: list[LLMService],
        strategy_type: type[StrategyType] = ServiceSwitcherStrategyManual,
    ):
        """Initialize the service switcher with a list of LLMs and a switching strategy.

        Args:
            llms: List of LLM services to switch between.
            strategy_type: The strategy class to use for switching between LLMs.
                Defaults to ``ServiceSwitcherStrategyManual``.
        """
        super().__init__(cast(list[FrameProcessor], llms), strategy_type)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame, syncing context direct functions on all member LLMs.

        On an ``LLMContextFrame``, the direct functions advertised in the context
        are synced on every member LLM — active or not — so that direct functions
        listed via ``LLMContext(tools=[...])`` keep working across service switches.

        This is needed because member LLMs sit behind per-branch filters: only the
        active LLM receives the context frame and would otherwise sync its handlers,
        leaving inactive LLMs out of step with the advertised tools.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            self._sync_registered_direct_functions(frame.context.tools)

    def _sync_registered_direct_functions(self, tools) -> None:
        """Sync the context's direct functions on every member LLM.

        Args:
            tools: The advertised tools whose direct-function handlers should be
                synced on all member LLMs.
        """
        for llm in self.llms:
            llm._sync_registered_direct_functions(tools)

    @property
    def llms(self) -> list[LLMService]:
        """Get the list of LLMs managed by this switcher.

        Returns:
            List of LLM services managed by this switcher.
        """
        return cast(list[LLMService], self.services)

    @property
    def active_llm(self) -> LLMService:
        """Get the currently active LLM.

        Returns:
            The currently active LLM service, or None if no LLM is active.
        """
        return cast(LLMService, self.strategy.active_service)

    async def run_inference(self, context: LLMContext, **kwargs) -> str | None:
        """Run a one-shot, out-of-band (i.e. out-of-pipeline) inference with the given LLM context, using the currently active LLM.

        Args:
            context: The LLM context containing conversation history.
            **kwargs: Additional arguments forwarded to the active LLM's run_inference
                (e.g. max_tokens, system_instruction).

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        if self.active_llm:
            return await self.active_llm.run_inference(context=context, **kwargs)
        return None

    def register_function(
        self,
        function_name: str | None,
        handler: Any,
        *,
        cancel_on_interruption: bool = True,
        timeout_secs: float | None = None,
    ):
        """Register a function handler for LLM function calls, on all LLMs, active or not.

        Args:
            function_name: The name of the function to handle. Use None to handle
                all function calls with a catch-all handler.
            handler: The function handler. Should accept a single FunctionCallParams
                parameter.
            cancel_on_interruption: Whether to cancel this function call when an
                interruption occurs. Defaults to True.
            timeout_secs: Optional timeout in seconds for the function call.
        """
        for llm in self.llms:
            llm.register_function(
                function_name=function_name,
                handler=handler,
                cancel_on_interruption=cancel_on_interruption,
                timeout_secs=timeout_secs,
            )

    def register_direct_function(
        self,
        handler: DirectFunction,
        *,
        cancel_on_interruption: bool = True,
        timeout_secs: float | None = None,
    ):
        """Register a direct function handler for LLM function calls, on all LLMs, active or not.

        .. deprecated:: 1.4.0
            Direct functions advertised via ``LLMContext(tools=[...])`` are now
            registered on every member LLM automatically. List them in the
            context for tools available at session start, or push an
            ``LLMSetToolsFrame`` to change tools mid-session. This method will be
            removed in a future version.

        Args:
            handler: The direct function to register. Must follow DirectFunction protocol.
            cancel_on_interruption: Whether to cancel this function call when an
                interruption occurs. Defaults to True.
            timeout_secs: Optional timeout in seconds for the function call.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "`LLMSwitcher.register_direct_function()` is deprecated since 1.4.0. "
                "List direct functions in `LLMContext(tools=[...])` (registered on "
                "all member LLMs automatically), or push an `LLMSetToolsFrame` to "
                "change tools mid-session.",
                DeprecationWarning,
                stacklevel=2,
            )
        for llm in self.llms:
            llm._register_direct_function(
                handler=handler,
                cancel_on_interruption=cancel_on_interruption,
                timeout_secs=timeout_secs,
            )
