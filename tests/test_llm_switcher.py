#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for LLMSwitcher."""

import unittest

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.settings import LLMSettings


class _MockLLMService(LLMService):
    """Minimal LLM service for testing direct-function registration."""

    def __init__(self, **kwargs):
        settings = LLMSettings(
            model="test-model",
            system_instruction=None,
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=None,
            user_turn_completion_config=None,
        )
        super().__init__(settings=settings, **kwargs)


async def get_current_weather(params: FunctionCallParams, location: str):
    """Get the current weather.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
    """
    await params.result_callback({"conditions": "nice"})


@tool_options(cancel_on_interruption=False, timeout_secs=60)
async def end_call_handler(params: FunctionCallParams):
    """A classic handler carrying @tool_options call options."""
    await params.result_callback({"status": "ending"})


@tool_options(cancel_on_interruption=False, timeout_secs=60)
async def end_call(params: FunctionCallParams, reason: str):
    """End the call.

    Args:
        reason: Why the call is ending.
    """
    await params.result_callback({"status": "ending"})


class TestLLMSwitcherDirectFunctions(unittest.TestCase):
    """An LLMSwitcher must register context direct functions on every member LLM."""

    def test_sync_registered_tool_handlers_registers_handler(self):
        """LLMService._sync_registered_tool_handlers registers the handler."""
        llm = _MockLLMService()
        llm._sync_registered_tool_handlers(LLMContext(tools=[get_current_weather]).tools)
        self.assertIn("get_current_weather", llm._functions)

    def test_context_direct_functions_registered_on_all_member_llms(self):
        """A direct function advertised via the context registers on all members.

        Member LLMs sit behind per-branch filters, so at runtime only the active
        LLM receives the LLMContextFrame. The switcher must still register the
        direct-function handler on every member — active or not — so the tool
        keeps working after a service switch.
        """
        llm1 = _MockLLMService()
        llm2 = _MockLLMService()
        switcher = LLMSwitcher(llms=[llm1, llm2])

        switcher._sync_registered_tool_handlers(LLMContext(tools=[get_current_weather]).tools)

        for llm in (llm1, llm2):
            self.assertIn("get_current_weather", llm._functions)

    def test_register_direct_function_is_deprecated_but_fans_out(self):
        """The deprecated LLMSwitcher.register_direct_function still registers on all members."""
        llm1 = _MockLLMService()
        llm2 = _MockLLMService()
        switcher = LLMSwitcher(llms=[llm1, llm2])

        with self.assertWarns(DeprecationWarning):
            switcher.register_direct_function(get_current_weather)

        for llm in (llm1, llm2):
            self.assertIn("get_current_weather", llm._functions)


class TestLLMSwitcherRegisterFunctionOptionPrecedence(unittest.TestCase):
    """Explicit arg > @tool_options decorator > default, propagated to every member.

    The switcher forwards values to each member, which does the resolution; these
    check it forwards to all members — passing None when no explicit arg is given,
    so a member reads the decorator rather than a default that would clobber it.
    Covers both register_function and register_direct_function.
    """

    def _switcher(self):
        members = (_MockLLMService(), _MockLLMService())
        return LLMSwitcher(llms=list(members)), members

    def test_register_function_decorator_values_used_when_no_explicit_args(self):
        switcher, members = self._switcher()
        switcher.register_function("end_call", end_call_handler)  # decorated: False / 60
        for llm in members:
            item = llm._functions["end_call"]
            self.assertFalse(item.cancel_on_interruption)
            self.assertEqual(item.timeout_secs, 60)

    def test_register_function_explicit_arg_overrides_decorator(self):
        switcher, members = self._switcher()
        switcher.register_function("end_call", end_call_handler, cancel_on_interruption=True)
        for llm in members:
            item = llm._functions["end_call"]
            self.assertTrue(item.cancel_on_interruption)  # explicit wins
            self.assertEqual(item.timeout_secs, 60)  # decorator still applies

    def test_register_direct_function_decorator_values_used_when_no_explicit_args(self):
        # Regression: the switcher used to default cancel_on_interruption to True
        # and forward it as an explicit value, overriding the handler's @tool_options.
        switcher, members = self._switcher()
        with self.assertWarns(DeprecationWarning):
            switcher.register_direct_function(end_call)  # decorated: False / 60
        for llm in members:
            item = llm._functions["end_call"]
            self.assertFalse(item.cancel_on_interruption)
            self.assertEqual(item.timeout_secs, 60)

    def test_register_direct_function_explicit_arg_overrides_decorator(self):
        switcher, members = self._switcher()
        with self.assertWarns(DeprecationWarning):
            switcher.register_direct_function(end_call, cancel_on_interruption=True)
        for llm in members:
            item = llm._functions["end_call"]
            self.assertTrue(item.cancel_on_interruption)  # explicit wins
            self.assertEqual(item.timeout_secs, 60)  # decorator still applies


if __name__ == "__main__":
    unittest.main()
