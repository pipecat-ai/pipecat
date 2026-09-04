#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS Nova Sonic reports token usage from its ``usageEvent`` stream events.

Nova Sonic emits a ``usageEvent`` whose ``details.delta`` carries the tokens
consumed since the previous event, split into speech/text buckets for input and
output. The service collapses those buckets into the modality-agnostic
``LLMTokenUsage`` (``prompt_tokens`` / ``completion_tokens``) and reports the
*delta* — not the cumulative ``details.total`` — so usage stays incremental per
event, matching the other speech-to-speech services.

The service is imported with ``pytest.importorskip`` so the suite is skipped
rather than failing collection when the optional AWS dependencies aren't
installed.
"""

import unittest
from unittest.mock import AsyncMock

import pytest

from pipecat.metrics.metrics import LLMTokenUsage


class TestAWSNovaSonicUsageMetrics(unittest.IsolatedAsyncioTestCase):
    def _service(self):
        mod = pytest.importorskip("pipecat.services.aws.nova_sonic.llm")
        return mod.AWSNovaSonicLLMService(
            secret_access_key="test", access_key_id="test", region="us-east-1"
        )

    def test_can_generate_metrics(self):
        # Without this, start_llm_usage_metrics is a no-op and nothing is emitted.
        service = self._service()
        self.assertTrue(service.can_generate_metrics())

    async def test_usage_event_reports_collapsed_delta(self):
        service = self._service()
        service.start_llm_usage_metrics = AsyncMock()

        # A real usageEvent: delta is the increment for this event, total is the
        # cumulative session count. We report the delta, collapsing speech + text.
        await service._handle_usage_event(
            {
                "usageEvent": {
                    "details": {
                        "delta": {
                            "input": {"speechTokens": 0, "textTokens": 3},
                            "output": {"speechTokens": 20, "textTokens": 0},
                        },
                        "total": {
                            "input": {"speechTokens": 288, "textTokens": 3443},
                            "output": {"speechTokens": 694, "textTokens": 203},
                        },
                    },
                }
            }
        )

        service.start_llm_usage_metrics.assert_awaited_once()
        (tokens,) = service.start_llm_usage_metrics.await_args.args
        self.assertIsInstance(tokens, LLMTokenUsage)
        self.assertEqual(tokens.prompt_tokens, 3)  # 0 speech + 3 text
        self.assertEqual(tokens.completion_tokens, 20)  # 20 speech + 0 text
        self.assertEqual(tokens.total_tokens, 23)

    async def test_usage_event_with_no_tokens_is_skipped(self):
        # A zero-token delta (e.g. an event carrying no new usage) must not emit
        # an empty metrics frame.
        service = self._service()
        service.start_llm_usage_metrics = AsyncMock()

        await service._handle_usage_event(
            {
                "usageEvent": {
                    "details": {
                        "delta": {
                            "input": {"speechTokens": 0, "textTokens": 0},
                            "output": {"speechTokens": 0, "textTokens": 0},
                        },
                    },
                }
            }
        )

        service.start_llm_usage_metrics.assert_not_awaited()

    async def test_usage_event_with_missing_details_is_safe(self):
        # A malformed/partial usageEvent must not raise and must not emit.
        service = self._service()
        service.start_llm_usage_metrics = AsyncMock()

        await service._handle_usage_event({"usageEvent": {}})

        service.start_llm_usage_metrics.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
