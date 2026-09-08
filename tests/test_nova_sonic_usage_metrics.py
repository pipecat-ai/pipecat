#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS Nova Sonic reports token usage from its ``usageEvent`` stream events.

Nova Sonic emits a ``usageEvent`` whose ``details.delta`` carries the tokens
consumed since the previous event, split into speech/text buckets for input and
output. The service reports combined ``prompt_tokens`` / ``completion_tokens``
alongside their audio subsets in ``LLMTokenUsage``. It reports the *delta* — not
the cumulative ``details.total`` — so usage stays incremental per event,
matching the other speech-to-speech services.

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

    async def test_usage_event_reports_delta_with_audio_breakdown(self):
        service = self._service()
        service.start_llm_usage_metrics = AsyncMock()

        # A real usageEvent: delta is the increment for this event, total is the
        # cumulative session count. Only the incremental counts are emitted.
        await service._handle_usage_event(
            {
                "usageEvent": {
                    "details": {
                        "delta": {
                            "input": {"speechTokens": 12, "textTokens": 3},
                            "output": {"speechTokens": 20, "textTokens": 4},
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
        self.assertEqual(tokens.prompt_tokens, 15)
        self.assertEqual(tokens.completion_tokens, 24)
        self.assertEqual(tokens.total_tokens, 39)
        self.assertEqual(tokens.input_audio_tokens, 12)
        self.assertEqual(tokens.output_audio_tokens, 20)

    async def test_audio_counts_distinguish_unreported_and_zero(self):
        for audio_tokens in (None, 0):
            with self.subTest(audio_tokens=audio_tokens):
                service = self._service()
                service.start_llm_usage_metrics = AsyncMock()
                input_tokens = {"textTokens": 3}
                output_tokens = {"textTokens": 4}
                if audio_tokens is not None:
                    input_tokens["speechTokens"] = audio_tokens
                    output_tokens["speechTokens"] = audio_tokens

                await service._handle_usage_event(
                    {
                        "usageEvent": {
                            "details": {"delta": {"input": input_tokens, "output": output_tokens}}
                        }
                    }
                )

                service.start_llm_usage_metrics.assert_awaited_once()
                (tokens,) = service.start_llm_usage_metrics.await_args.args
                self.assertEqual(tokens.prompt_tokens, 3)
                self.assertEqual(tokens.completion_tokens, 4)
                self.assertEqual(tokens.total_tokens, 7)
                self.assertEqual(tokens.input_audio_tokens, audio_tokens)
                self.assertEqual(tokens.output_audio_tokens, audio_tokens)

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
