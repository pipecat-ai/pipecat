#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Sentry transactions SentryMetrics sends."""

from unittest.mock import MagicMock, patch

import pytest

sentry_sdk = pytest.importorskip("sentry_sdk")

from pipecat.processors.metrics import sentry as sentry_metrics  # noqa: E402
from pipecat.processors.metrics.sentry import SentryMetrics  # noqa: E402
from pipecat.utils.asyncio.task_manager import TaskManager  # noqa: E402


async def _ttfb_transactions(report_only_initial_ttfb: bool, measurements: int) -> MagicMock:
    """Measure TTFB a number of times and return the patched start_transaction."""
    with (
        patch.object(sentry_sdk, "is_initialized", return_value=True),
        patch.object(sentry_metrics, "_start_transaction") as start_transaction,
        patch.object(sentry_sdk, "flush"),
    ):
        metrics = SentryMetrics()
        metrics.set_processor_name("TestProcessor")
        await metrics.setup(TaskManager())
        for _ in range(measurements):
            await metrics.start_ttfb_metrics(report_only_initial_ttfb=report_only_initial_ttfb)
            await metrics.stop_ttfb_metrics()
        await metrics.cleanup()
    return start_transaction


@pytest.mark.asyncio
async def test_only_the_initial_ttfb_is_sent_when_only_the_initial_one_is_reported():
    start_transaction = await _ttfb_transactions(report_only_initial_ttfb=True, measurements=2)

    start_transaction.assert_called_once_with(op="ttfb", name="TTFB for TestProcessor")
    start_transaction.return_value.finish.assert_called_once()


@pytest.mark.asyncio
async def test_every_ttfb_is_sent_when_every_one_is_reported():
    start_transaction = await _ttfb_transactions(report_only_initial_ttfb=False, measurements=2)

    assert start_transaction.call_count == 2
    assert start_transaction.return_value.finish.call_count == 2


def test_start_transaction_uses_start_transaction_on_sentry_sdk_2():
    with patch.object(sentry_metrics, "sentry_sdk") as sdk:
        tx = sentry_metrics._start_transaction(op="ttfb", name="TTFB for TestProcessor")

    sdk.start_transaction.assert_called_once_with(op="ttfb", name="TTFB for TestProcessor")
    assert tx is sdk.start_transaction.return_value


def test_start_transaction_uses_an_inactive_span_on_sentry_sdk_3():
    sdk = MagicMock(spec=["start_span"])  # sentry-sdk 3 has no start_transaction
    with patch.object(sentry_metrics, "sentry_sdk", sdk):
        tx = sentry_metrics._start_transaction(op="ttfb", name="TTFB for TestProcessor")

    sdk.start_span.assert_called_once_with(
        name="TTFB for TestProcessor", attributes={"sentry.op": "ttfb"}, active=False
    )
    assert tx is sdk.start_span.return_value
