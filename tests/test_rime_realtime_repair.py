#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""History repair outcomes through the pipeline and published SDK."""

import asyncio

import pytest

pytest.importorskip("rimelabs_sdk")

from pipecat.frames.frames import (
    ErrorFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMRunFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.rime.realtime.llm import PrismHistoryUpdateError
from tests.rime_realtime_peer import conversation, until
from tests.test_rime_realtime_service import answer, sent


@pytest.mark.asyncio
async def test_response_during_repair_read_makes_order_uncertain():
    context = LLMContext([{"role": "user", "content": "Original"}])
    async with conversation(context=context) as (
        peer,
        service,
        worker,
        _,
        __,
        upstream,
        downstream,
    ):
        gate = peer.gates["prism.conversation.get"] = asyncio.Event()
        peer.history_failure_at = peer.history_mutations + 2
        await worker.queue_frame(
            LLMMessagesUpdateFrame(
                messages=[{"role": "user", "content": "Replacement"}], run_llm=True
            )
        )
        await until(lambda: bool(sent(peer, "prism.conversation.get")))
        response = await peer.start()
        await until(
            lambda: any(
                isinstance(frame, LLMFullResponseStartFrame)
                and frame.metadata["rime"]["response_id"] == response
                for frame in downstream
            )
        )
        await peer.end(response)
        await until(
            lambda: sum(isinstance(frame, LLMFullResponseEndFrame) for frame in downstream) == 2
        )
        gate.set()
        await until(lambda: any(isinstance(frame, ErrorFrame) for frame in upstream))
        error = next(frame.exception for frame in upstream if isinstance(frame, ErrorFrame))
        assert isinstance(error, PrismHistoryUpdateError)
        assert error.status == "uncertain"
        assert error.confirmed_context is None
        assert not error.recoverable
        assert not service.is_usable
        assert len(sent(peer, "prism.conversation.get")) == 1
        assert len(sent(peer, "response.create")) == 1


@pytest.mark.asyncio
async def test_refused_repair_read_cannot_report_recoverable_partial_state():
    context = LLMContext([{"role": "user", "content": "Original"}])
    async with conversation(context=context) as (peer, service, worker, _, __, upstream, ___):
        peer.history_failure_at = peer.history_mutations + 2
        peer.refuse["prism.conversation.get"] = "invalid_request"
        await worker.queue_frame(
            LLMMessagesUpdateFrame(
                messages=[{"role": "user", "content": "Replacement"}], run_llm=True
            )
        )
        await until(lambda: any(isinstance(frame, ErrorFrame) for frame in upstream))
        error = next(frame.exception for frame in upstream if isinstance(frame, ErrorFrame))
        assert isinstance(error, PrismHistoryUpdateError)
        assert error.status == "uncertain"
        assert error.confirmed_context is None
        assert not error.recoverable
        assert not service.is_usable
        assert len(sent(peer, "prism.conversation.get")) == 1
        assert len(sent(peer, "conversation.item.delete")) == 1
        assert len(sent(peer, "response.create")) == 1


@pytest.mark.asyncio
async def test_partial_multi_item_import_projects_only_acknowledged_items():
    async with conversation() as (peer, service, worker, _, __, upstream, ___):
        peer.history_failure_at = peer.history_mutations + 2
        messages = [
            {
                "role": "assistant",
                "content": "Accepted text",
                "tool_calls": [
                    {
                        "id": "imported-call",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "imported-call", "content": "Not imported"},
        ]
        await worker.queue_frame(LLMMessagesAppendFrame(messages=messages, run_llm=True))
        await until(lambda: any(isinstance(frame, ErrorFrame) for frame in upstream))
        error = next(frame.exception for frame in upstream if isinstance(frame, ErrorFrame))
        assert isinstance(error, PrismHistoryUpdateError)
        assert error.status == "partial"
        assert error.recoverable
        assert error.confirmed_context == [{"role": "assistant", "content": "Accepted text"}]
        assert service.is_usable
        assert len(sent(peer, "prism.conversation.get")) == 1
        assert len(sent(peer, "response.create")) == 1
        error.confirmed_context[0]["content"] = "Mutated detached projection"
        await worker.queue_frame(LLMRunFrame())
        await answer(peer)
        assert len(sent(peer, "conversation.item.create")) == 2
        assert peer.history[0]["text"] == "Accepted text"
