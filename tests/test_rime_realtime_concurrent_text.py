#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Native typed turn ownership while provider admission is delayed."""

import asyncio

import pytest

pytest.importorskip("rimelabs_sdk")

from pipecat.frames.frames import (
    ErrorFrame,
    InputTextRawFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMSetToolChoiceFrame,
)
from tests.rime_realtime_peer import conversation, until
from tests.test_rime_realtime_service import answer, sent


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_choice", ["auto", "none"])
async def test_queued_native_turns_keep_text_and_provider_identity(tool_choice):
    async with conversation() as (peer, service, worker, context, _, upstream, downstream):
        await worker.queue_frame(LLMSetToolChoiceFrame(tool_choice=tool_choice))
        await worker.queue_frame(InputTextRawFrame(text="First invoice 314"))
        first_request = await peer.next()
        await worker.queue_frame(InputTextRawFrame(text="Second invoice 271"))
        await asyncio.sleep(0.03)
        assert len(sent(peer, "response.create")) == 2
        assert context.get_messages() == []

        first_response = await answer(peer, first_request, text="First answer", samples=0)
        second_request = await peer.next()
        assert second_request["response"]["prism_input_text"] == "Second invoice 271"
        second_response = await answer(peer, second_request, text="Second answer", samples=0)
        await until(lambda: len(context.get_messages()) == 4)

        assert context.get_messages() == [
            {"role": "user", "content": "First invoice 314"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second invoice 271"},
            {"role": "assistant", "content": "Second answer"},
        ]
        provider_inputs = {item["id"]: item["text"] for item in peer.history}
        for response, text in (
            (first_response, "First invoice 314"),
            (second_response, "Second invoice 271"),
        ):
            frame = next(
                frame
                for frame in downstream
                if isinstance(frame, LLMFullResponseStartFrame)
                and frame.metadata["rime"]["response_id"] == response
            )
            (input_id,) = frame.metadata["rime"]["input_item_ids"]
            assert provider_inputs[input_id] == text
        assert not sent(peer, "conversation.item.create")
        assert not any(isinstance(frame, ErrorFrame) for frame in upstream)
        assert service.is_usable


@pytest.mark.asyncio
async def test_interruption_drops_queued_turn_without_overwriting_late_accepted_text():
    async with conversation() as (peer, service, worker, context, output, upstream, _):
        await worker.queue_frame(InputTextRawFrame(text="Accepted before interruption"))
        first_request = await peer.next()
        await worker.queue_frame(InputTextRawFrame(text="Queued before interruption"))
        await asyncio.sleep(0.03)
        await worker.queue_frame(InterruptionFrame())
        await asyncio.sleep(0.03)
        first_response = await peer.start(first_request)
        await until(
            lambda: any(
                event["response_id"] == first_response for event in sent(peer, "response.cancel")
            )
        )
        await until(lambda: len(context.get_messages()) == 1)
        assert context.get_messages() == [
            {"role": "user", "content": "Accepted before interruption"}
        ]
        await worker.queue_frame(InputTextRawFrame(text="Fresh turn after interruption"))
        request = await peer.next()
        assert request["response"]["prism_input_text"] == "Fresh turn after interruption"
        await answer(peer, request, text="Fresh answer", samples=0)
        await until(lambda: len(context.get_messages()) == 3)
        assert context.get_messages()[1] == {
            "role": "user",
            "content": "Fresh turn after interruption",
        }
        assert len(sent(peer, "response.create")) == 3
        assert not output.writes
        assert not any(isinstance(frame, ErrorFrame) for frame in upstream)
        assert service.is_usable
