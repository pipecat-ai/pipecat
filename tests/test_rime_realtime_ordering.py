#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Application history edits precede their dependent native Prism turns."""

import pytest

pytest.importorskip("rimelabs_sdk")

from pipecat.frames.frames import (
    LLMMessagesAppendFrame,
    LLMMessagesTransformFrame,
    LLMMessagesUpdateFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.rime.realtime.llm import RimeReplyFrame
from tests.rime_realtime_peer import conversation
from tests.test_rime_realtime_service import answer, sent


@pytest.mark.asyncio
@pytest.mark.parametrize("edit_kind", ["append", "update", "transform"])
async def test_native_turn_imports_preceding_deferred_history_edit(edit_kind):
    original = [{"role": "user", "content": "Original history"}]
    new = [{"role": "user", "content": "Application edit"}]
    async with conversation(context=LLMContext(original)) as (peer, service, worker, *_):
        if edit_kind == "append":
            frame = LLMMessagesAppendFrame(messages=new, run_llm=False)
            expected = [*original, *new]
        elif edit_kind == "update":
            frame = LLMMessagesUpdateFrame(messages=new, run_llm=False)
            expected = new
        else:
            frame = LLMMessagesTransformFrame(transform=lambda _: new, run_llm=False)
            expected = new
        await worker.queue_frames(
            [frame, RimeReplyFrame(text="Native question", response_metadata={"case": edit_kind})]
        )
        request = await peer.next()
        assert request["response"]["prism_input_text"] == "Native question"
        assert request["response"]["metadata"]["case"] == edit_kind
        assert [item["text"] for item in peer.history] == [
            message["content"] for message in expected
        ]
        await answer(peer, request)
        assert len(sent(peer, "response.create")) == 2
        assert service.is_usable


def test_native_text_and_proactive_instruction_are_separate_intents():
    with pytest.raises(ValueError, match="separate operations"):
        RimeReplyFrame(text="User turn", instruction="Proactive instruction")
