#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for BackendLLMWorker, _BackendSession and _delegate_to_backend.

A scripted LLM service stands in for the backend model: each LLMContextFrame
plays the next scripted response (text and/or function calls), so the tests
exercise the real aggregators, tool loop and job plumbing under a
WorkerRunner. The attached contract (attach, message, cancel) is driven
through a _BackendSession; the run contract through _delegate_to_backend.
"""

import asyncio
from contextlib import aclosing
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.bus.messages import BusJobRequestMessage
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
)
from pipecat.pipeline.job_context import JobError
from pipecat.pipeline.job_decorator import job
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, FunctionCallParams, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.llm import BackendLLMWorker
from pipecat.workers.llm.backend_llm_worker import (
    BACKEND_JOB_NAME,
    CANCELLED_NOTE,
    BackendError,
    BackendIdle,
    BackendOutput,
    BackendToolCall,
    _BackendFinalOutput,
    _BackendSession,
    _delegate_to_backend,
    _render_transcript_request,
)
from pipecat.workers.runner import WorkerRunner


class _ScriptedLLM(LLMService):
    """Plays one scripted response per LLMContextFrame.

    A script step is ``("text", str)``, ``("thought", str)``,
    ``("error", str)`` or ``("call", name, call_id, args)``.
    ``settle_secs`` holds the response open after issuing its calls so a fast
    tool can return before the response ends.
    """

    def __init__(self, runs: list[list[tuple]], *, settle_secs: float = 0.0):
        super().__init__(
            settings=LLMSettings(
                model="scripted",
                system_instruction=None,
                temperature=None,
                max_tokens=None,
                top_p=None,
                top_k=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                filter_incomplete_user_turns=False,
                user_turn_completion_config=None,
            )
        )
        self._runs = list(runs)
        self._settle_secs = settle_secs
        self.contexts_seen: list[list[Any]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        self.contexts_seen.append(list(frame.context.get_messages()))
        steps = self._runs.pop(0) if self._runs else []
        await self.push_frame(LLMFullResponseStartFrame())
        calls = []
        for step in steps:
            if step[0] == "text":
                await self.push_frame(LLMTextFrame(step[1]))
            elif step[0] == "thought":
                await self.push_frame(LLMThoughtStartFrame())
                await self.push_frame(LLMThoughtTextFrame(step[1]))
                await self.push_frame(LLMThoughtEndFrame())
            elif step[0] == "error":
                await self.push_error(error_msg=step[1])
            else:
                _, name, call_id, args = step
                calls.append(
                    FunctionCallFromLLM(
                        context=frame.context,
                        tool_call_id=call_id,
                        function_name=name,
                        arguments=args,
                    )
                )
        if calls:
            await self.run_function_calls(calls)
            if self._settle_secs:
                await asyncio.sleep(self._settle_secs)
        await self.push_frame(LLMFullResponseEndFrame())


async def get_weather(params: FunctionCallParams, location: str):
    """Get the weather.

    Args:
        location: The city.
    """
    await params.result_callback({"temp": 62, "conditions": "rain"})


async def check_flight_status(params: FunctionCallParams, flight_number: str):
    """Check a flight's status.

    Args:
        flight_number: The flight number.
    """
    await params.result_callback({"status": "delayed", "departure_time": "14:30"})


async def raise_an_error(params: FunctionCallParams):
    """Fail."""
    raise RuntimeError("the tool broke")


async def book_taxi(params: FunctionCallParams, time: str):
    """Book a taxi.

    Args:
        time: The time to book it for.
    """
    await params.result_callback({"status": "done"})


async def _run_backend(
    llm: _ScriptedLLM,
    *,
    request: str = "Do it",
    transform_output=None,
    tools: list | None = None,
) -> tuple[str, list[BackendOutput], BackendLLMWorker]:
    """Run one delegation against ``llm`` under a WorkerRunner."""
    backend = BackendLLMWorker(
        llm=llm,
        name="backend",
        context=LLMContext(
            [{"role": "system", "content": "You are the backend."}], tools or [get_weather]
        ),
        transform_output=transform_output,
    )
    requester = BaseWorker("requester")
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(requester, backend)

    updates: list[BackendOutput] = []
    finals: list[BackendOutput] = []
    backend.tool_calls = []  # type: ignore[attr-defined]  # the BackendToolCall phases seen

    async def body():
        try:
            async for event in _delegate_to_backend(
                requester, "backend", request=request, timeout_secs=10
            ):
                if isinstance(event, BackendToolCall):
                    backend.tool_calls.append(event)  # type: ignore[attr-defined]
                elif isinstance(event, _BackendFinalOutput):
                    finals.append(event.output)
                    updates.append(event.output)
                else:
                    updates.append(event)
        finally:
            await runner.cancel()

    await asyncio.wait_for(asyncio.gather(runner.run(), body()), timeout=15)
    return (finals[-1].text if finals else ""), updates, backend


@pytest.mark.asyncio
async def test_backend_runs_a_tool_loop_and_streams_intermediate_responses():
    llm = _ScriptedLLM(
        [
            [("text", "Let me check."), ("call", "get_weather", "call_1", {"location": "Seattle"})],
            [("text", "It's 62 and raining in Seattle.")],
        ]
    )

    text, updates, backend = await _run_backend(
        llm,
        request=_render_transcript_request(
            [
                {"role": "user", "content": "what's the weather in seattle"},
                {"role": "assistant", "content": "Let me find out."},
            ],
            instruction="Task from the voice assistant: What's the weather in Seattle?",
        ),
    )

    assert text == "It's 62 and raining in Seattle."
    # The backend's own call is relayed phase by phase, for the frontend to report.
    assert [(c.phase, c.function_name, c.tool_call_id) for c in backend.tool_calls] == [
        ("started", "get_weather", "call_1"),
        ("in_progress", "get_weather", "call_1"),
        ("result", "get_weather", "call_1"),
    ]
    assert backend.tool_calls[1].arguments == {"location": "Seattle"}
    assert backend.tool_calls[2].result == {"temp": 62, "conditions": "rain"}
    # Only the answer is prefers_spoken; what the backend says on the way is not.
    assert updates == [
        BackendOutput(text="Let me check.", prefers_spoken=False),
        BackendOutput(text="It's 62 and raining in Seattle.", prefers_spoken=True),
    ]

    # The backend saw the rendered request first, then the tool result.
    first_request = llm.contexts_seen[0][-1]
    assert first_request["role"] == "user"
    assert first_request["content"] == (
        "Voice conversation so far:\n"
        "USER: what's the weather in seattle\n"
        "ASSISTANT: Let me find out.\n"
        "\n"
        "Task from the voice assistant: What's the weather in Seattle?"
    )
    assert len(llm.contexts_seen) == 2
    roles = [m.get("role") for m in backend.context.get_messages()]
    assert roles == ["system", "user", "assistant", "assistant", "tool", "assistant"]


@pytest.mark.asyncio
async def test_fast_tool_result_before_response_end_does_not_finish_the_run_early():
    llm = _ScriptedLLM(
        [
            [("text", "Checking."), ("call", "get_weather", "call_1", {"location": "Seattle"})],
            [("text", "Rain, 62 degrees.")],
        ],
        settle_secs=0.1,
    )

    text, updates, _ = await _run_backend(llm)

    assert text == "Rain, 62 degrees."
    assert [u.text for u in updates] == [
        "Checking.",
        "Rain, 62 degrees.",
    ]


@pytest.mark.asyncio
async def test_a_chained_request_finishes_on_the_last_round_not_an_earlier_one():
    llm = _ScriptedLLM(
        [
            [
                ("thought", "Check the flight first."),
                ("call", "check_flight_status", "call_1", {"flight_number": "AA100"}),
            ],
            [
                ("text", "It's delayed, so I'm booking a taxi for 12:30."),
                ("call", "book_taxi", "call_2", {"time": "12:30"}),
            ],
            [("text", "Taxi booked for 12:30.")],
        ]
    )

    text, updates, _ = await _run_backend(llm, tools=[check_flight_status, book_taxi])

    assert text == "Taxi booked for 12:30."
    assert len(llm.contexts_seen) == 3
    # What the backend says between rounds is progress; the last round's is the final output.
    assert [(u.text, u.is_thought) for u in updates] == [
        ("Check the flight first.", True),
        ("It's delayed, so I'm booking a taxi for 12:30.", False),
        ("Taxi booked for 12:30.", False),
    ]


@pytest.mark.asyncio
async def test_tool_only_response_sends_no_update_and_still_completes():
    llm = _ScriptedLLM(
        [
            [("call", "get_weather", "call_1", {"location": "Seattle"})],
            [("text", "It's raining.")],
        ]
    )

    text, updates, _ = await _run_backend(llm)

    assert text == "It's raining."
    # The tool-only response produces no text; only the final answer is sent.
    assert [u.text for u in updates] == ["It's raining."]


@pytest.mark.asyncio
async def test_a_delegation_the_requester_abandons_is_cancelled_and_the_next_starts_clean():
    """Closing the stream cancels the job; the backend stops and takes the next delegation."""
    started = asyncio.Event()

    async def slow_lookup(params: FunctionCallParams):
        """Look something up, slowly."""
        started.set()
        await asyncio.sleep(30)
        await params.result_callback({"never": "reached"})

    llm = _ScriptedLLM([[("call", "slow_lookup", "call_1", {})], [("text", "Second time round.")]])
    backend = BackendLLMWorker(llm=llm, name="backend", context=LLMContext(tools=[slow_lookup]))
    requester = BaseWorker("requester")
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(requester, backend)
    second: list[BackendOutput] = []

    async def body():
        try:
            async with aclosing(
                _delegate_to_backend(requester, "backend", request="First")
            ) as events:
                async for _ in events:
                    await asyncio.wait_for(started.wait(), 5)
                    break  # the requester loses interest mid-lookup
            for _ in range(50):
                if backend._run is None:
                    break
                await asyncio.sleep(0.1)
            assert backend._run is None
            async for event in _delegate_to_backend(requester, "backend", request="Second"):
                if isinstance(event, _BackendFinalOutput):
                    second.append(event.output)
        finally:
            await runner.cancel()

    await asyncio.wait_for(asyncio.gather(runner.run(), body()), timeout=15)
    assert [o.text for o in second] == ["Second time round."]


@pytest.mark.asyncio
async def test_a_backend_llm_error_fails_the_job():
    llm = _ScriptedLLM([[("error", "the provider is down")]])

    with pytest.raises(JobError, match="errored"):
        await _run_backend(llm)


@pytest.mark.asyncio
async def test_a_transform_that_raises_fails_the_job():
    llm = _ScriptedLLM([[("text", "Done.")]])

    async def broken(output: BackendOutput) -> BackendOutput:
        raise RuntimeError("boom")

    with pytest.raises(JobError, match="errored"):
        await _run_backend(llm, transform_output=broken)


@pytest.mark.asyncio
async def test_a_tool_handler_that_raises_leaves_the_delegation_running():
    llm = _ScriptedLLM(
        [
            [("call", "raise_an_error", "call_1", {})],
            [("text", "That did not work, sorry.")],
        ]
    )

    text, updates, _ = await _run_backend(llm, tools=[raise_an_error])

    assert text == "That did not work, sorry."
    assert [u.text for u in updates] == ["That did not work, sorry."]


@pytest.mark.asyncio
async def test_thoughts_are_streamed_as_thought_updates():
    llm = _ScriptedLLM(
        [
            [
                ("thought", "I should check the weather."),
                ("call", "get_weather", "call_1", {"location": "Seattle"}),
            ],
            [("thought", "Rain; keep it short."), ("text", "It's raining.")],
        ]
    )

    text, updates, _ = await _run_backend(llm)

    assert text == "It's raining."
    assert [(u.text, u.is_thought, u.prefers_spoken) for u in updates] == [
        ("I should check the weather.", True, False),
        ("Rain; keep it short.", True, False),
        ("It's raining.", False, True),
    ]


@pytest.mark.asyncio
async def test_follow_up_tasks_render_only_the_turns_since_the_last_one():
    llm = _ScriptedLLM([[("text", "First.")], [("text", "Second.")]])
    backend = BackendLLMWorker(llm=llm, name="backend", context=LLMContext())
    requester = BaseWorker("requester")
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(requester, backend)

    async def delegate(request: str):
        async for _ in _delegate_to_backend(requester, "backend", request=request):
            pass

    async def body():
        try:
            await delegate(
                _render_transcript_request([{"role": "user", "content": "one"}], first=True)
            )
            await delegate(
                _render_transcript_request([{"role": "user", "content": "two"}], first=False)
            )
        finally:
            await runner.cancel()

    await asyncio.wait_for(asyncio.gather(runner.run(), body()), timeout=15)

    requests = [m["content"] for m in llm.contexts_seen[1] if m.get("role") == "user"]
    assert requests[0].startswith("Voice conversation so far:\nUSER: one")
    assert requests[1].startswith("Voice conversation since the previous delegation:\nUSER: two")


def test_render_transcript_request_is_the_instruction_alone_when_nothing_was_said():
    assert _render_transcript_request([], instruction="Do it") == "Do it"


def test_render_transcript_request_points_the_backend_at_the_conversation():
    rendered = _render_transcript_request([{"role": "user", "content": "what's the weather"}])
    assert rendered == (
        "Voice conversation so far:\n"
        "USER: what's the weather\n"
        "\n"
        "Act on the user's most recent request in the conversation above."
    )


@pytest.mark.asyncio
async def test_a_task_less_job_runs_from_the_conversation_alone():
    llm = _ScriptedLLM([[("text", "It's raining.")]])

    text, _, _ = await _run_backend(
        llm,
        request=_render_transcript_request([{"role": "user", "content": "what's the weather"}]),
    )

    assert text == "It's raining."
    request = llm.contexts_seen[0][-1]
    assert request["content"].endswith(
        "Act on the user's most recent request in the conversation above."
    )


@pytest.mark.asyncio
async def test_transform_output_can_rewrite_text_and_speakability():
    llm = _ScriptedLLM(
        [
            [
                ("text", ">> Checking."),
                ("call", "get_weather", "call_1", {"location": "Seattle"}),
            ],
            [("text", "Internal note.")],
        ]
    )

    async def transform_output(output: BackendOutput) -> BackendOutput:
        if output.text.startswith(">>"):
            return replace(output, text=output.text[2:].lstrip(), prefers_spoken=True)
        return replace(output, prefers_spoken=False)

    _, updates, _ = await _run_backend(llm, transform_output=transform_output)

    assert [(u.text, u.prefers_spoken) for u in updates] == [
        ("Checking.", True),
        ("Internal note.", False),
    ]


@pytest.mark.asyncio
async def test_a_transform_returning_none_for_the_final_leaves_the_backend_with_nothing_to_say():
    llm = _ScriptedLLM([[("text", "Not for the user.")]])

    async def keep_it(output: BackendOutput) -> BackendOutput | None:
        return None

    text, updates, _ = await _run_backend(llm, transform_output=keep_it)

    assert text == ""
    assert [(u.text, u.prefers_spoken) for u in updates] == [("", False)]


@pytest.mark.asyncio
async def test_a_transform_that_empties_the_text_still_sends_the_output():
    llm = _ScriptedLLM(
        [[("text", "Checking."), ("call", "get_weather", "call_1", {})], [("text", "Done.")]]
    )

    async def empty_progress(output: BackendOutput) -> BackendOutput:
        return output if output.prefers_spoken else replace(output, text="")

    _, updates, _ = await _run_backend(llm, transform_output=empty_progress)

    assert [u.text for u in updates] == ["", "Done."]


@pytest.mark.asyncio
async def test_the_response_carries_the_transformed_final_output():
    """The final update and the return value are the same answer, transform included."""
    llm = _ScriptedLLM([[("text", "raw answer")]])

    async def transform_output(output: BackendOutput) -> BackendOutput:
        return replace(output, text=output.text.upper())

    text, updates, _ = await _run_backend(llm, transform_output=transform_output)

    assert [u.text for u in updates] == ["RAW ANSWER"]
    assert text == "RAW ANSWER"


@pytest.mark.asyncio
async def test_an_output_sent_with_nobody_listening_is_dropped():
    backend = BackendLLMWorker(llm=_ScriptedLLM([]), name="backend", context=LLMContext())
    backend.send_job_update = AsyncMock()  # type: ignore[method-assign]

    await backend.send_output(BackendOutput(text="Nobody is listening."))

    backend.send_job_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_updates_of_another_type_are_not_outputs():
    """The stream may carry update types the iterator does not know; those are skipped.

    The backend here is a plain worker speaking the job contract, which is
    also what a backend registered by name may be.
    """

    class _ContractBackend(BaseWorker):
        @job(name=BACKEND_JOB_NAME, sequential=True)
        async def run_delegation(self, message: BusJobRequestMessage):
            await self.send_job_update(message.job_id, {"type": "progress", "percent": 50})
            await self.send_job_response(message.job_id, BackendOutput(text="Done.").to_payload())

    requester = BaseWorker("requester")
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(requester, _ContractBackend("backend"))
    outputs: list = []

    async def body():
        try:
            async for output in _delegate_to_backend(requester, "backend", request="Do it"):
                outputs.append(output)
        finally:
            await runner.cancel()

    await asyncio.wait_for(asyncio.gather(runner.run(), body()), timeout=15)
    assert outputs == [_BackendFinalOutput(BackendOutput(text="Done."))]


def test_a_payload_names_its_type():
    assert BackendOutput(text="hello").to_payload()["type"] == "output"
    assert BackendToolCall("started", "f", "c1").to_payload()["type"] == "tool_call"


def test_a_tool_call_phase_survives_the_payload_round_trip():
    call = BackendToolCall("result", "get_weather", "c1", arguments={"a": 1}, result={"t": 62})
    assert BackendToolCall.from_payload(call.to_payload()) == call


def test_a_payload_leaves_the_flags_it_omits_at_their_defaults():
    """A sender that predates a flag should not decide its value."""
    assert BackendOutput.from_payload({"text": "hello"}) == BackendOutput(text="hello")


def test_a_payload_coerces_the_flags_it_carries():
    """Flags cross a bus, so what arrives may not be a bool."""
    rebuilt = BackendOutput.from_payload({"text": "hello", "is_thought": 1, "prefers_spoken": 0})
    assert (rebuilt.is_thought, rebuilt.prefers_spoken) == (True, False)


def test_render_transcript_request_flattens_what_a_transcript_can_hold():
    """A frontend can pass its context slice as-is; only spoken text survives."""
    rendered = _render_transcript_request(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": [{"type": "text", "text": "what is this"}]},
            {
                "role": "assistant",
                "content": "Let me look.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "look", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"seen": true}'},
            LLMSpecificMessage(llm="anthropic", message={"type": "thought", "text": "hmm"}),
        ],
    )
    assert rendered == (
        "Voice conversation so far:\n"
        "USER: what is this\n"
        "ASSISTANT: Let me look.\n"
        "\n"
        "Act on the user's most recent request in the conversation above."
    )


# ---------------------------------------------------------------------------
# The attached contract
# ---------------------------------------------------------------------------


def _attached_backend(
    llm: _ScriptedLLM, *, tools: list | None = None, transform_output=None
) -> tuple[BackendLLMWorker, BaseWorker, WorkerRunner]:
    """A backend and a requester under one runner, not yet running."""
    backend = BackendLLMWorker(
        llm=llm,
        name="backend",
        context=LLMContext(
            [{"role": "system", "content": "You are the backend."}], tools or [get_weather]
        ),
        transform_output=transform_output,
    )
    requester = BaseWorker("requester")
    runner = WorkerRunner(handle_sigint=False)
    return backend, requester, runner


async def _drive(runner: WorkerRunner, requester: BaseWorker, backend: BackendLLMWorker, body):
    """Run the runner and ``body`` together, cancelling the runner when the body is done."""
    await runner.add_workers(requester, backend)

    async def run_body():
        try:
            await body()
        finally:
            await runner.cancel()

    await asyncio.wait_for(asyncio.gather(runner.run(), run_body()), timeout=15)


async def _until_idle(session: _BackendSession) -> list:
    """Collect the session's events up to and including the next idle."""
    events = []
    async for event in session:
        events.append(event)
        if isinstance(event, BackendIdle):
            break
    return events


@pytest.mark.asyncio
async def test_an_attached_frontend_hears_the_backend_work_a_request_through():
    llm = _ScriptedLLM(
        [
            [("text", "Let me check."), ("call", "get_weather", "call_1", {"location": "Seattle"})],
            [("text", "It's 62 and raining in Seattle.")],
        ]
    )
    backend, requester, runner = _attached_backend(llm)
    events: list = []
    statuses: list[str] = []

    async def body():
        async with _BackendSession(requester, "backend") as session:
            statuses.append(await session.send("what's the weather in seattle"))
            events.extend(await _until_idle(session))
            assert session.capabilities == {
                "steering": True,
                "cancellation": True,
                "progress": True,
            }

    await _drive(runner, requester, backend, body)

    assert statuses == ["idle"]
    # A turn that called tools is narration; the one that did not is spoken.
    outputs = [e for e in events if isinstance(e, BackendOutput)]
    assert outputs == [
        BackendOutput(text="Let me check.", prefers_spoken=False),
        BackendOutput(text="It's 62 and raining in Seattle.", prefers_spoken=True),
    ]
    calls = [e for e in events if isinstance(e, BackendToolCall)]
    assert [(c.phase, c.function_name) for c in calls] == [
        ("started", "get_weather"),
        ("in_progress", "get_weather"),
        ("result", "get_weather"),
    ]
    assert isinstance(events[-1], BackendIdle)
    assert llm.contexts_seen[0][-1] == {"role": "user", "content": "what's the weather in seattle"}
    assert not backend.working


@pytest.mark.asyncio
async def test_a_message_sent_while_the_backend_works_joins_the_work():
    """The second request is taken up after the current step, with both in view."""
    lookup_started = asyncio.Event()
    lookup_may_finish = asyncio.Event()

    async def slow_lookup(params: FunctionCallParams):
        """Look something up, slowly."""
        lookup_started.set()
        await lookup_may_finish.wait()
        await params.result_callback({"found": True})

    llm = _ScriptedLLM(
        [
            [("call", "slow_lookup", "call_1", {})],
            [("text", "Done with both.")],
        ]
    )
    backend, requester, runner = _attached_backend(llm, tools=[slow_lookup])
    statuses: list[str] = []
    events: list = []

    async def body():
        async with _BackendSession(requester, "backend") as session:
            statuses.append(await session.send("First"))
            await asyncio.wait_for(lookup_started.wait(), 5)
            statuses.append(await session.send("Second"))
            lookup_may_finish.set()
            events.extend(await _until_idle(session))

    await _drive(runner, requester, backend, body)

    assert statuses == ["idle", "working"]
    # The run after the lookup saw both requests.
    users = [m["content"] for m in llm.contexts_seen[-1] if m.get("role") == "user"]
    assert users == ["First", "Second"]
    assert [e.text for e in events if isinstance(e, BackendOutput)] == ["Done with both."]


@pytest.mark.asyncio
async def test_cancel_stops_the_calls_in_flight_and_notes_the_cancellation():
    cancelled: list[str] = []
    started = asyncio.Event()

    async def slow_lookup(params: FunctionCallParams):
        """Look something up, slowly."""
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append("sync")
            raise

    @tool_options(cancel_on_interruption=False)
    async def slow_async_lookup(params: FunctionCallParams):
        """Look something up, slowly, surviving interruptions."""
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append("async")
            raise

    llm = _ScriptedLLM(
        [[("call", "slow_lookup", "call_1", {}), ("call", "slow_async_lookup", "call_2", {})]]
    )
    backend, requester, runner = _attached_backend(llm, tools=[slow_lookup, slow_async_lookup])
    results: list[bool] = []

    async def body():
        async with _BackendSession(requester, "backend") as session:
            await session.send("Look it up")
            await asyncio.wait_for(started.wait(), 5)
            results.append(await session.cancel("never mind"))
            for _ in range(50):
                if not backend.working:
                    break
                await asyncio.sleep(0.1)
            results.append(await session.cancel("nothing left"))

    await _drive(runner, requester, backend, body)

    assert results == [True, False]
    assert sorted(cancelled) == ["async", "sync"]
    assert not backend.working
    assert backend.context.get_messages()[-1] == {"role": "user", "content": CANCELLED_NOTE}


@pytest.mark.asyncio
async def test_closing_the_session_stops_the_backend_and_the_next_session_starts_clean():
    started = asyncio.Event()

    async def slow_lookup(params: FunctionCallParams):
        """Look something up, slowly."""
        started.set()
        await asyncio.sleep(30)
        await params.result_callback({"never": "reached"})

    llm = _ScriptedLLM([[("call", "slow_lookup", "call_1", {})], [("text", "Second time round.")]])
    backend, requester, runner = _attached_backend(llm, tools=[slow_lookup])
    second: list = []

    async def body():
        async with _BackendSession(requester, "backend") as session:
            await session.send("First")
            await asyncio.wait_for(started.wait(), 5)
        for _ in range(50):
            if backend._attached is None and not backend.working:
                break
            await asyncio.sleep(0.1)
        assert backend._attached is None
        async with _BackendSession(requester, "backend") as session:
            await session.send("Second")
            second.extend(await _until_idle(session))

    await _drive(runner, requester, backend, body)

    assert [e.text for e in second if isinstance(e, BackendOutput)] == ["Second time round."]


@pytest.mark.asyncio
async def test_a_second_frontend_cannot_attach_and_a_message_needs_an_attached_frontend():
    llm = _ScriptedLLM([[("text", "Hello.")]])
    backend, requester, runner = _attached_backend(llm)
    other = BaseWorker("other")

    async def body():
        with pytest.raises(JobError, match="no frontend is attached"):
            await _BackendSession(requester, "backend").send("Anyone there?")
        async with _BackendSession(requester, "backend") as session:
            await session.send("Hello")
            await _until_idle(session)
            async with _BackendSession(other, "backend") as second:
                with pytest.raises(JobError, match="already attached"):
                    await second.__anext__()
            with pytest.raises(JobError):
                async for _ in _delegate_to_backend(requester, "backend", request="Run it"):
                    pass

    await runner.add_workers(other)
    await _drive(runner, requester, backend, body)


@pytest.mark.asyncio
async def test_a_backend_error_reaches_the_frontend_and_the_stream_goes_on():
    llm = _ScriptedLLM([[("error", "the provider is down")], [("text", "Back again.")]])
    backend, requester, runner = _attached_backend(llm)
    events: list = []

    async def body():
        async with _BackendSession(requester, "backend") as session:
            await session.send("First")
            async for event in session:
                events.append(event)
                if isinstance(event, BackendError):
                    break
            await session.send("Second")
            events.extend(await _until_idle(session))

    await _drive(runner, requester, backend, body)

    # The empty turn ends, and so goes idle, before the error frame reaches the worker.
    assert BackendError(error="the provider is down") in events
    assert [e.text for e in events if isinstance(e, BackendOutput)] == ["Back again."]


@pytest.mark.asyncio
async def test_an_attached_frontend_hears_thoughts_and_the_apps_own_outputs():
    llm = _ScriptedLLM(
        [
            [
                ("thought", "Check the weather first."),
                ("call", "report", "call_1", {"text": "Halfway there."}),
            ],
            [("text", "Done.")],
        ]
    )

    async def report(params: FunctionCallParams, text: str):
        """Tell the user something.

        Args:
            text: What to tell them.
        """
        await backend.send_output(BackendOutput(text=text))
        await backend.send_output(BackendOutput(text="shaped"), apply_transform_output=True)
        await params.result_callback("told")

    async def shout(output: BackendOutput) -> BackendOutput | None:
        if output.is_thought:
            return None
        return replace(output, text=output.text.upper())

    backend, requester, runner = _attached_backend(llm, tools=[report], transform_output=shout)
    events: list = []

    async def body():
        async with _BackendSession(requester, "backend") as session:
            await session.send("Do it")
            events.extend(await _until_idle(session))

    await _drive(runner, requester, backend, body)

    # The thought was dropped by the transform; the app's outputs pass it unless asked.
    assert [
        (e.text, e.is_thought, e.prefers_spoken) for e in events if isinstance(e, BackendOutput)
    ] == [
        ("Halfway there.", False, True),
        ("SHAPED", False, True),
        ("DONE.", False, True),
    ]


@pytest.mark.asyncio
async def test_the_session_ignores_stream_updates_it_does_not_know():
    """A backend speaking the contract may send update types the session does not know."""

    class _ContractBackend(BaseWorker):
        @job(name="attach")
        async def attach_frontend(self, message: BusJobRequestMessage):
            await self.send_job_update(message.job_id, {"type": "progress", "percent": 50})
            await self.send_job_update(message.job_id, BackendOutput(text="Done.").to_payload())
            await self.send_job_update(message.job_id, {"type": "idle"})
            await asyncio.Event().wait()

    requester = BaseWorker("requester")
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(requester, _ContractBackend("backend"))
    events: list = []

    async def body():
        try:
            async with _BackendSession(requester, "backend") as session:
                events.extend(await _until_idle(session))
        finally:
            await runner.cancel()

    await asyncio.wait_for(asyncio.gather(runner.run(), body()), timeout=15)
    assert events == [BackendOutput(text="Done."), BackendIdle()]
