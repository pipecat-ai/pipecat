#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for BackendLLMWorker and _BackendSession.

A scripted LLM service stands in for the backend model: each LLMContextFrame
plays the next scripted response (text and/or function calls), so the tests
exercise the real aggregators, tool loop and job plumbing under a
WorkerRunner, driven through a _BackendSession as a frontend drives them.
"""

import asyncio
from dataclasses import replace
from typing import Any

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
from pipecat.services.llm_service import (
    FunctionCallFromLLM,
    FunctionCallParams,
    LLMService,
)
from pipecat.services.settings import LLMSettings
from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.llm import BackendLLMWorker
from pipecat.workers.llm.backend_llm_worker import (
    BACKEND_PAIRING_INSTRUCTION,
    SPOKEN_MARK,
    BackendError,
    BackendIdle,
    BackendOutput,
    BackendToolCall,
    _BackendSession,
    _render_explicit_request,
    _render_transcript_request,
    _split_spoken,
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


def test_render_transcript_request_is_the_instruction_alone_when_nothing_was_said():
    assert _render_transcript_request([], instruction="Do it") == "Do it"


def test_a_mark_on_a_later_line_is_not_spoken_as_text():
    note, spoken = _split_spoken("Found it.\n\n>> The test is fixed.\n>> All 42 pass.")
    assert (note, spoken) == ("Found it.", "The test is fixed.\nAll 42 pass.")


def test_render_explicit_request_puts_the_instruction_after_the_request():
    assert _render_explicit_request("Book a taxi", instruction="Do it") == "Book a taxi\n\nDo it"


def test_render_transcript_request_points_the_backend_at_the_conversation():
    rendered = _render_transcript_request([{"role": "user", "content": "what's the weather"}])
    assert rendered == (
        "Voice conversation so far:\n"
        "USER: what's the weather\n"
        "\n"
        "Act on the user's most recent request in the conversation above. If it asks for something, tell the user the result as soon as you have it, before going on with other work, in a message that begins with >>. If it only stops or changes work already under way, tell them nothing: the assistant already has."
    )


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
        "Act on the user's most recent request in the conversation above. If it asks for something, tell the user the result as soon as you have it, before going on with other work, in a message that begins with >>. If it only stops or changes work already under way, tell them nothing: the assistant already has."
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
            [("text", ">> It's 62 and raining in Seattle.")],
        ]
    )
    backend, requester, runner = _attached_backend(llm)
    events: list = []
    statuses: list[str] = []

    async def body():
        async with _BackendSession(requester, "backend") as session:
            statuses.append(await session.send("what's the weather in seattle"))
            events.extend(await _until_idle(session))
            assert session.capabilities == {"steering": True, "progress": True}

    await _drive(runner, requester, backend, body)

    assert statuses == ["idle"]
    # A message the model begins with the mark is for the user, and arrives
    # without it; anything else it writes is notes on the work.
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
async def test_a_message_sent_while_the_model_runs_is_taken_up_by_the_next_run():
    """No run of its own: the run that follows the current step sees the request beside the result."""
    lookup_started = asyncio.Event()
    lookup_may_finish = asyncio.Event()

    async def slow_lookup(params: FunctionCallParams):
        """Look something up, slowly."""
        lookup_started.set()
        await lookup_may_finish.wait()
        await params.result_callback({"found": True})

    # The response stays open for a while after issuing its call, so the second
    # message arrives while the model is still busy with the first.
    llm = _ScriptedLLM(
        [
            [("call", "slow_lookup", "call_1", {})],
            [("text", ">> Done with both.")],
        ],
        settle_secs=1.0,
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
    assert len(llm.contexts_seen) == 2
    users = [m["content"] for m in llm.contexts_seen[1] if m.get("role") == "user"]
    assert users == ["First", "Second"]
    assert [e.text for e in events if isinstance(e, BackendOutput)] == ["Done with both."]


@pytest.mark.asyncio
async def test_a_message_sent_while_a_synchronous_tool_is_in_flight_waits_for_its_result():
    """The context lacks the call's result until it returns, so the request rides the run it brings."""
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
            [("text", ">> The lookup is done, and on to the second thing.")],
        ]
    )
    backend, requester, runner = _attached_backend(llm, tools=[slow_lookup])
    events: list = []

    async def body():
        async with _BackendSession(requester, "backend") as session:
            await session.send("First")
            await asyncio.wait_for(lookup_started.wait(), 5)
            assert (await session.send("Second")) == "working"
            await asyncio.sleep(0.2)
            assert len(llm.contexts_seen) == 1
            lookup_may_finish.set()
            events.extend(await _until_idle(session))

    await _drive(runner, requester, backend, body)

    assert [e.text for e in events if isinstance(e, BackendOutput)] == [
        "The lookup is done, and on to the second thing.",
    ]
    # One run for both: the result and the request were in view together.
    assert len(llm.contexts_seen) == 2
    contents = [m.get("content") for m in llm.contexts_seen[1]]
    assert "Second" in contents
    assert any(isinstance(c, str) and "found" in c for c in contents)


@pytest.mark.asyncio
async def test_a_message_sent_while_only_an_asynchronous_tool_is_in_flight_runs_at_once():
    """The model is idle and the call has its placeholder, so the request does not wait."""
    lookup_started = asyncio.Event()
    lookup_may_finish = asyncio.Event()

    @tool_options(cancel_on_interruption=False)
    async def slow_lookup(params: FunctionCallParams):
        """Look something up, slowly."""
        lookup_started.set()
        await lookup_may_finish.wait()
        await params.result_callback({"found": True})

    llm = _ScriptedLLM(
        [
            [("call", "slow_lookup", "call_1", {})],
            [("text", ">> On the second thing now.")],
            [("text", ">> And the lookup is done.")],
        ]
    )
    backend, requester, runner = _attached_backend(llm, tools=[slow_lookup])
    events: list = []

    async def body():
        async with _BackendSession(requester, "backend") as session:
            await session.send("First")
            await asyncio.wait_for(lookup_started.wait(), 5)
            assert (await session.send("Second")) == "working"
            async for event in session:
                events.append(event)
                if isinstance(event, BackendOutput):
                    break
            lookup_may_finish.set()
            events.extend(await _until_idle(session))

    await _drive(runner, requester, backend, body)

    # The second request ran before the lookup returned, on a context that
    # still had the call in progress.
    assert [e.text for e in events if isinstance(e, BackendOutput)] == [
        "On the second thing now.",
        "And the lookup is done.",
    ]
    assert len(llm.contexts_seen) == 3
    assert "Second" in [m.get("content") for m in llm.contexts_seen[1]]


@pytest.mark.asyncio
async def test_closing_the_session_stops_the_backend_and_the_next_session_starts_clean():
    started = asyncio.Event()

    async def slow_lookup(params: FunctionCallParams):
        """Look something up, slowly."""
        started.set()
        await asyncio.sleep(30)
        await params.result_callback({"never": "reached"})

    llm = _ScriptedLLM(
        [[("call", "slow_lookup", "call_1", {})], [("text", ">> Second time round.")]]
    )
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
async def test_detaching_lets_go_of_a_request_held_for_a_call_in_flight():
    """The cancellations a detach brings must not run the model on the request they settle."""
    lookup_started = asyncio.Event()

    async def slow_lookup(params: FunctionCallParams):
        """Look something up, slowly."""
        lookup_started.set()
        await asyncio.sleep(30)
        await params.result_callback({"found": True})

    llm = _ScriptedLLM([[("call", "slow_lookup", "call_1", {})], [("text", ">> Never ran.")]])
    backend, requester, runner = _attached_backend(llm, tools=[slow_lookup])

    async def body():
        async with _BackendSession(requester, "backend") as session:
            await session.send("First")
            await asyncio.wait_for(lookup_started.wait(), 5)
            # Held for the synchronous call in flight.
            assert (await session.send("Second")) == "working"
        # Detached: the call is cancelled, and the held request goes with it.
        await asyncio.sleep(0.5)
        assert not backend.working
        assert len(llm.contexts_seen) == 1

    await _drive(runner, requester, backend, body)


@pytest.mark.asyncio
async def test_a_second_frontend_cannot_attach_and_a_message_needs_an_attached_frontend():
    llm = _ScriptedLLM([[("text", ">> Hello.")]])
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

    await runner.add_workers(other)
    await _drive(runner, requester, backend, body)


@pytest.mark.asyncio
async def test_a_backend_error_reaches_the_frontend_and_the_stream_goes_on():
    llm = _ScriptedLLM([[("error", "the provider is down")], [("text", ">> Back again.")]])
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
            [("text", ">> Done.")],
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


@pytest.mark.asyncio
async def test_transform_output_shapes_what_the_model_writes():
    llm = _ScriptedLLM(
        [
            [("text", ">> Checking."), ("call", "get_weather", "call_1", {"location": "Seattle"})],
            [("thought", "Keep it short."), ("text", "Internal note.")],
            [("text", "")],
        ]
    )

    async def transform_output(output: BackendOutput) -> BackendOutput | None:
        if output.is_thought:
            return None
        if output.prefers_spoken:
            return replace(output, text=output.text.upper())
        return replace(output, text="")

    backend, requester, runner = _attached_backend(llm, transform_output=transform_output)
    events: list = []

    async def body():
        async with _BackendSession(requester, "backend") as session:
            await session.send("Do it")
            events.extend(await _until_idle(session))

    await _drive(runner, requester, backend, body)

    # The transform sees each output after the mark rule (flag set, mark
    # stripped): the thought was dropped, the spoken output reshaped, and an
    # emptied note still sent.
    assert [(e.text, e.prefers_spoken) for e in events if isinstance(e, BackendOutput)] == [
        ("CHECKING.", True),
        ("", False),
    ]


@pytest.mark.asyncio
async def test_a_marked_message_beside_tool_calls_is_spoken_and_the_calls_still_run():
    llm = _ScriptedLLM(
        [
            [
                ("text", ">> Your flight is delayed, so I'm booking you a taxi."),
                ("call", "book_taxi", "call_1", {"time": "12:30"}),
            ],
            [("text", ">> Taxi booked for 12:30.")],
        ]
    )
    backend, requester, runner = _attached_backend(llm, tools=[book_taxi])
    events: list = []

    async def body():
        async with _BackendSession(requester, "backend") as session:
            await session.send("Check my flight and book a taxi")
            events.extend(await _until_idle(session))

    await _drive(runner, requester, backend, body)

    # The mark decides, not whether the turn made calls: a result can ride the
    # turn that starts the next step.
    assert [(e.text, e.prefers_spoken) for e in events if isinstance(e, BackendOutput)] == [
        ("Your flight is delayed, so I'm booking you a taxi.", True),
        ("Taxi booked for 12:30.", True),
    ]
    assert {e.function_name for e in events if isinstance(e, BackendToolCall)} == {"book_taxi"}
    assert len(llm.contexts_seen) == 2
    assert isinstance(events[-1], BackendIdle)
    # The model's own context keeps the message as written, mark and all.
    assistant = [m for m in llm.contexts_seen[1] if m.get("role") == "assistant"]
    assert any(">> Your flight is delayed" in str(m.get("content")) for m in assistant)


@pytest.mark.asyncio
async def test_a_mark_partway_through_a_message_splits_it_into_a_note_and_a_result():
    llm = _ScriptedLLM(
        [
            [
                (
                    "text",
                    "Found the issue: a fixed backoff. Fixing it next.\n\n"
                    ">> The runbook says to check the status page first.",
                ),
                ("call", "book_taxi", "call_1", {"time": "12:30"}),
            ],
            [("text", ">> Fixed, and the tests pass.")],
        ]
    )
    backend, requester, runner = _attached_backend(llm, tools=[book_taxi])
    events: list = []

    async def body():
        async with _BackendSession(requester, "backend") as session:
            await session.send("Fix the test and look up the runbook")
            events.extend(await _until_idle(session))

    await _drive(runner, requester, backend, body)

    # What precedes the marked line is a note; from the mark on is spoken.
    assert [(e.text, e.prefers_spoken) for e in events if isinstance(e, BackendOutput)] == [
        ("Found the issue: a fixed backoff. Fixing it next.", False),
        ("The runbook says to check the status page first.", True),
        ("Fixed, and the tests pass.", True),
    ]


@pytest.mark.asyncio
async def test_an_unmarked_message_is_a_note_and_a_thought_is_never_spoken():
    """A message that stops the work gets no spoken reply: the model writes notes, or nothing."""
    lookup_started = asyncio.Event()

    @tool_options(cancel_on_interruption=False, cancellable_by_llm=True)
    async def slow_lookup(params: FunctionCallParams):
        """Look something up, slowly."""
        lookup_started.set()
        await asyncio.sleep(30)
        await params.result_callback({"found": True})

    llm = _ScriptedLLM(
        [
            [("thought", ">> Look it up first."), ("call", "slow_lookup", "call_1", {})],
            [("text", "Stopping the lookup."), ("call", "cancel_slow_lookup", "call_c", {})],
            # The cancellation's result brings one more run.
            [("text", "Nothing left to do.")],
        ]
    )
    backend, requester, runner = _attached_backend(llm, tools=[slow_lookup])
    events: list = []

    async def body():
        async with _BackendSession(requester, "backend") as session:
            await session.send("Look it up")
            await asyncio.wait_for(lookup_started.wait(), 5)
            assert (await session.send("Never mind")) == "working"
            events.extend(await _until_idle(session))

    await _drive(runner, requester, backend, body)

    # A thought stays a note whatever it begins with, and keeps its text as is.
    assert [
        (e.text, e.is_thought, e.prefers_spoken) for e in events if isinstance(e, BackendOutput)
    ] == [
        (">> Look it up first.", True, False),
        ("Stopping the lookup.", False, False),
        ("Nothing left to do.", False, False),
    ]
    phases = [(e.function_name, e.phase) for e in events if isinstance(e, BackendToolCall)]
    assert ("slow_lookup", "cancelled") in phases
    assert isinstance(events[-1], BackendIdle)
    assert len(llm.contexts_seen) == 3


def test_the_model_is_told_how_what_it_writes_reaches_the_user():
    llm = _ScriptedLLM([])
    BackendLLMWorker(llm=llm)

    told = llm._settings.system_instruction or ""
    assert BACKEND_PAIRING_INSTRUCTION in told
    assert SPOKEN_MARK in BACKEND_PAIRING_INSTRUCTION


def test_an_app_can_replace_the_backends_pairing_instruction():
    llm = _ScriptedLLM([])
    BackendLLMWorker(llm=llm, pairing_instruction="Begin what the user should hear with >>.")

    told = llm._settings.system_instruction or ""
    assert "Begin what the user should hear with >>." in told
    assert BACKEND_PAIRING_INSTRUCTION not in told


@pytest.mark.asyncio
async def test_a_tool_handler_that_raises_leaves_the_backend_working():
    llm = _ScriptedLLM(
        [
            [("call", "raise_an_error", "call_1", {})],
            [("text", ">> That did not work, sorry.")],
        ]
    )
    backend, requester, runner = _attached_backend(llm, tools=[raise_an_error])
    events: list = []

    async def body():
        async with _BackendSession(requester, "backend") as session:
            await session.send("Do it")
            events.extend(await _until_idle(session))

    await _drive(runner, requester, backend, body)

    assert not any(isinstance(e, BackendError) for e in events)
    assert [e.text for e in events if isinstance(e, BackendOutput)] == ["That did not work, sorry."]


@pytest.mark.asyncio
async def test_a_chained_request_reports_at_each_step_and_idles_at_the_end():
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
            [("text", ">> Taxi booked for 12:30.")],
        ]
    )
    backend, requester, runner = _attached_backend(llm, tools=[check_flight_status, book_taxi])
    events: list = []

    async def body():
        async with _BackendSession(requester, "backend") as session:
            await session.send("Get me to the airport")
            events.extend(await _until_idle(session))

    await _drive(runner, requester, backend, body)

    assert len(llm.contexts_seen) == 3
    assert [(e.text, e.is_thought) for e in events if isinstance(e, BackendOutput)] == [
        ("Check the flight first.", True),
        ("It's delayed, so I'm booking a taxi for 12:30.", False),
        ("Taxi booked for 12:30.", False),
    ]
    assert isinstance(events[-1], BackendIdle)
    assert sum(isinstance(e, BackendIdle) for e in events) == 1
