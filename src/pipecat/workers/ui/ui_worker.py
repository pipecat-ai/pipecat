#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""UIWorker: an LLM worker that observes and drives a client GUI over RTVI."""

import asyncio
import json
import time
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from loguru import logger
from pydantic import BaseModel

from pipecat.bus.messages import (
    BusJobRequestMessage,
    BusJobResponseMessage,
    BusJobResponseUrgentMessage,
    BusJobUpdateMessage,
    BusJobUpdateUrgentMessage,
    BusMessage,
    BusTTSSpeakMessage,
)
from pipecat.bus.ui.messages import (
    _UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME,
    _UI_SNAPSHOT_BUS_EVENT_NAME,
    BusUICommandMessage,
    BusUIEventMessage,
    BusUIJobCompletedMessage,
    BusUIJobUpdateMessage,
)
from pipecat.frames.frames import LLMContextFrame, LLMMessagesAppendFrame, LLMMessagesUpdateFrame
from pipecat.pipeline.job_context import JobGroupError, JobStatus
from pipecat.pipeline.job_decorator import job
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
)
from pipecat.processors.frameworks.rtvi.models import (
    Click,
    Highlight,
    ScrollTo,
    SelectText,
    SetInputValue,
)
from pipecat.services.llm_service import LLMService
from pipecat.workers.llm.llm_context_worker import LLMContextWorker
from pipecat.workers.ui.ui_event_decorator import _collect_ui_event_handlers
from pipecat.workers.ui.ui_job_context import UIJobGroupContext
from pipecat.workers.ui.ui_prompts import UI_STATE_PROMPT_GUIDE


@dataclass
class _UIJobGroupRegistration:
    """Per-group metadata a UIWorker keeps for each in-flight user job group.

    Consulted by ``on_bus_message`` to decide which bus job messages to forward
    to the client and whether a ``__cancel_job_group`` event should be honored.

    Parameters:
        worker_names: Names of the workers the group was dispatched to.
        label: Optional human-readable label shown on the client job-group card.
        cancellable: Whether the client may cancel the group via ``__cancel_job_group``.
    """

    worker_names: list[str]
    label: str | None
    cancellable: bool


class UIWorker(LLMContextWorker):
    """LLM worker that reads and drives a client GUI over the RTVI UI channel.

    A ``UIWorker`` connects an LLM to whatever the user is looking at: it sees
    the screen as accessibility snapshots, reacts to the user's UI events, and
    acts on the page by sending commands to the client. It is the delegate side
    of a voice/UI split -- a voice layer (the main pipeline's LLM, or a separate
    ``LLMWorker``) handles speech and hands screen-relevant work to this worker.

    Capabilities:

    - See the screen. The latest accessibility snapshot is rendered as
      ``<ui_state>`` and auto-injected into the LLM context before each inference.
    - React to UI events, dispatched to ``@ui_event(name)`` handlers.
    - Drive the UI with ``send_command`` and the ``scroll_to`` / ``highlight`` /
      ``select_text`` / ``click`` / ``set_input_value`` helpers.
    - Answer as a delegate. The built-in single-flight ``respond`` job runs one
      screen-grounded LLM turn that a ``@tool`` ends by calling ``respond_to_job``
      (which decides how the answer reaches the user).
    - Surface long work. ``ui_job_group`` / ``start_ui_job_group`` fan work out to
      peer workers as cancellable job-group cards on the client.

    ``PipelineWorker`` connects a UIWorker to the client automatically when RTVI
    is enabled -- no extra wiring. A working subclass needs only an LLM and a
    ``@tool`` that calls ``respond_to_job``; override ``render_query`` to read a
    non-default job payload.

    Example::

        class MyUIWorker(UIWorker):
            @ui_event("nav_click")
            async def on_nav(self, message):
                view = message.payload.get("view")
                ...

            @tool
            async def answer(self, params, text: str):
                await self.respond_to_job(text)
                await params.result_callback(None)

        worker = MyUIWorker("ui", llm=OpenAILLMService(api_key="..."))

    Note:
        With client ``trackViewport`` on (the default), off-screen nodes carry
        ``[offscreen]`` in ``<ui_state>``; ``scroll_to`` before acting on them.
    """

    def __init__(
        self,
        name: str,
        *,
        llm: LLMService[Any],
        context: LLMContext | None = None,
        assistant_params: LLMAssistantAggregatorParams | None = None,
        inject_events: bool = True,
        auto_inject_ui_state: bool = True,
        keep_history: bool = False,
        prompt_guide: str | None = UI_STATE_PROMPT_GUIDE,
    ):
        """Initialize the UIWorker.

        Args:
            name: Unique name for this worker.
            llm: The LLM service.
            context: Optional pre-built ``LLMContext``. Seeded messages are part
                of the mutable history and are cleared on each
                ``keep_history=False`` reset; put durable instructions in the
                LLM's ``system_instruction`` instead.
            assistant_params: Optional assistant-aggregator parameters, e.g. to
                enable context summarization for ``keep_history=True`` workers.
            inject_events: When True (the default), append each UI event to the
                context as a ``<ui_event>`` developer message. Override
                ``render_ui_event`` to change the content, or set False to
                disable.
            auto_inject_ui_state: When True (the default), append the latest
                ``<ui_state>`` snapshot to the context before every inference
                (via the LLM's ``on_before_process_frame`` hook). Set False to
                inject manually with ``inject_ui_state()``.
            keep_history: When False (the default), the context is cleared at the
                start of every job, so each turn sees only the current
                ``<ui_state>`` and query -- best for the stateless-delegate role.
                When True, history accumulates across jobs so the LLM can resolve
                multi-turn references ("the next one", "the Pro version"), at the
                cost of more tokens and possible confusion from stale
                ``<ui_state>`` blocks. Use context summarization to prune the
                history when it gets too large.
            prompt_guide: Wire-format guide appended to the LLM's
                ``system_instruction`` so it can parse the ``<ui_state>`` /
                ``<ui_event>`` messages. Defaults to ``UI_STATE_PROMPT_GUIDE``;
                pass a string to override or ``None`` to disable. Living in
                ``system_instruction``, it survives context resets.
        """
        super().__init__(
            name,
            llm=llm,
            active=True,
            defer_tool_frames=True,
            context=context,
            assistant_params=assistant_params,
        )
        # Auto-append the UI wire-format guide to the LLM's system
        # instruction so the author doesn't have to concatenate it manually
        # and it survives the per-job context reset. Pass a string to
        # ``prompt_guide`` to override the text, or ``None`` to disable.
        if prompt_guide:
            self.llm.append_system_instruction(prompt_guide)
        self._inject_events = inject_events
        self._auto_inject_ui_state = auto_inject_ui_state
        self._keep_history = keep_history
        self._ui_event_handlers = _collect_ui_event_handlers(self)
        # Latest accessibility snapshot received from the client. Updated
        # in ``on_bus_message`` when a ``__ui_snapshot`` event arrives.
        # Rendered into LLM context via ``inject_ui_state``.
        self._latest_snapshot: dict[str, Any] | None = None
        # Job currently being processed by this worker. Set in
        # ``_run_llm_turn``, cleared when the job completes. Lets
        # ``@tool`` methods (and the mixin tools) close out the job
        # without having to thread the job id through every call.
        self._current_job: BusJobRequestMessage | None = None
        # Resolved by ``respond_to_job`` to hand the result back to the
        # in-flight ``_run_llm_turn`` handler, which then sends the
        # job response. See the "Single-flight job semantics" section
        # in the class docstring.
        self._pending: asyncio.Future | None = None
        # Registry of in-flight user job groups dispatched by this
        # worker (see ``ui_job_group``). Keyed by ``job_id``.
        # ``on_bus_message`` consults this to decide which job
        # update / response messages should be forwarded to the
        # client as ``ui-job-group`` envelopes.
        self._ui_job_groups: dict[str, _UIJobGroupRegistration] = {}

        # Auto-inject the current ``<ui_state>`` snapshot into the context just
        # before each inference. Driven by the LLM's ``on_before_process_frame``
        # so it fires whenever the worker runs its LLM (e.g. a ``respond`` job),
        # appending the snapshot to the same context the request is built from.
        # The snapshot is a normal, persistent developer message; growth is
        # managed by ``keep_history`` + context summarization.
        @self.llm.event_handler("on_before_process_frame")
        async def _inject_ui_state(_llm, frame):
            if not (self._auto_inject_ui_state and isinstance(frame, LLMContextFrame)):
                return
            # Only inject on a user-turn-initiating inference, not the follow-up
            # inference the LLM runs after a tool result (which would stack a
            # duplicate ``<ui_state>`` within the same turn).
            if not _is_user_turn(frame.context):
                return
            content = self.render_ui_state()
            if content:
                frame.context.add_message({"role": "developer", "content": content})

    async def send_command(self, name: str, payload: Any = None) -> None:
        """Send a named UI command to the client.

        Publishes a ``BusUICommandMessage``; when RTVI is enabled,
        ``PipelineWorker`` translates it into an ``RTVIUICommandFrame`` on the
        pipeline. Client-side handlers subscribed to ``RTVIEvent.UICommand``
        (or React's ``useUICommandHandler``) dispatch on the command name.

        Args:
            name: App-defined command name (e.g. ``"toast"``,
                ``"navigate"``, or any app-specific name).
            payload: One of:

                - A pydantic ``BaseModel`` instance (including the
                  built-in command models in
                  ``pipecat.processors.frameworks.rtvi.models``).
                  Converted to a plain dict with ``model_dump()``.
                - A dataclass instance. Converted to a plain dict with
                  ``dataclasses.asdict``.
                - A ``dict`` forwarded as-is.
                - ``None``, forwarded as an empty dict.
        """
        if payload is None:
            serialized: Any = {}
        elif isinstance(payload, BaseModel):
            serialized = payload.model_dump()
        elif is_dataclass(payload) and not isinstance(payload, type):
            serialized = asdict(payload)
        else:
            serialized = payload

        await self.send_bus_message(
            BusUICommandMessage(
                source=self.name,
                target=None,
                command_name=name,
                payload=serialized,
            )
        )

    async def scroll_to(self, ref: str) -> None:
        """Send a ``scroll_to`` UI command to bring an element into view.

        Convenience wrapper around ``send_command("scroll_to", ScrollTo(ref=ref))``.
        These ``scroll_to`` / ``highlight`` / ``select_text`` / ``click`` /
        ``set_input_value`` helpers are plain methods, not LLM tools: compose
        them inside a custom ``@tool`` body, or use ``ReplyToolMixin`` for the
        standard shape.

        Args:
            ref: Snapshot ref (e.g. ``"e42"``) from the latest ``<ui_state>``.
        """
        await self.send_command("scroll_to", ScrollTo(ref=ref))

    async def highlight(self, ref: str) -> None:
        """Send a ``highlight`` UI command to briefly flash an element.

        Args:
            ref: Snapshot ref (e.g. ``"e42"``) from the latest ``<ui_state>``.
        """
        await self.send_command("highlight", Highlight(ref=ref))

    async def select_text(
        self,
        ref: str,
        *,
        start_offset: int | None = None,
        end_offset: int | None = None,
    ) -> None:
        """Send a ``select_text`` UI command to select an element's text.

        Selects the whole element by default, or the ``start_offset``..
        ``end_offset`` character sub-range (over the element's concatenated
        ``textContent``) when both are given. Used for deixis -- pointing at
        content via the page's text selection.

        Args:
            ref: Snapshot ref (e.g. ``"e42"``) from the latest ``<ui_state>``.
            start_offset: Optional start character offset of the selection.
            end_offset: Optional end character offset (exclusive).
        """
        await self.send_command(
            "select_text",
            SelectText(ref=ref, start_offset=start_offset, end_offset=end_offset),
        )

    async def click(self, ref: str) -> None:
        """Send a ``click`` UI command (checkboxes, radios, submit buttons).

        The standard client handler no-ops on ``disabled`` targets, so the
        worker can't bypass affordances meant to be user-controlled.

        Args:
            ref: Snapshot ref (e.g. ``"e42"``) from the latest ``<ui_state>``.
        """
        await self.send_command("click", Click(ref=ref))

    async def set_input_value(
        self,
        ref: str,
        value: str,
        *,
        replace: bool = True,
    ) -> None:
        """Send a ``set_input_value`` UI command to fill a text input/textarea.

        Args:
            ref: Snapshot ref (e.g. ``"e42"``) of the input or textarea.
            value: Text to write into the field.
            replace: When True (the default), overwrite the field; when False,
                append (e.g. to continue a long answer in a textarea).
        """
        await self.send_command(
            "set_input_value",
            SetInputValue(ref=ref, value=value, replace=replace),
        )

    async def on_bus_message(self, message: BusMessage) -> None:
        """Dispatch UI events alongside base lifecycle handling."""
        await super().on_bus_message(message)

        # Forward job lifecycle for user-facing job groups before
        # touching anything else. This is independent of UI event
        # handling and may fire on messages targeted at this worker
        # (the requester) for groups it dispatched.
        if isinstance(message, (BusJobUpdateMessage, BusJobUpdateUrgentMessage)):
            await self._maybe_forward_job_update(message)
            return
        if isinstance(message, (BusJobResponseMessage, BusJobResponseUrgentMessage)):
            await self._maybe_forward_job_completed(message)
            return

        if not isinstance(message, BusUIEventMessage):
            return
        if message.target and message.target != self.name:
            return

        # Reserved snapshot event: store and return without dispatch or
        # ``<ui_event>`` injection. Apps render via ``inject_ui_state``.
        if message.event_name == _UI_SNAPSHOT_BUS_EVENT_NAME:
            if isinstance(message.payload, dict):
                self._latest_snapshot = message.payload
            return

        # Reserved cancel event: route to ``cancel_job_group`` for the
        # registered user job group. Honored only when the group was
        # registered with ``cancellable=True``.
        if message.event_name == _UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME:
            await self._handle_cancel_job_event(message)
            return

        await self._handle_ui_event(message)

    @property
    def current_job(self) -> BusJobRequestMessage | None:
        """The job this worker is currently processing, or ``None`` when idle.

        Set when a respond turn starts and cleared when the job
        completes. Lets ``@tool`` methods inspect the in-flight job
        without threading the message through every call.

        Returns:
            The in-flight ``BusJobRequestMessage``, or ``None`` when idle.
        """
        return self._current_job

    @job(name="respond", sequential=True)
    async def _respond_job(self, message: BusJobRequestMessage) -> None:
        await self._run_llm_turn(message)

    def render_query(self, message: BusJobRequestMessage) -> str:
        """Extract the user's query text from a job request.

        Override to read a different payload shape. The returned string
        is appended to the LLM context as a user message before the LLM
        runs. The default reads ``payload["query"]``.

        Args:
            message: The inbound job request.

        Returns:
            The query text to feed into the LLM.
        """
        return (message.payload or {}).get("query", "")

    async def _run_llm_turn(self, message: BusJobRequestMessage) -> None:
        """Run one LLM turn for a job and respond when a ``@tool`` completes it.

        Body of the built-in ``respond`` job. Records the in-flight job, clears
        the context when ``keep_history=False``, appends the rendered query, and
        runs the LLM (the current ``<ui_state>`` is injected by the
        ``on_before_process_frame`` hook). Then blocks until a ``@tool`` calls
        ``respond_to_job``, which chooses how the answer is delivered, and sends
        the job response.

        Spanning the full round-trip is what makes the job single-flight
        (``@job(..., sequential=True)``; see the class docstring).

        Args:
            message: The inbound job request.
        """
        self._current_job = message
        self._pending = asyncio.get_running_loop().create_future()
        try:
            if not self._keep_history:
                await self._reset_context()
            # The query goes in as a "user" message, not "developer": it's the
            # request to act on, whereas the SDK-injected <ui_state> / <ui_event>
            # messages are the "developer" (programmatic context) content. The
            # "user" role also marks the turn boundary that gates <ui_state>
            # injection -- see _is_user_turn() and the on_before_process_frame
            # hook in __init__ (a non-user tail would skip the snapshot).
            await self.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": self.render_query(message)}],
                    run_llm=True,
                )
            )
            result = await self._pending
            await self.send_job_response(
                message.job_id, response=result["response"], status=result["status"]
            )
        finally:
            self._current_job = None
            self._pending = None

    async def _reset_context(self) -> None:
        """Clear the LLM conversation history.

        Replaces all messages in the running context with an empty list via
        ``LLMMessagesUpdateFrame``. The system prompt (``system_instruction``)
        is unaffected, but messages seeded via ``context=`` live in the same
        mutable list and ARE cleared. ``keep_history=False`` workers reset
        automatically per job; ``keep_history=True`` workers call this to
        deliberately start over.
        """
        await self.queue_frame(LLMMessagesUpdateFrame(messages=[], run_llm=False))

    async def respond_to_job(
        self,
        answer: str | None = None,
        *,
        tts_speak: bool = False,
        status: JobStatus = JobStatus.COMPLETED,
    ) -> None:
        """Complete the in-flight job with the worker's answer.

        Called from a ``@tool`` once the worker has decided how to answer.
        ``tts_speak`` picks the delivery; the two modes are mutually exclusive
        (one voice per turn):

        - default: the job responds with ``{"answer": answer}`` for the
          requester's voice LLM to phrase.
        - ``tts_speak=True``: ``answer`` is spoken verbatim by the requester's
          TTS (via ``BusTTSSpeakMessage``, and added to its context) while the
          job responds ``None`` so the voice LLM doesn't also speak.

        A falsy ``answer`` completes the turn silently. No-op when no job is in
        flight or it was already answered.

        Args:
            answer: The worker's answer -- spoken verbatim (``tts_speak=True``)
                or handed to the requester's voice LLM to phrase (default).
            tts_speak: Speak ``answer`` verbatim via the requester's TTS instead
                of returning it for the requester's voice LLM to phrase.
            status: Completion status. Defaults to ``JobStatus.COMPLETED``.
        """
        pending = self._pending
        if pending is None or pending.done() or self._current_job is None:
            return
        if tts_speak:
            if answer:
                await self.send_bus_message(
                    BusTTSSpeakMessage(
                        source=self.name,
                        target=self._current_job.source,
                        text=answer,
                        append_to_context=True,
                    )
                )
            response: dict | None = None
        else:
            response = {"answer": answer} if answer else None
        pending.set_result({"response": response, "status": status})

    def ui_job_group(
        self,
        *worker_names: str,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
        label: str | None = None,
        cancellable: bool = True,
    ) -> UIJobGroupContext:
        """Dispatch a job group whose lifecycle is forwarded to the client.

        Like ``job_group(...)``, but also forwards the group's lifecycle to the
        client as ``ui-job-group`` envelopes so the user can watch (and optionally
        cancel) the work. See ``UIJobGroupContext`` for the forwarding details.

        Args:
            *worker_names: Names of the workers to send the job to.
            name: Optional job name for routing to named ``@job``
                handlers on the workers.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds covering both the
                ready-wait and job execution.
            cancel_on_error: Whether to cancel the group if a worker
                errors. Defaults to True.
            label: Optional human-readable label surfaced to the
                client. The client UI uses it to title the in-flight
                job-group card.
            cancellable: Whether the client may request cancellation
                of this group via the reserved ``__cancel_job_group``
                event. Defaults to True.

        Returns:
            A ``UIJobGroupContext`` to use with ``async with``.

        Example::

            async with self.ui_job_group(
                "researcher_a", "researcher_b",
                payload={"query": query},
                label=f"Research: {query}",
            ) as tg:
                async for event in tg:
                    ...
        """
        for worker_name in worker_names:
            if not isinstance(worker_name, str):
                raise TypeError(
                    f"{self} Expected worker name as str, got {type(worker_name).__name__}"
                )
        return UIJobGroupContext(
            self,
            worker_names,
            name=name,
            payload=payload,
            timeout=timeout,
            cancel_on_error=cancel_on_error,
            label=label,
            cancellable=cancellable,
        )

    async def start_ui_job_group(
        self,
        *worker_names: str,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
        label: str | None = None,
        cancellable: bool = True,
    ) -> str:
        """Fire-and-forget version of ``ui_job_group``.

        Dispatches the group in the background and returns immediately (the
        lifecycle still forwards to the client). Use it when a ``@tool`` wants to
        kick off work and unblock the voice worker; use ``ui_job_group`` to
        consume worker events inline. Worker exceptions are logged, not propagated.

        Args:
            *worker_names: Names of the workers to send the job to.
            name: Optional job name for routing to named ``@job``
                handlers on the workers.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds covering both the
                ready-wait and job execution.
            cancel_on_error: Whether to cancel the group if a worker
                errors. Defaults to True.
            label: Optional human-readable label surfaced to the
                client. The client UI uses it to title the in-flight
                job-group card.
            cancellable: Whether the client may request cancellation
                of this group via the reserved ``__cancel_job_group``
                event. Defaults to True.

        Returns:
            The ``job_id`` of the dispatched group. Useful if the
            caller wants to track it (e.g. to cancel programmatically
            via ``cancel_job_group(job_id)``).

        Example::

            @tool
            async def reply(self, params, answer, research_query=None):
                if research_query:
                    await self.start_ui_job_group(
                        "wikipedia", "news", "scholar",
                        payload={"query": research_query},
                        label=f"Research: {research_query}",
                    )
                await self.respond_to_job(answer)
                await params.result_callback(None)
        """
        ctx = self.ui_job_group(
            *worker_names,
            name=name,
            payload=payload,
            timeout=timeout,
            cancel_on_error=cancel_on_error,
            label=label,
            cancellable=cancellable,
        )
        # Enter the context now so the caller has a valid job_id and
        # the ``group_started`` envelope has fired before we return.
        # The body and exit run in a background asyncio task so the
        # caller doesn't await worker completion.
        await ctx.__aenter__()
        job_id = ctx.job_id

        async def _run_to_completion() -> None:
            # Drain the event stream so __aexit__ sees a fully
            # consumed group, matching what ``async with ... : pass``
            # does. Cancellation and worker errors are expected exits
            # for fire-and-forget groups: the client already learned
            # via the group_completed envelope, so we log at debug.
            iteration_exc: BaseException | None = None
            try:
                async for _ in ctx:
                    pass
            except Exception as e:
                iteration_exc = e

            try:
                if iteration_exc is None:
                    await ctx.__aexit__(None, None, None)
                else:
                    await ctx.__aexit__(
                        type(iteration_exc),
                        iteration_exc,
                        iteration_exc.__traceback__,
                    )
            except JobGroupError as e:
                logger.debug(
                    f"UIWorker '{self.name}': background user job group {job_id} ended: {e}"
                )
            except Exception as e:
                logger.warning(
                    f"UIWorker '{self.name}': background user job group {job_id} failed: {e}"
                )

        self.create_task(
            _run_to_completion(),
            f"{self.name}::ui_job_group::{job_id}",
        )
        return job_id

    def _register_ui_job_group(
        self,
        *,
        job_id: str,
        worker_names: list[str],
        label: str | None,
        cancellable: bool,
    ) -> None:
        """Register an in-flight user job group for lifecycle forwarding.

        Called from ``UIJobGroupContext.__aenter__``. Subsequent
        ``BusJobUpdateMessage`` / ``BusJobResponseMessage`` whose
        ``job_id`` matches this entry will be forwarded to the client.
        """
        if job_id in self._ui_job_groups:
            logger.warning(
                f"UIWorker '{self.name}': user job group {job_id} already registered; overwriting"
            )
        self._ui_job_groups[job_id] = _UIJobGroupRegistration(
            worker_names=list(worker_names),
            label=label,
            cancellable=cancellable,
        )

    def _unregister_ui_job_group(self, job_id: str) -> None:
        """Remove a user job group from the forwarding registry.

        Called from ``UIJobGroupContext.__aexit__``. After this,
        late-arriving updates or responses for the group are not
        forwarded.
        """
        self._ui_job_groups.pop(job_id, None)

    async def _maybe_forward_job_update(
        self, message: BusJobUpdateMessage | BusJobUpdateUrgentMessage
    ) -> None:
        """Forward a worker update for a registered user job group.

        No-op if the message's ``job_id`` is not registered.
        """
        if message.job_id not in self._ui_job_groups:
            return
        await self.send_bus_message(
            BusUIJobUpdateMessage(
                source=self.name,
                target=None,
                job_id=message.job_id,
                worker_name=message.source,
                data=message.update,
                at=int(time.time() * 1000),
            )
        )

    async def _maybe_forward_job_completed(
        self, message: BusJobResponseMessage | BusJobResponseUrgentMessage
    ) -> None:
        """Forward a worker response for a registered user job group.

        No-op if the message's ``job_id`` is not registered.
        """
        if message.job_id not in self._ui_job_groups:
            return
        await self.send_bus_message(
            BusUIJobCompletedMessage(
                source=self.name,
                target=None,
                job_id=message.job_id,
                worker_name=message.source,
                status=str(message.status),
                response=message.response,
                at=int(time.time() * 1000),
            )
        )

    async def _handle_cancel_job_event(self, message: BusUIEventMessage) -> None:
        """Translate a client ``__cancel_job_group`` event into ``cancel_job_group``.

        Looks up the registered group and calls
        ``cancel_job_group(job_id, reason)``. Ignores the request
        silently if the group is unknown or was registered with
        ``cancellable=False``.
        """
        payload = message.payload if isinstance(message.payload, dict) else {}
        job_id = payload.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            logger.warning(
                f"UIWorker '{self.name}': received {_UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME} "
                "with no job_id; ignoring"
            )
            return
        registration = self._ui_job_groups.get(job_id)
        if registration is None:
            logger.debug(
                f"UIWorker '{self.name}': {_UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME} for "
                f"unknown job_id {job_id}; ignoring"
            )
            return
        if not registration.cancellable:
            logger.debug(
                f"UIWorker '{self.name}': {_UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME} for "
                f"non-cancellable group {job_id}; ignoring"
            )
            return
        reason = payload.get("reason")
        if reason is not None and not isinstance(reason, str):
            reason = None
        await self.cancel_job_group(job_id, reason=reason or "cancelled by user")

    def render_ui_state(self) -> str:
        """Render the latest accessibility snapshot as a ``<ui_state>`` block.

        Produces Playwright-MCP-style indented text with stable element
        refs. Apps inject the output via ``inject_ui_state()`` when they
        want the LLM to see what's on screen.

        When the snapshot carries a current text selection, a nested
        ``<selection ref="...">...</selection>`` block is appended
        inside ``<ui_state>`` so the LLM can resolve deictic references
        ("this paragraph", "what I selected") against on-page content.

        Override to customize the rendered form.

        Returns:
            The ``<ui_state>`` block, or an empty string if no snapshot
            has been received yet.
        """
        if not self._latest_snapshot:
            return ""
        root = self._latest_snapshot.get("root")
        if not isinstance(root, dict):
            return ""
        lines = ["<ui_state>"]
        _render_node(root, depth=0, lines=lines)
        selection = self._latest_snapshot.get("selection")
        if isinstance(selection, dict):
            _render_selection(selection, lines)
        lines.append("</ui_state>")
        return "\n".join(lines)

    async def inject_ui_state(self) -> None:
        """Append the latest ``<ui_state>`` block to the LLM context.

        No-op when no snapshot has been received. Frame has
        ``run_llm=False`` — the snapshot is context, not a user turn.
        """
        content = self.render_ui_state()
        if not content:
            return
        await self.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "developer", "content": content}],
                run_llm=False,
            )
        )

    def render_ui_event(self, message: BusUIEventMessage) -> str:
        """Render a UI event as a string for LLM context injection.

        Override to customize the injected content. The default wraps
        the event in a single ``<ui_event>`` XML tag with a ``name``
        attribute and a JSON-encoded payload as inner text.

        Args:
            message: The UI event to render.

        Returns:
            A string to append to the LLM context as a developer message.
        """
        payload_repr = json.dumps(message.payload, default=str)
        return f'<ui_event name="{message.event_name}">{payload_repr}</ui_event>'

    async def _handle_ui_event(self, message: BusUIEventMessage) -> None:
        """Inject the event into LLM context, then dispatch to the handler.

        Injection runs synchronously first so the ``<ui_event>``
        developer message lands in the context before any side effects
        the handler triggers. The matching ``@ui_event`` handler
        then runs in its own asyncio task so the bus dispatcher isn't
        held open while the handler awaits downstream work (job
        requests, network calls). Events with no registered handler
        are a no-op after injection.
        """
        if self._inject_events:
            content = self.render_ui_event(message)
            if content:
                await self.queue_frame(
                    LLMMessagesAppendFrame(
                        messages=[{"role": "developer", "content": content}],
                        run_llm=False,
                    )
                )

        handler = self._ui_event_handlers.get(message.event_name)
        if handler is None:
            return

        # Handlers run in their own asyncio task so the bus dispatcher
        # is never held open while a handler awaits downstream work
        # (job requests, network calls, etc.). Same pattern as ``@job``.
        self.create_task(
            handler(message),
            f"{self.name}::ui_event_{message.event_name}",
        )


def _is_user_turn(context: LLMContext) -> bool:
    """Whether the context's last message is the user's turn.

    Distinguishes a fresh user-turn inference (tail is the user message) from
    the follow-up inference the LLM runs after a tool result (tail is the tool
    result / assistant output), so the ``<ui_state>`` snapshot is injected once
    per turn rather than again on each tool round.
    """
    messages = context.messages
    if not messages:
        return False
    last = messages[-1]
    return isinstance(last, dict) and last.get("role") == "user"


def _render_node(node: dict[str, Any], *, depth: int, lines: list[str]) -> None:
    """Render one A11yNode dict as Playwright-MCP-style indented text.

    Format per node::

        - role "name" [level=N] [cols=N] [rows=N] [state1] [state2] [ref=eN]:

    Trailing ``:`` when the node has children. ``name``, ``level``,
    grid dims, and state tags are emitted only when present on the
    node.
    """
    role = node.get("role", "generic")
    name = node.get("name")
    value = node.get("value")
    state = node.get("state") or []
    level = node.get("level")
    colcount = node.get("colcount")
    rowcount = node.get("rowcount")
    ref = node.get("ref", "")
    children = node.get("children") or []

    parts: list[str] = [f"- {role}"]
    if isinstance(name, str) and name:
        parts.append(f'"{name}"')
    if isinstance(value, str) and value:
        parts.append(f'= "{value}"')
    if isinstance(level, int):
        parts.append(f"[level={level}]")
    if isinstance(colcount, int):
        parts.append(f"[cols={colcount}]")
    if isinstance(rowcount, int):
        parts.append(f"[rows={rowcount}]")
    if isinstance(state, list):
        for s in state:
            if isinstance(s, str) and s:
                parts.append(f"[{s}]")
    if isinstance(ref, str) and ref:
        parts.append(f"[ref={ref}]")

    indent = "  " * depth
    line = indent + " ".join(parts)
    if children:
        line += ":"
    lines.append(line)

    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                _render_node(child, depth=depth + 1, lines=lines)


def _render_selection(selection: dict[str, Any], lines: list[str]) -> None:
    """Render an ``A11ySelection`` dict as a ``<selection>`` block.

    Emitted at the root of ``<ui_state>`` (no leading indent) so the
    LLM can spot it without parsing the tree::

        <selection ref="e42">
        the actual selected text
        </selection>

    No-op when the selection lacks a ``ref`` or ``text``.
    """
    ref = selection.get("ref")
    text = selection.get("text")
    if not isinstance(ref, str) or not ref:
        return
    if not isinstance(text, str) or not text:
        return
    lines.append(f'<selection ref="{ref}">')
    lines.append(text)
    lines.append("</selection>")
