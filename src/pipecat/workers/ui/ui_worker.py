#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""UIWorker: an LLM worker that observes and drives a client GUI over RTVI."""

import asyncio
import json
from typing import Any

from pipecat.bus.messages import BusJobRequestMessage, BusTTSSpeakMessage
from pipecat.bus.ui.messages import BusUIEventMessage
from pipecat.classifiers.base_classifier import BaseClassifier
from pipecat.frames.frames import LLMContextFrame, LLMMessagesAppendFrame, LLMMessagesUpdateFrame
from pipecat.pipeline.job_context import JobGroupContext, JobGroupParams, JobStatus
from pipecat.pipeline.job_decorator import job
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
)
from pipecat.services.llm_service import LLMService
from pipecat.utils.deprecation import deprecated
from pipecat.workers.base_ui_worker import BaseUIWorker
from pipecat.workers.llm.llm_context_worker import LLMContextWorker
from pipecat.workers.ui.ui_prompts import UI_STATE_PROMPT_GUIDE


class UIWorker(BaseUIWorker, LLMContextWorker):
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
    - Decide small things without an LLM turn, through the ``classifier`` of
      :class:`~pipecat.workers.base_ui_worker.BaseUIWorker`, which also owns the
      screen state, the UI events and the commands.

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
        classifier: BaseClassifier | None = None,
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
            classifier: Answers small questions about the screen without an
                LLM turn; see :class:`~pipecat.workers.base_ui_worker.BaseUIWorker`.
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
            classifier=classifier,
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

    @deprecated(
        "`UIWorker.ui_job_group` is deprecated since 1.8.0 and will be removed in 2.0.0. "
        "Use `job_group` instead."
    )
    def ui_job_group(
        self,
        *worker_names: str,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
        label: str | None = None,
        cancellable: bool = True,
    ) -> JobGroupContext:
        """Deprecated wrapper for client-visible job groups.

        .. deprecated:: 1.8.0
            Use :meth:`~pipecat.workers.base_ui_worker.BaseUIWorker.job_group`
            instead, since every group a ``BaseUIWorker`` dispatches is
            client-visible. Will be removed in 2.0.0.
        """
        return self.job_group(
            *worker_names,
            params=JobGroupParams(
                name=name,
                payload=payload,
                timeout=timeout,
                cancel_on_error=cancel_on_error,
                label=label,
                cancellable=cancellable,
            ),
        )

    @deprecated(
        "`UIWorker.start_ui_job_group` is deprecated since 1.8.0 and will be removed in 2.0.0. "
        "Use `request_job_group` instead."
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
        """Deprecated wrapper for fire-and-forget client-visible job groups.

        .. deprecated:: 1.8.0
            Use :meth:`~pipecat.workers.base_ui_worker.BaseUIWorker.request_job_group`
            instead, since every group a ``BaseUIWorker`` dispatches is
            client-visible. Will be removed in 2.0.0.
        """
        return await self.request_job_group(
            *worker_names,
            params=JobGroupParams(
                name=name,
                payload=payload,
                timeout=timeout,
                cancel_on_error=cancel_on_error,
                label=label,
                cancellable=cancellable,
            ),
        )

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
        """Inject the event into the LLM context, then dispatch it.

        Injection runs first so the ``<ui_event>`` developer message lands in
        the context before any side effects the handler triggers.
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
        await super()._handle_ui_event(message)


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
