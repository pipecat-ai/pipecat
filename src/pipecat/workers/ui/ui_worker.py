#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""UIWorker: an LLM worker that observes and drives a client GUI over RTVI."""

import asyncio
import json
import time
from dataclasses import asdict, is_dataclass
from typing import Any, NamedTuple

from loguru import logger
from pydantic import BaseModel

from pipecat.bus.messages import (
    BusJobRequestMessage,
    BusJobResponseMessage,
    BusJobResponseUrgentMessage,
    BusJobStreamEndMessage,
    BusJobUpdateMessage,
    BusJobUpdateUrgentMessage,
    BusMessage,
    BusTTSSpeakMessage,
)
from pipecat.bus.ui.messages import (
    UI_CANCEL_JOB_GROUP_EVENT_NAME,
    UI_SNAPSHOT_EVENT_NAME,
    BusUICommandMessage,
    BusUIEventMessage,
    BusUIJobCompletedMessage,
    BusUIJobGroupCompletedMessage,
    BusUIJobGroupStartedMessage,
    BusUIJobUpdateMessage,
)
from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceQuestion,
    ClassifierError,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.classifiers.llm.classifier import LLMClassifier
from pipecat.frames.frames import LLMContextFrame, LLMMessagesAppendFrame, LLMMessagesUpdateFrame
from pipecat.pipeline.job_context import (
    JobGroup,
    JobGroupContext,
    JobGroupParams,
    JobGroupResponse,
    JobStatus,
)
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
from pipecat.utils.deprecation import deprecated
from pipecat.workers.llm.llm_context_worker import LLMContextWorker
from pipecat.workers.ui.ui_event_decorator import _collect_ui_event_handlers
from pipecat.workers.ui.ui_prompts import UI_STATE_PROMPT_GUIDE

# The most named elements put to the classifier as the options of one
# question. It matches Jev's limit on choice options, and an LLM does no
# better past that many either.
_MAX_ELEMENT_OPTIONS = 255


class _Element(NamedTuple):
    """A named element of the snapshot, as the screen questions see it."""

    ref: str
    role: str
    name: str
    state: list[str]
    value: str | None


#: What ``act`` can do to an element, each a command helper on the worker.
_ACTIONS = frozenset({"click", "scroll_to", "highlight", "select_text", "set_input_value"})


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
    - Decide small things with a classifier, not an LLM turn: whether a UI
      event deserves a comment (``should_respond``), which element on screen
      the user means (``which_element``), whether something is true of the
      screen (``check_screen``), which elements match a description
      (``select_elements``), and ``act`` on an element named in words.
      ``say`` speaks a line through the pipeline's TTS.
    - Answer a voice LLM's questions about the screen through the ``screen``
      job: find an element, check whether something is true, select the
      elements matching a description, list what is on screen, or click,
      scroll to, highlight, select or fill an element. Every answer is short
      data and never the page; :func:`~pipecat.workers.ui.ui_tools.screen_tools`
      gives the voice LLM the tool that sends it.
    - Answer as a delegate. The built-in single-flight ``respond`` job runs one
      screen-grounded LLM turn that a ``@tool`` ends by calling ``respond_to_job``
      (which decides how the answer reaches the user).
    - Surface long work. Every job group this worker dispatches is reported
      to the client as it goes: a card when the group starts, a line per
      worker's progress and completion, and the close when the group
      completes, whether normally, by cancellation or by timeout. The client
      can cancel a group dispatched as cancellable.

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
            classifier: Answers the small questions about the screen
                (``should_respond``, ``which_element``). Without one, the
                ``llm`` answers them through an
                :class:`~pipecat.classifiers.llm.classifier.LLMClassifier`,
                which costs an LLM call per question; a
                :class:`~pipecat.classifiers.jev.classifier.JevClassifier`
                answers in about a tenth of a second with a calibrated
                probability.
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
        self._classifier = classifier or LLMClassifier(llm=llm)
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
    def classifier(self) -> BaseClassifier:
        """The classifier this worker asks the small questions about the screen."""
        return self._classifier

    async def on_activated(self, args: dict | None) -> None:
        """Set the classifier up with this worker's task manager, then activate as usual.

        Args:
            args: Optional activation arguments.
        """
        await self._classifier.setup(self.task_manager)
        await super().on_activated(args)

    async def cleanup(self) -> None:
        """Clean up the classifier along with the worker."""
        await self._classifier.cleanup()
        await super().cleanup()

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

    async def should_respond(
        self,
        message: BusUIEventMessage,
        criteria: str = "the assistant should say something about what the user just did",
    ) -> bool:
        """Ask the classifier whether a UI event calls for the assistant to speak.

        Most clicks and edits need no comment, and a handler that reacts to
        events has to tell the few that do apart without an LLM turn. The
        question carries the event and the latest ``<ui_state>`` snapshot.

        Args:
            message: The UI event.
            criteria: What is being checked for, as a yes or no question.

        Returns:
            Whether the assistant should respond to the event.

        Raises:
            ClassifierError: If the classifier could not answer.
        """
        state: dict[str, Any] = {
            "event": {"name": message.event_name, "payload": message.payload},
        }
        screen = self.render_ui_state()
        if screen:
            state["screen"] = screen
        question = YesNoQuestion(
            instructions=criteria,
            yes="the event changes what the user is doing or asks for the assistant's attention",
            no="a routine click, scroll, hover or edit that needs no comment",
        )
        result = (await self._classifier.yes_no(state, {"respond": question}))["respond"]
        logger.debug(f"{self.name}: respond to '{message.event_name}'? {result.probability:.2f}")
        return result.is_yes

    async def which_element(self, description: str, threshold: float = 0.5) -> str | None:
        """Ask the classifier which element on screen the user means.

        The candidates are the snapshot's named elements, described by their
        role and name. The user's words and the screen are the state.

        Args:
            description: What the user said, such as "the blue button".
            threshold: The probability below which no element is returned.

        Returns:
            The element's snapshot ref, or ``None`` when there is no
            snapshot, no named element, or no confident answer.

        Raises:
            ClassifierError: If the classifier could not answer.
        """
        found = await self._find(description)
        if found["confidence"] < threshold:
            return None
        return found["ref"]

    async def check_screen(self, criteria: str) -> YesNoResult:
        """Ask the classifier whether something is true of the screen.

        Args:
            criteria: What is being checked for, as a yes or no question, such
                as "is anything on the list still unchecked?".

        Returns:
            How likely the answer is yes.

        Raises:
            ClassifierError: If the classifier could not answer.
        """
        question = YesNoQuestion(instructions=criteria)
        state = {"screen": self.render_ui_state()}
        result = (await self._classifier.yes_no(state, {"check": question}))["check"]
        logger.debug(f"{self.name}: '{criteria}'? {result.probability:.2f}")
        return result

    async def select_elements(self, criteria: str) -> list[dict[str, Any]]:
        """Ask the classifier which named elements on screen match a description.

        One yes or no question per element, all in one call.

        Args:
            criteria: What the elements should be, such as "dairy products".

        Returns:
            The matching elements, each as ``ref``, ``label`` and
            ``probability``, most likely first. Empty when nothing on screen
            matches or there is no snapshot.

        Raises:
            ClassifierError: If the classifier could not answer.
        """
        elements = self._named_elements()
        if not elements:
            return []
        questions = {
            e.ref: YesNoQuestion(instructions=f'{criteria}: does {e.role} "{e.name}" match?')
            for e in elements
        }
        state = {"criteria": criteria, "screen": self.render_ui_state()}
        results = await self._classifier.yes_no(state, questions)
        matches = [
            {"ref": e.ref, "label": e.name, "probability": results[e.ref].probability}
            for e in elements
            if results[e.ref].is_yes
        ]
        matches.sort(key=lambda m: m["probability"], reverse=True)
        logger.debug(f"{self.name}: '{criteria}' -> {[m['label'] for m in matches]}")
        return matches

    async def act(self, action: str, description: str, *, value: str | None = None) -> str | None:
        """Find the element the description means and act on it.

        Args:
            action: One of ``click``, ``scroll_to``, ``highlight``,
                ``select_text`` or ``set_input_value``.
            description: The element in words, such as "the checkout button".
            value: The text to write, for ``set_input_value``.

        Returns:
            The ref of the element acted on, or ``None`` when no element
            matched with enough confidence.

        Raises:
            ValueError: If ``action`` is not one of the five.
            ClassifierError: If the classifier could not answer.
        """
        if action not in _ACTIONS:
            raise ValueError(f"unknown screen action {action!r}, not one of {sorted(_ACTIONS)}")
        ref = await self.which_element(description)
        if not ref:
            return None
        if action == "set_input_value":
            await self.set_input_value(ref, value or "")
        else:
            await getattr(self, action)(ref)
        return ref

    def list_elements(self, role: str | None = None) -> list[dict[str, Any]]:
        """The named elements on screen, without their refs.

        Args:
            role: Only elements of this role, such as ``checkbox``; all when
                ``None``.

        Returns:
            One entry per element: ``role``, ``name``, its ``state`` tags and,
            for an input, its ``value``.
        """
        return [
            {"role": e.role, "name": e.name, "state": e.state, "value": e.value}
            for e in self._named_elements()
            if role is None or e.role == role
        ]

    async def say(self, text: str, *, target: str | None = None) -> None:
        """Have the pipeline say something through its TTS, with no LLM turn.

        Publishes a ``BusTTSSpeakMessage``; the pipeline worker that receives
        it queues a ``TTSSpeakFrame``, and the text goes into the conversation
        context as something the assistant said.

        Args:
            text: What to say.
            target: The pipeline worker to address. ``None``, the default,
                reaches every pipeline worker, which is the one there is in a
                single-bot app.
        """
        await self.send_bus_message(BusTTSSpeakMessage(source=self.name, target=target, text=text))

    async def on_bus_message(self, message: BusMessage) -> None:
        """Dispatch UI events alongside base lifecycle handling."""
        await super().on_bus_message(message)

        if not isinstance(message, BusUIEventMessage):
            return
        if message.target and message.target != self.name:
            return

        # Reserved snapshot event: store and return without dispatch or
        # ``<ui_event>`` injection. Apps render via ``inject_ui_state``.
        if message.event_name == UI_SNAPSHOT_EVENT_NAME:
            if isinstance(message.payload, dict):
                self._latest_snapshot = message.payload
            return

        # Reserved cancel event: a cancel request, never an app event.
        if message.event_name == UI_CANCEL_JOB_GROUP_EVENT_NAME:
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

    @job(name="screen")
    async def _screen_job(self, message: BusJobRequestMessage) -> None:
        """Answer a question about the screen, or act on it, for a voice LLM.

        The payload names the ``action``, its ``target`` and, for a fill, the
        ``value``. Every answer is short data and never the page.
        """
        payload = message.payload or {}
        action = self._text(message, "action")
        target = self._text(message, "target")
        value = payload.get("value")
        try:
            answer = await self._screen(action, target, str(value) if value else None)
        except (ClassifierError, ValueError) as e:
            logger.warning(f"{self.name}: screen {action!r} failed: {e}")
            await self.send_job_response(message.job_id, {"error": str(e)}, status=JobStatus.ERROR)
            return
        await self.send_job_response(message.job_id, answer)

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

    async def create_job_group_and_request_job(self, worker_names: list[str], **kwargs) -> JobGroup:
        """Dispatch a job group and announce it to the client.

        Args:
            worker_names: Names of the workers to send the job to.
            **kwargs: Everything
                :meth:`~pipecat.workers.base_worker.BaseWorker.create_job_group_and_request_job`
                takes, forwarded unchanged.

        Returns:
            The created ``JobGroup``.
        """
        group = await super().create_job_group_and_request_job(worker_names, **kwargs)
        await self.send_bus_message(
            BusUIJobGroupStartedMessage(
                source=self.name,
                target=None,
                job_id=group.job_id,
                workers=list(group.worker_names),
                label=group.label,
                cancellable=group.cancellable,
                at=int(time.time() * 1000),
            )
        )
        return group

    async def cancel_job_group(self, job_id: str, *, reason: str | None = None) -> None:
        """Cancel a running job group and complete its client card.

        Args:
            job_id: The job identifier to cancel.
            reason: Optional human-readable reason for cancellation.
        """
        # Capture the group before ``super()`` tears it down: the client's
        # card is completed from it below.
        group = self._job_groups.get(job_id)
        await super().cancel_job_group(job_id, reason=reason)
        if not group:
            return
        # The workers' own CANCELLED responses arrive after the group is
        # gone, so synthesize the terminal envelope for every worker the
        # cancellation actually cut short, deterministically instead of
        # racing the round trip. Workers that already finished keep the
        # status the client saw.
        for worker_name in group.worker_names:
            if worker_name in group.terminated:
                continue
            await self._send_job_completed(
                job_id=job_id,
                worker_name=worker_name,
                status=str(JobStatus.CANCELLED),
                response=None,
            )
        await self._send_group_completed(job_id)

    async def on_job_update(self, message: BusJobUpdateMessage | BusJobUpdateUrgentMessage) -> None:
        """Forward a worker's progress update to the client."""
        await super().on_job_update(message)
        # A group torn down by a cancellation still has messages in flight
        # from its workers; the client's card is already closed.
        if message.job_id not in self._job_groups:
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

    async def on_job_response(
        self, message: BusJobResponseMessage | BusJobResponseUrgentMessage
    ) -> None:
        """Forward a worker's response to the client as its terminal envelope.

        Runs before the group is torn down, so on an error status (with
        ``cancel_on_error``) the client learns which worker failed before
        the card closes.
        """
        await super().on_job_response(message)
        # A cancelled group is already gone, and every worker it cut short
        # was reported at cancellation; their own CANCELLED responses land
        # here afterwards and would double up.
        if message.job_id not in self._job_groups:
            return
        await self._send_job_completed(
            job_id=message.job_id,
            worker_name=message.source,
            status=str(message.status),
            response=message.response,
        )

    async def on_job_stream_end(self, message: BusJobStreamEndMessage) -> None:
        """Forward a worker's stream end as its terminal envelope.

        A worker may finish by ending its stream instead of responding; the
        client is told it completed, with the final stream data as the
        response payload.
        """
        await super().on_job_stream_end(message)
        if message.job_id not in self._job_groups:
            return
        await self._send_job_completed(
            job_id=message.job_id,
            worker_name=message.source,
            status=str(JobStatus.COMPLETED),
            response=message.data,
        )

    async def on_job_completed(self, result: JobGroupResponse) -> None:
        """Complete the client's card for a group whose workers all finished."""
        await super().on_job_completed(result)
        await self._send_group_completed(result.job_id)

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
            Use :meth:`~pipecat.workers.base_worker.BaseWorker.job_group`
            instead, since every group a ``UIWorker`` dispatches is
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
            Use :meth:`~pipecat.workers.base_worker.BaseWorker.request_job_group`
            instead, since every group a ``UIWorker`` dispatches is
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

    async def _find(self, description: str) -> dict[str, Any]:
        """The element the description means: ``ref``, ``label`` and ``confidence``.

        The candidates are the snapshot's named elements, described by role
        and name; ``ref`` and ``label`` are ``None`` when there are none.
        """
        elements = self._named_elements()
        if not elements:
            return {"ref": None, "label": None, "confidence": 0.0}
        options: dict[str, str | dict[str, Any] | list[Any] | None] = {
            e.ref: f'{e.role} "{e.name}"' for e in elements
        }
        state = {"utterance": description, "screen": self.render_ui_state()}
        question = ChoiceQuestion(
            instructions="the element on screen the user is referring to", options=options
        )
        result = (await self._classifier.choice(state, {"element": question}))["element"]
        logger.debug(f"{self.name}: '{description}' -> {result.choice} ({result.confidence:.2f})")
        label = next(e.name for e in elements if e.ref == result.choice)
        return {"ref": result.choice, "label": label, "confidence": result.confidence}

    def _named_elements(self) -> list[_Element]:
        """The snapshot's named elements, breadth first up to the cap."""
        elements: list[_Element] = []
        root = (self._latest_snapshot or {}).get("root")
        if not isinstance(root, dict):
            return elements
        pending = [root]
        while pending:
            if len(elements) == _MAX_ELEMENT_OPTIONS:
                logger.debug(
                    f"{self.name}: the screen has more than {_MAX_ELEMENT_OPTIONS} named "
                    "elements; deeper ones are not candidates"
                )
                break
            node = pending.pop(0)
            ref = node.get("ref")
            name = node.get("name")
            if isinstance(ref, str) and ref and isinstance(name, str) and name:
                state = [s for s in node.get("state") or [] if isinstance(s, str)]
                value = node.get("value")
                elements.append(
                    _Element(
                        ref=ref,
                        role=str(node.get("role", "element")),
                        name=name,
                        state=state,
                        value=value if isinstance(value, str) else None,
                    )
                )
            children = node.get("children") or []
            pending.extend(c for c in children if isinstance(c, dict))
        return elements

    def _text(self, message: BusJobRequestMessage, key: str) -> str:
        """The string a job's payload carries under ``key``, empty when missing."""
        value = (message.payload or {}).get(key)
        return value if isinstance(value, str) else ""

    async def _screen(self, action: str, target: str, value: str | None) -> dict[str, Any]:
        """The answer to one screen action, as the ``screen`` job returns it."""
        if action == "find":
            found = await self._find(target)
            return {"label": found["label"], "confidence": found["confidence"]}
        if action == "check":
            result = await self.check_screen(target)
            return {"yes": result.is_yes, "probability": result.probability}
        if action == "select":
            matches = await self.select_elements(target)
            return {"matches": [{k: m[k] for k in ("label", "probability")} for m in matches]}
        if action == "list":
            return {"elements": self.list_elements(target or None)}
        command = "set_input_value" if action == "fill" else action
        ref = await self.act(command, target, value=value)
        label = next((e.name for e in self._named_elements() if e.ref == ref), None)
        return {"done": ref is not None, "label": label}

    async def _send_job_completed(
        self,
        *,
        job_id: str,
        worker_name: str,
        status: str,
        response: dict | None,
    ) -> None:
        """Publish one worker's terminal envelope."""
        await self.send_bus_message(
            BusUIJobCompletedMessage(
                source=self.name,
                target=None,
                job_id=job_id,
                worker_name=worker_name,
                status=status,
                response=response,
                at=int(time.time() * 1000),
            )
        )

    async def _send_group_completed(self, job_id: str) -> None:
        """Publish the envelope that closes the client's card.

        Reached once per group: a group either completes with every worker
        having finished, or is cancelled, and the two paths are exclusive.
        """
        await self.send_bus_message(
            BusUIJobGroupCompletedMessage(
                source=self.name,
                target=None,
                job_id=job_id,
                at=int(time.time() * 1000),
            )
        )

    async def _handle_cancel_job_event(self, message: BusUIEventMessage) -> None:
        """Translate the client's cancel event into a cancel request.

        Hands the request to
        :meth:`~pipecat.workers.base_worker.BaseWorker.request_cancel_job_group`,
        which refuses it for a group that is unknown or was dispatched as
        non-cancellable.
        """
        payload = message.payload if isinstance(message.payload, dict) else {}
        job_id = payload.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            logger.warning(f"{self.name}: received a cancel event with no job_id; ignoring")
            return
        reason = payload.get("reason")
        if reason is not None and not isinstance(reason, str):
            reason = None
        cancelled = await self.request_cancel_job_group(
            job_id, reason=reason or "cancelled by user"
        )
        if not cancelled:
            logger.debug(
                f"{self.name}: cancel event for unknown or non-cancellable group {job_id}; ignoring"
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
