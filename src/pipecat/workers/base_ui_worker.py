#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""BaseUIWorker: a worker whose jobs and job groups surface on the client UI.

Every group a ``BaseUIWorker`` dispatches streams its lifecycle -- start,
per-worker progress, and completion -- to the UI client as ``ui-job-group``
envelopes, and the client's reserved ``__cancel_job_group`` event is honored
for groups dispatched as cancellable. ``JobGroupParams.label`` titles the
client's progress card. Dispatch from a plain ``BaseWorker`` instead when the
work should stay invisible.

A ``BaseUIWorker`` also holds the screen: it keeps the client's latest
accessibility snapshot, dispatches the client's UI events to ``@ui_event``
handlers, and drives the page with commands. With a classifier it decides the
small things about the screen, whether an event deserves a response and which
element the user means, without an LLM. No LLM is involved: it is instantiable
as-is (its inherited ``run()`` is a bus-only loop). ``UIWorker`` inherits it
and adds an LLM turn over the screen.
"""

import time
from dataclasses import asdict, is_dataclass
from typing import Any

from loguru import logger
from pydantic import BaseModel

from pipecat.bus.messages import (
    BusJobResponseMessage,
    BusJobResponseUrgentMessage,
    BusJobStreamEndMessage,
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
    BusUIJobGroupCompletedMessage,
    BusUIJobGroupStartedMessage,
    BusUIJobUpdateMessage,
)
from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceQuestion,
    ClassifierError,
    YesNoQuestion,
)
from pipecat.pipeline.job_context import (
    JobGroup,
    JobGroupResponse,
    JobStatus,
)
from pipecat.processors.frameworks.rtvi.models import (
    Click,
    Highlight,
    ScrollTo,
    SelectText,
    SetInputValue,
)
from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.ui_event_decorator import _collect_ui_event_handlers

# The most named elements put to the classifier as the options of one
# question. It matches Jev's limit on choice options, and an LLM does no
# better past that many either.
_MAX_ELEMENT_OPTIONS = 255


class BaseUIWorker(BaseWorker):
    """Worker that surfaces its jobs and job groups on the client UI.

    Every group this worker dispatches is registered for lifecycle
    forwarding: a ``group_started`` envelope is published at dispatch,
    worker updates and responses are forwarded as ``job_update`` /
    ``job_completed`` envelopes, ``group_completed`` is published at group
    teardown (normal completion, cancellation, or timeout), and the
    client's reserved ``__cancel_job_group`` event is translated into
    ``cancel_job_group`` for groups dispatched as cancellable.

    It also owns the screen: the latest snapshot (``render_ui_state``), the
    client's UI events (``@ui_event`` handlers), the commands that drive the
    page (``send_command`` and the ``click`` / ``scroll_to`` / ... helpers),
    and, given a ``classifier``, the small decisions about it
    (``should_respond``, ``which_element``).

    Instantiable directly (no LLM): register one on the runner as a
    dispatcher when a pipeline app wants client-visible background work::

        ui_jobs = BaseUIWorker("ui-jobs")
        job_id = await ui_jobs.request_job_group(
            "wikipedia", "news",
            params=JobGroupParams(
                payload={"query": query},
                label=f"Research: {query}",
            ),
        )
    """

    def __init__(
        self, name: str | None = None, *, classifier: BaseClassifier | None = None, **kwargs
    ):
        """Initialize the worker.

        Args:
            name: Unique name for this worker.
            classifier: Answers small questions about the screen, such as
                whether a UI event deserves a response (``should_respond``)
                or which element the user means (``which_element``). Set up
                when the worker starts and cleaned up when it stops.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        super().__init__(name, **kwargs)
        self._classifier = classifier
        self._ui_event_handlers = _collect_ui_event_handlers(self)
        # Latest accessibility snapshot received from the client. Updated
        # in ``on_bus_message`` when a ``__ui_snapshot`` event arrives.
        self._latest_snapshot: dict[str, Any] | None = None

    @property
    def classifier(self) -> BaseClassifier | None:
        """The classifier this worker asks small questions, if it has one."""
        return self._classifier

    async def on_activated(self, args: dict | None) -> None:
        """Set the classifier up with this worker's task manager, then activate as usual.

        Args:
            args: Optional activation arguments.
        """
        if self._classifier:
            await self._classifier.setup(self.task_manager)
        await super().on_activated(args)

    async def cleanup(self) -> None:
        """Clean up the classifier along with the worker."""
        if self._classifier:
            await self._classifier.cleanup()
        await super().cleanup()

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

    async def on_bus_message(self, message: BusMessage) -> None:
        """Keep the latest snapshot, dispatch UI events, honor the cancel event.

        Everything else this worker forwards to the client hangs off the
        job hooks (:meth:`on_job_update`, :meth:`on_job_response`,
        :meth:`on_job_stream_end`, :meth:`on_job_completed`), which the
        base class calls at the right point in a group's lifecycle.

        Args:
            message: The ``BusMessage`` to process.
        """
        await super().on_bus_message(message)
        if not isinstance(message, BusUIEventMessage):
            return
        if message.target and message.target != self.name:
            return
        if message.event_name == _UI_SNAPSHOT_BUS_EVENT_NAME:
            if isinstance(message.payload, dict):
                self._latest_snapshot = message.payload
            return
        if message.event_name == _UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME:
            await self._handle_cancel_job_event(message)
            return
        await self._handle_ui_event(message)

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

    async def should_respond(
        self,
        message: BusUIEventMessage,
        criteria: str = "the assistant should say something about what the user just did",
    ) -> bool:
        """Ask the classifier whether a UI event calls for the assistant to speak.

        Most clicks and edits need no comment. Asking first costs one small
        question instead of an LLM turn for every event. The question carries
        the event and the latest ``<ui_state>`` snapshot.

        Args:
            message: The UI event.
            criteria: What is being checked for, as a yes or no question.

        Returns:
            Whether the assistant should respond to the event.

        Raises:
            ClassifierError: If the worker has no classifier, or it could not
                answer.
        """
        if not self._classifier:
            raise ClassifierError(f"{self.name} has no classifier to ask")
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
            ClassifierError: If the worker has no classifier, or it could not
                answer.
        """
        if not self._classifier:
            raise ClassifierError(f"{self.name} has no classifier to ask")
        options = self._element_options()
        if not options:
            return None
        state = {"utterance": description, "screen": self.render_ui_state()}
        question = ChoiceQuestion(
            instructions="the element on screen the user is referring to", options=options
        )
        result = (await self._classifier.choice(state, {"element": question}))["element"]
        logger.debug(f"{self.name}: '{description}' -> {result.choice} ({result.confidence:.2f})")
        if result.confidence < threshold:
            return None
        return result.choice

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
        ``textContent``) when both are given. Used for deixis, pointing at
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
        """Translate a client ``__cancel_job_group`` event into a cancel request.

        Hands the request to
        :meth:`~pipecat.workers.base_worker.BaseWorker.request_cancel_job_group`,
        which refuses it for a group that is unknown or was dispatched as
        non-cancellable.
        """
        payload = message.payload if isinstance(message.payload, dict) else {}
        job_id = payload.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            logger.warning(
                f"Worker '{self.name}': received {_UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME} "
                "with no job_id; ignoring"
            )
            return
        reason = payload.get("reason")
        if reason is not None and not isinstance(reason, str):
            reason = None
        cancelled = await self.request_cancel_job_group(
            job_id, reason=reason or "cancelled by user"
        )
        if not cancelled:
            logger.debug(
                f"Worker '{self.name}': {_UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME} for "
                f"unknown or non-cancellable group {job_id}; ignoring"
            )

    async def _handle_ui_event(self, message: BusUIEventMessage) -> None:
        """Dispatch a UI event to its ``@ui_event`` handler.

        The handler runs in its own asyncio task so the bus dispatcher isn't
        held open while it awaits downstream work (job requests, network
        calls). Events with no registered handler are a no-op.
        """
        handler = self._ui_event_handlers.get(message.event_name)
        if not handler:
            return

        # Handlers run in their own asyncio task so the bus dispatcher
        # is never held open while a handler awaits downstream work
        # (job requests, network calls, etc.). Same pattern as ``@job``.
        self.create_task(
            handler(message),
            f"{self.name}::ui_event_{message.event_name}",
        )

    def _element_options(self) -> dict[str, str | dict[str, Any] | list[Any] | None]:
        """The snapshot's named elements, ref to description, breadth first up to the cap."""
        options: dict[str, str | dict[str, Any] | list[Any] | None] = {}
        root = (self._latest_snapshot or {}).get("root")
        if not isinstance(root, dict):
            return options
        pending = [root]
        while pending:
            if len(options) == _MAX_ELEMENT_OPTIONS:
                logger.debug(
                    f"{self.name}: the screen has more than {_MAX_ELEMENT_OPTIONS} named "
                    "elements; deeper ones are not candidates"
                )
                break
            node = pending.pop(0)
            ref = node.get("ref")
            name = node.get("name")
            if isinstance(ref, str) and ref and isinstance(name, str) and name:
                options[ref] = f'{node.get("role", "element")} "{name}"'
            children = node.get("children") or []
            pending.extend(c for c in children if isinstance(c, dict))
        return options


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
