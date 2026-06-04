#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Opt-in tool mixin for ``UIWorker``.

Ships ``ReplyToolMixin``: a single bundled ``reply`` tool (a required spoken
answer plus the standard UI actions) covering the common app shapes, for
subclasses that don't need a custom tool schema. See the class for details.
"""

from loguru import logger

from pipecat.services.llm_service import FunctionCallParams
from pipecat.workers.llm.tool_decorator import tool


class ReplyToolMixin:
    """Expose a ``reply`` tool covering the full standard action set.

    Single bundled LLM tool with a required spoken ``answer`` plus
    optional visual and state-changing actions. One tool call per
    turn, no chaining; the required ``answer`` argument is enforced
    by the API schema so the model cannot omit the terminator.

    Compose alongside ``UIWorker``::

        class MyUIWorker(ReplyToolMixin, UIWorker):
            ...

    Covers pointing apps (``scroll_to`` + ``highlight``), reading
    apps (``scroll_to`` + ``select_text``), form apps (``fills`` +
    ``click``), and any blend (e.g. a document review with
    selection-based deixis AND voice-driven note-taking). The LLM
    uses whichever fields fit the user's request per turn; unused
    fields stay ``null`` and don't affect behavior.

    Delivers ``answer`` as verbatim TTS
    (``respond_to_job(answer, tts_speak=True)``) -- the worker speaks
    the exact phrase. Apps that want a minimal schema (only the fields
    actually used, or app-specific commands), or that want the
    requester's voice LLM to phrase the reply instead, write their own
    ``@tool reply`` on the ``UIWorker`` subclass directly. Use the
    helper methods on ``UIWorker`` plus ``send_command`` to dispatch the
    underlying UI commands.

    The host class must provide ``scroll_to``, ``highlight``,
    ``select_text``, ``click``, ``set_input_value``, and
    ``respond_to_job`` (``UIWorker`` does) and must be the target of
    ``@tool`` discovery on the LLM pipeline.
    """

    @tool
    async def reply(
        self,
        params: FunctionCallParams,
        answer: str,
        scroll_to: str | None = None,
        highlight: list[str] | None = None,
        select_text: str | None = None,
        fills: list[dict] | None = None,
        click: list[str] | None = None,
    ):
        """Reply to the user. Optionally point at content and act on inputs.

        Always called exactly once per turn. ``answer`` is required;
        the action fields are optional and may be combined.

        Visual / pointing actions (draw the user's attention):

        - ``scroll_to`` brings an element into view (single ref).
        - ``highlight`` flashes elements briefly (list of refs).
          Best for short emphasis like a button or a fact.
        - ``select_text`` puts the page's text selection on an
          element (single ref). Best for "this paragraph" / "the
          section about X" so the user sees exactly what was meant.
          Persists until the user clicks elsewhere.

        State-changing actions (modify form / app state):

        - ``fills`` writes values into inputs (list of
          ``{"ref", "value"}`` objects, multi-fill in one turn).
        - ``click`` clicks elements (list of refs in order). Use for
          checkboxes, radios, submit buttons.

        Order of dispatch within a turn: ``scroll_to``, then
        ``highlight``, then ``select_text``, then ``fills``, then
        ``click``, then speak the answer.

        Args:
            params: Framework-provided tool invocation context.
            answer: The spoken reply in plain language. One short
                sentence. No markdown, no symbols.
            scroll_to: Optional snapshot ref. Scrolls the element
                into view before speaking.
            highlight: Optional list of snapshot refs. Visually
                pulses each element.
            select_text: Optional snapshot ref. Places the page's
                text selection on that element.
            fills: Optional list of ``{"ref": "eN", "value": "..."}``
                objects. Writes each value into the input at ``ref``.
            click: Optional list of snapshot refs to click in order.
        """
        preview = (answer or "").strip()
        if len(preview) > 80:
            preview = preview[:80] + "…"
        logger.debug(
            f"{self}: reply(answer={preview!r}, scroll_to={scroll_to!r}, "
            f"highlight={highlight!r}, select_text={select_text!r}, "
            f"fills={fills!r}, click={click!r})"
        )
        # Defensive guards on the list arguments: an LLM that emits a
        # malformed entry (None, a bare string, etc.) would crash the
        # tool body before respond_to_job fires, leaving the
        # single-flight lock held until the requester's timeout cancels
        # us. Skip non-conforming entries instead.
        if scroll_to:
            await self.scroll_to(scroll_to)  # type: ignore[attr-defined]
        if highlight:
            for ref in highlight:
                if not isinstance(ref, str):
                    continue
                await self.highlight(ref)  # type: ignore[attr-defined]
        if select_text:
            await self.select_text(select_text)  # type: ignore[attr-defined]
        if fills:
            for entry in fills:
                if not isinstance(entry, dict):
                    continue
                ref = entry.get("ref")
                value = entry.get("value")
                if not isinstance(ref, str) or value is None:
                    continue
                await self.set_input_value(ref, str(value))  # type: ignore[attr-defined]
        if click:
            for ref in click:
                if not isinstance(ref, str):
                    continue
                await self.click(ref)  # type: ignore[attr-defined]
        await self.respond_to_job(answer, tts_speak=True)  # type: ignore[attr-defined]
        await params.result_callback(None)
