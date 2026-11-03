#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Request strategies: how a ``delegate`` call becomes the backend's request.

A :class:`BackendRequestStrategy` defines the ``delegate`` tool the frontend
calls and turns each call into the request the backend receives. There are
two, chosen by frontend kind when the app gives none.
:class:`TranscriptBackendRequestStrategy` hands over the conversation since
the previous delegation: the default for a text frontend, whose context is
current when the tool runs. :class:`ExplicitBackendRequestStrategy` has the
frontend model word the request itself: the default for a speech-to-speech
frontend, which responds to audio before the transcript of that audio
reaches its context, so the turns that prompted a handoff may not be in the
context yet, and a transcript taken then would miss them.
"""

from typing import Any

from pipecat.pipeline.llm_with_backend.prompts import (
    EXPLICIT_FRONTEND_INSTRUCTION,
    EXPLICIT_TOOL_DESCRIPTION,
    TRANSCRIPT_FRONTEND_INSTRUCTION,
    TRANSCRIPT_TOOL_DESCRIPTION,
    _is_backend_message,
)
from pipecat.services.llm_service import FunctionCallParams
from pipecat.workers.llm.backend_llm_worker import (
    EXPLICIT_REQUEST_INSTRUCTION,
    TRANSCRIPT_REQUEST_INSTRUCTION,
    _render_explicit_request,
    _render_transcript_request,
)


class BackendRequestStrategy:
    """Defines the ``delegate`` tool the frontend calls, and turns a call into the request the backend receives.

    The tool's parameters are this strategy's to declare, so the connector's
    handler has one signature and reads whatever was declared back from
    ``params.arguments`` in :meth:`compose_request`.
    """

    #: JSON-schema properties of the ``delegate`` tool; empty when it takes none.
    tool_parameters: dict[str, Any] = {}
    #: Which of the properties are required.
    tool_required: list[str] = []
    #: The strategy's guidance to the frontend model, if any: how the request
    #: is worded. It goes where the pairing instruction's
    #: :data:`PAIRING_INSTRUCTION_STRATEGY_PLACEHOLDER` stands.
    frontend_instruction: str | None = None
    #: How the ``delegate`` tool is described to the frontend model.
    tool_description: str = ""
    #: Whether the strategy reads the conversation from the frontend's context,
    #: so it needs that context to be current when the ``delegate`` tool runs.
    #: A speech-to-speech frontend's context lags the audio, so the connector
    #: refuses such a strategy there.
    requires_context: bool = False
    #: Whether a ``delegate`` call has to carry arguments for the request to
    #: be built. Predictive delegation makes its ``delegate`` call with none,
    #: so it works only with a strategy that sets this False.
    delegate_needs_arguments: bool = True

    async def compose_request(self, params: FunctionCallParams) -> str | None:
        """Turn a ``delegate`` call into the text put to the backend.

        Args:
            params: The call, carrying the arguments the tool declared and the
                frontend's context.

        Returns:
            The request text, or ``None`` when the call carries nothing new
            for the backend, in which case nothing is sent.
        """
        raise NotImplementedError


class TranscriptBackendRequestStrategy(BackendRequestStrategy):
    """Hands the backend the conversation and lets it work out the request.

    The frontend model calls the tool with no arguments; the request is the
    transcript of what was said since the previous delegation (the whole
    conversation the first time), rendered by
    :func:`~pipecat.workers.llm.backend_llm_worker._render_transcript_request`.
    The backend's own messages in the conversation are left out: the backend
    already has what it said. The default for a text frontend, whose context
    is current when the tool runs.

    The cursor assumes the conversation accrues. A rewritten context or a
    failed delegation's turns are not re-sent; the user's next request
    carries what matters. A call that finds no new turn since the previous
    delegation sends nothing.
    """

    requires_context = True
    delegate_needs_arguments = False

    def __init__(
        self,
        *,
        frontend_instruction: str | None = TRANSCRIPT_FRONTEND_INSTRUCTION,
        tool_description: str = TRANSCRIPT_TOOL_DESCRIPTION,
        request_instruction: str = TRANSCRIPT_REQUEST_INSTRUCTION,
    ):
        """Initialize the strategy.

        Args:
            frontend_instruction: The strategy's guidance to the frontend
                model, placed in the pairing instruction: that a delegation
                hands over the whole conversation, unworded.
            tool_description: How the ``delegate`` tool is described.
            request_instruction: What the backend should do with the
                transcript, placed after it.
        """
        self.frontend_instruction = frontend_instruction
        self.tool_description = tool_description
        self._request_instruction = request_instruction
        self._delegated_through = 0

    async def compose_request(self, params: FunctionCallParams) -> str | None:
        """Render the turns since the previous delegation as the request."""
        messages = params.context.get_messages()
        if self._delegated_through > len(messages):
            # The context holds fewer messages than the previous delegation
            # read: it was reset since, so the transcript starts from the top.
            self._delegated_through = 0
        # An LLMSpecificMessage, in a service's own format, is not a user or
        # assistant turn, so the transcript leaves it out.
        conversation = [
            m
            for m in messages[self._delegated_through :]
            if isinstance(m, dict) and not _is_backend_message(m)
        ]
        first = self._delegated_through == 0
        self._delegated_through = len(messages)
        if not any(m.get("role") in ("user", "assistant") for m in conversation):
            return None
        return _render_transcript_request(
            conversation, instruction=self._request_instruction, first=first
        )


class ExplicitBackendRequestStrategy(BackendRequestStrategy):
    """Has the frontend model word the request itself.

    The tool takes a ``request`` argument, which is sent to the backend as it
    stands. The default for a speech-to-speech frontend, which responds to
    audio before the transcript of that audio reaches its context, so the
    turns that prompted a handoff may not be recorded when the tool runs.
    """

    tool_parameters = {
        "request": {
            "type": "string",
            "description": (
                "What the user is asking for now, self-contained: their goal, the details "
                "they gave (places, dates, names) and their latest correction, or what they "
                "said about work already under way, such as to stop it or change it. Work "
                "already handed over stays with the backend; a new request joins it, so do "
                "not restate earlier requests."
            ),
        }
    }
    tool_required = ["request"]

    def __init__(
        self,
        *,
        frontend_instruction: str | None = EXPLICIT_FRONTEND_INSTRUCTION,
        tool_description: str = EXPLICIT_TOOL_DESCRIPTION,
        request_instruction: str = EXPLICIT_REQUEST_INSTRUCTION,
    ):
        """Initialize the strategy.

        Args:
            frontend_instruction: The strategy's guidance to the frontend
                model, placed in the pairing instruction: how to word the
                request.
            tool_description: How the ``delegate`` tool is described.
            request_instruction: What the backend should do with the request,
                placed after it.
        """
        self.frontend_instruction = frontend_instruction
        self.tool_description = tool_description
        self._request_instruction = request_instruction

    async def compose_request(self, params: FunctionCallParams) -> str | None:
        """Send the model's request as it stands, with the reporting expectation after it."""
        request = str(params.arguments.get("request") or "").strip()
        if not request:
            return None
        return _render_explicit_request(request, instruction=self._request_instruction)
