#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""An LLM with a backend: a conversational frontend delegating the rest.

The frontend is any LLM service, text or speech-to-speech, and holds the
conversation. The backend is a :class:`~pipecat.workers.llm.backend_llm_worker.BackendLLMWorker`
running a heavier model with the tools, and does the work the frontend hands
off. :class:`LLMWithBackend` wraps the frontend so the pair drops into a
pipeline where an LLM goes, and installs the tools that join them.

The two exchange messages. The frontend's ``delegate`` tool puts a message
to the backend and returns at once; what the backend has to say comes back
as a stream of outputs, each appended to the frontend's conversation as a
message marked ``Backend:``, spoken or silent as the backend's flag says. A
second request joins the first, and a correction changes or stops it. How a
request is worded is the :class:`BackendRequestStrategy`'s business, and the
:class:`BackendConnector` owns the tools, the session with the backend, and
how each output is rendered into the conversation.

Everything public is importable from this package, and lives in one of its
modules:

- :mod:`~pipecat.pipeline.llm_with_backend.prompts`: what the frontend's
  model reads, from the pairing instruction to the marks on the backend's
  messages.
- :mod:`~pipecat.pipeline.llm_with_backend.request_strategies`: how a
  ``delegate`` call becomes the backend's request.
- :mod:`~pipecat.pipeline.llm_with_backend.connector`:
  :class:`BackendConnector`, which joins a frontend to its backend.
- :mod:`~pipecat.pipeline.llm_with_backend.predictive_delegation`:
  :class:`PredictiveDelegation`, an optional speed-up for a text frontend.
- :mod:`~pipecat.pipeline.llm_with_backend.llm_with_backend`:
  :class:`LLMWithBackend`, which puts the pair where an LLM goes.
"""

from pipecat.pipeline.llm_with_backend.connector import (
    DELEGATE_TOOL_NAME,
    BackendConnector,
    ConnectorContext,
)
from pipecat.pipeline.llm_with_backend.llm_with_backend import LLMWithBackend
from pipecat.pipeline.llm_with_backend.predictive_delegation import PredictiveDelegation
from pipecat.pipeline.llm_with_backend.prompts import (
    BACKEND_MESSAGE_PREFIX,
    BACKEND_MESSAGE_RELAY_INSTRUCTION,
    BACKEND_NOTE_PREFIX,
    BACKEND_PREFIXES,
    BACKEND_THOUGHT_PREFIX,
    BACKEND_WORKING_PREFIX,
    EXPLICIT_FRONTEND_INSTRUCTION,
    EXPLICIT_TOOL_DESCRIPTION,
    FRONTEND_PAIRING_INSTRUCTION,
    PAIRING_INSTRUCTION_STRATEGY_PLACEHOLDER,
    TRANSCRIPT_FRONTEND_INSTRUCTION,
    TRANSCRIPT_TOOL_DESCRIPTION,
)
from pipecat.pipeline.llm_with_backend.request_strategies import (
    BackendRequestStrategy,
    ExplicitBackendRequestStrategy,
    TranscriptBackendRequestStrategy,
)

__all__ = [
    "BACKEND_MESSAGE_PREFIX",
    "BACKEND_MESSAGE_RELAY_INSTRUCTION",
    "BACKEND_NOTE_PREFIX",
    "BACKEND_PREFIXES",
    "BACKEND_THOUGHT_PREFIX",
    "BACKEND_WORKING_PREFIX",
    "BackendConnector",
    "BackendRequestStrategy",
    "ConnectorContext",
    "DELEGATE_TOOL_NAME",
    "EXPLICIT_FRONTEND_INSTRUCTION",
    "EXPLICIT_TOOL_DESCRIPTION",
    "ExplicitBackendRequestStrategy",
    "FRONTEND_PAIRING_INSTRUCTION",
    "LLMWithBackend",
    "PAIRING_INSTRUCTION_STRATEGY_PLACEHOLDER",
    "PredictiveDelegation",
    "TRANSCRIPT_FRONTEND_INSTRUCTION",
    "TRANSCRIPT_TOOL_DESCRIPTION",
    "TranscriptBackendRequestStrategy",
]
