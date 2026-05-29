#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""UI worker: an LLM worker that observes and drives a client GUI over RTVI.

Composes the RTVI UI wire protocol (client events, accessibility snapshots,
server UI commands) with an opt-in ``ReplyToolMixin`` for the bundled reply
tool. ``PipelineWorker`` connects a ``UIWorker`` to the client automatically
whenever RTVI is enabled — no decorator or separate component to wire up.
"""

from pipecat.bus.ui.messages import (
    BusUICommandMessage,
    BusUIEventMessage,
    BusUIJobCompletedMessage,
    BusUIJobGroupCompletedMessage,
    BusUIJobGroupStartedMessage,
    BusUIJobUpdateMessage,
)
from pipecat.workers.ui.ui_event_decorator import ui_event
from pipecat.workers.ui.ui_prompts import UI_STATE_PROMPT_GUIDE
from pipecat.workers.ui.ui_tools import ReplyToolMixin
from pipecat.workers.ui.ui_worker import UIWorker

# Built-in UI command payload models (Toast, Navigate, ScrollTo,
# Highlight, Focus, Click, SetInputValue, SelectText) live in
# ``pipecat.processors.frameworks.rtvi.models``. Import them from there
# directly.

__all__ = [
    "BusUICommandMessage",
    "BusUIEventMessage",
    "BusUIJobCompletedMessage",
    "BusUIJobGroupCompletedMessage",
    "BusUIJobGroupStartedMessage",
    "BusUIJobUpdateMessage",
    "ReplyToolMixin",
    "UIWorker",
    "UI_STATE_PROMPT_GUIDE",
    "ui_event",
]
