#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RTVI pipeline frame definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pipecat.frames.frames import SystemFrame
from pipecat.processors.frameworks.rtvi.models import UIJobGroupData

if TYPE_CHECKING:
    # Imported for typing only: observer.py imports this module, so a runtime
    # import back would be circular. With ``from __future__ import annotations``
    # the annotation below is just a string and the field's value is a plain dict.
    from pipecat.processors.frameworks.rtvi.observer import RTVIFunctionCallReportLevel


@dataclass
class RTVIClientMessageFrame(SystemFrame):
    """A frame for sending messages from the client to the RTVI server.

    This frame is meant for custom messaging from the client to the server
    and expects a server-response message.
    """

    msg_id: str
    type: str
    data: Any | None = None


@dataclass
class RTVIServerMessageFrame(SystemFrame):
    """A frame for sending server messages to the client.

    Parameters:
        data: The message data to send to the client.
    """

    data: Any

    def __str__(self):
        """String representation of the RTVI server message frame."""
        return f"{self.name}(data: {self.data})"


@dataclass
class RTVIServerResponseFrame(SystemFrame):
    """A frame for responding to a client RTVI message.

    This frame should be sent in response to an RTVIClientMessageFrame
    and include the original RTVIClientMessageFrame to ensure the response
    is properly attributed to the original request. To respond with an error,
    set the `error` field to a string describing the error. This will result
    in the client receiving an `error-response` message instead of a
    `server-response` message.
    """

    client_msg: RTVIClientMessageFrame
    data: Any | None = None
    error: str | None = None


@dataclass
class RTVIConfigureObserverFrame(SystemFrame):
    """Dynamically reconfigure a running :class:`RTVIObserver`.

    Lets a trusted, server-side source adjust what the observer exposes at
    runtime without baking the setting into the agent (where it would apply to
    every client). Only the fields that are set are applied; ``None`` fields
    leave the current observer configuration unchanged.

    The eval harness pushes this (via the eval-only
    :class:`~pipecat.evals.serializer.RTVIEvalSerializer`) to raise the
    function-call report level for the calls a scenario asserts on, so
    production agents can keep the secure default.

    Parameters:
        function_call_report_level: Per-function report-level map to apply to the
            observer (e.g. ``{"*": RTVIFunctionCallReportLevel.FULL}``), or
            ``None`` to leave it unchanged.
        vad_user_speaking_enabled: Whether the observer should emit raw VAD user
            started/stopped speaking messages, or ``None`` to leave it unchanged.
    """

    function_call_report_level: dict[str, RTVIFunctionCallReportLevel] | None = None
    vad_user_speaking_enabled: bool | None = None

    def __str__(self):
        """String representation of the observer-config frame."""
        return (
            f"{self.name}(function_call_report_level: {self.function_call_report_level}, "
            f"vad_user_speaking_enabled: {self.vad_user_speaking_enabled})"
        )


@dataclass
class RTVIUICommandFrame(SystemFrame):
    """A frame for sending a UI command to the client.

    Pipeline-side counterpart of the ``ui-command`` RTVI message.
    The observer wraps the ``command`` + ``payload`` into a
    ``UICommandMessage`` envelope before pushing it to the transport,
    so the wire shape is:
    ``{label, type: "ui-command", data: {command, payload}}``.

    Parameters:
        command: App-defined command (e.g. ``"toast"``,
            ``"navigate"``, or any app-specific command).
        payload: App-defined payload. Pydantic command models
            (``Toast``, ``Navigate``, ``ScrollTo``, ...) should be
            converted to a plain dict via ``model_dump()`` before
            being placed here; an arbitrary dict works as well.
    """

    command: str = ""
    payload: Any = None

    def __str__(self):
        """String representation of the UI command frame."""
        return f"{self.name}(command: {self.command})"


@dataclass
class RTVIUIJobGroupFrame(SystemFrame):
    """A frame for sending a UI job-group lifecycle envelope to the client.

    Pipeline-side counterpart of the ``ui-job-group`` RTVI message. The
    observer wraps the ``data`` into a ``UIJobGroupMessage`` envelope
    before pushing it to the transport, so the wire shape is:
    ``{label, type: "ui-job-group", data: <one of the four kinds>}``.

    Parameters:
        data: One of the four job-group lifecycle data models from
            ``rtvi.models`` (``UIJobGroupStartedData``,
            ``UIJobUpdateData``, ``UIJobCompletedData``, or
            ``UIJobGroupCompletedData``). The ``kind`` field on
            each discriminates which lifecycle phase this is.
    """

    data: UIJobGroupData | None = None

    def __str__(self):
        """String representation of the UI job-group frame."""
        kind = getattr(self.data, "kind", "?")
        return f"{self.name}(kind: {kind})"


@dataclass
class RTVIUIEventFrame(SystemFrame):
    """An inbound UI event from the client.

    Pushed downstream by ``RTVIProcessor`` whenever a ``ui-event``
    message arrives from the client, alongside firing the
    ``on_ui_message`` event handler. Mirrors the
    frame-and-event pattern used by ``client-message``: pipeline
    observers and processors that want to react to UI events at the
    pipeline level can match on this frame; code that subscribes to
    events instead (like ``UIWorker``) keeps using the event handler.

    Parameters:
        msg_id: The RTVI message id, as set by the client.
        event: App-defined event (the ``data.event`` field).
        payload: App-defined payload (the ``data.payload`` field).
    """

    msg_id: str = ""
    event: str = ""
    payload: Any = None

    def __str__(self):
        """String representation of the UI event frame."""
        return f"{self.name}(event: {self.event})"


@dataclass
class RTVIUISnapshotFrame(SystemFrame):
    """An inbound accessibility-snapshot from the client.

    Pushed downstream by ``RTVIProcessor`` whenever a ``ui-snapshot``
    message arrives, alongside firing ``on_ui_message``. Carries
    the serialized accessibility tree the client took of its DOM.

    Parameters:
        msg_id: The RTVI message id, as set by the client.
        tree: The serialized accessibility tree.
    """

    msg_id: str = ""
    tree: Any = None

    def __str__(self):
        """String representation of the UI snapshot frame."""
        return f"{self.name}"


@dataclass
class RTVIUICancelJobGroupFrame(SystemFrame):
    """An inbound user-job-group cancellation request from the client.

    Pushed downstream by ``RTVIProcessor`` whenever a
    ``ui-cancel-job-group`` message arrives, alongside firing
    ``on_ui_message``. The server-side framework should look up the
    matching job group and cancel it (subject to whatever
    cancellable policy the group was registered with).

    Parameters:
        msg_id: The RTVI message id, as set by the client.
        job_id: The job group id the client wants cancelled.
        reason: Optional human-readable reason.
    """

    msg_id: str = ""
    job_id: str = ""
    reason: str | None = None

    def __str__(self):
        """String representation of the UI cancel-job-group frame."""
        return f"{self.name}(job_id: {self.job_id})"
