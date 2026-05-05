#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RTVI pipeline frame definitions."""

from dataclasses import dataclass
from typing import Any

from pipecat.frames.frames import SystemFrame
from pipecat.processors.frameworks.rtvi.models import UITaskData


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
class RTVIUICommandFrame(SystemFrame):
    """A frame for sending a UI command to the client.

    Pipeline-side counterpart of the ``ui-command`` RTVI message.
    The observer wraps the ``command_name`` + ``payload`` into a
    ``UICommandMessage`` envelope before pushing it to the transport,
    so the wire shape is:
    ``{label, type: "ui-command", data: {name, payload}}``.

    Parameters:
        command_name: App-defined command name (e.g. ``"toast"``,
            ``"navigate"``, or any app-specific name). The wire
            field is ``data.name``; this avoids shadowing
            ``SystemFrame.name`` (the per-instance debug label).
        payload: App-defined payload. Pydantic command models
            (``Toast``, ``Navigate``, ``ScrollTo``, ...) should be
            converted to a plain dict via ``model_dump()`` before
            being placed here; an arbitrary dict works as well.
    """

    command_name: str = ""
    payload: Any = None

    def __str__(self):
        """String representation of the UI command frame."""
        return f"{self.name}(command: {self.command_name})"


@dataclass
class RTVIUITaskFrame(SystemFrame):
    """A frame for sending a UI task lifecycle envelope to the client.

    Pipeline-side counterpart of the ``ui-task`` RTVI message. The
    observer wraps the ``data`` into a ``UITaskMessage`` envelope
    before pushing it to the transport, so the wire shape is:
    ``{label, type: "ui-task", data: <one of the four kinds>}``.

    Parameters:
        data: One of the four task-lifecycle data models from
            ``rtvi.models`` (``UITaskGroupStartedData``,
            ``UITaskUpdateData``, ``UITaskCompletedData``, or
            ``UITaskGroupCompletedData``). The ``kind`` field on
            each discriminates which lifecycle phase this is.
    """

    data: UITaskData | None = None

    def __str__(self):
        """String representation of the UI task frame."""
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
    events instead (like the bridge in ``pipecat-ai-subagents``)
    keeps using the event handler.

    Parameters:
        msg_id: The RTVI message id, as set by the client.
        event_name: App-defined event name (the ``data.name`` field).
        payload: App-defined payload (the ``data.payload`` field).
    """

    msg_id: str = ""
    event_name: str = ""
    payload: Any = None

    def __str__(self):
        """String representation of the UI event frame."""
        return f"{self.name}(event: {self.event_name})"


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
class RTVIUICancelTaskFrame(SystemFrame):
    """An inbound user-task-group cancellation request from the client.

    Pushed downstream by ``RTVIProcessor`` whenever a
    ``ui-cancel-task`` message arrives, alongside firing
    ``on_ui_message``. The server-side framework should look up the
    matching task group and cancel it (subject to whatever
    cancellable policy the group was registered with).

    Parameters:
        msg_id: The RTVI message id, as set by the client.
        task_id: The task group id the client wants cancelled.
        reason: Optional human-readable reason.
    """

    msg_id: str = ""
    task_id: str = ""
    reason: str | None = None

    def __str__(self):
        """String representation of the UI cancel-task frame."""
        return f"{self.name}(task_id: {self.task_id})"


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
