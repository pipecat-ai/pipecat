#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bus carriers for the UI Worker protocol.

These dataclasses are the on-the-bus shape that ``UIWorker`` (see
``pipecat.workers.ui``) and ``PipelineWorker`` exchange. They are NOT
the on-the-wire format the client sees; that lives in
``pipecat.processors.frameworks.rtvi.models`` (``UIEventMessage``,
``UICommandMessage``, ``UIJobGroupMessage``, ...). When RTVI is enabled,
``PipelineWorker`` translates between the two: its ``on_ui_message``
handler republishes inbound client messages onto the bus, and its
``on_bus_message`` handler turns outbound carriers into RTVI frames.

All carriers subclass ``BusUIDataMessage``, which ``PipelineWorker``
dispatches on to translate outbound ones into RTVI frames.

- ``BusUIEventMessage`` and ``BusUICommandMessage`` carry client
  events and server commands respectively.
- ``BusUIJobGroupStartedMessage``, ``BusUIJobUpdateMessage``,
  ``BusUIJobCompletedMessage``, and ``BusUIJobGroupCompletedMessage``
  carry the four phases of a user-facing job group's lifecycle (see
  ``UIWorker.ui_job_group``).

The carriers live in the ``bus`` layer (rather than alongside
``UIWorker``) because both ``PipelineWorker`` (in ``pipecat.pipeline``)
and ``UIWorker`` (in ``pipecat.workers``) reference them, and
``pipeline`` must not import from ``workers``.
"""

from dataclasses import dataclass
from typing import Any

from pipecat.bus.messages import BusDataMessage

#: Internal ``event_name`` used by ``PipelineWorker`` when republishing a
#: ``ui-snapshot`` wire message onto the bus as a
#: ``BusUIEventMessage``. ``UIWorker``'s bus dispatch matches on this
#: name to route the snapshot into ``_latest_snapshot`` storage. The
#: leading double underscore marks the name as internal so app-defined
#: ``@ui_event`` handlers can't collide with it.
_UI_SNAPSHOT_BUS_EVENT_NAME = "__ui_snapshot"

#: Internal ``event_name`` used by ``PipelineWorker`` when republishing a
#: ``ui-cancel-job-group`` wire message onto the bus as a
#: ``BusUIEventMessage``. ``UIWorker``'s bus dispatch matches on this
#: name to route to ``cancel_job_group``. Internal; not part of the
#: public wire format.
_UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME = "__cancel_job_group"


@dataclass
class BusUIDataMessage(BusDataMessage):
    """Base for all UI Worker protocol bus carriers.

    ``PipelineWorker.on_bus_message`` dispatches on this type to translate a
    worker's outbound UI carriers into RTVI frames, so every UI bus message
    below subclasses it.
    """

    pass


@dataclass
class BusUIEventMessage(BusUIDataMessage):
    """A UI event sent from the client to a server-side worker.

    Emitted by ``PipelineWorker`` when the
    client dispatches an event via
    ``PipecatClient.sendUIEvent(event, payload)``. ``UIWorker``
    subclasses dispatch these to ``@ui_event(name)`` handlers.

    Parameters:
        event_name: App-defined event name.
        payload: App-defined payload. Schemaless by design.
    """

    event_name: str = ""
    payload: Any = None


@dataclass
class BusUICommandMessage(BusUIDataMessage):
    """A UI command sent from a server-side worker to the client.

    Published by ``UIWorker.send_command(name, payload)``. ``PipelineWorker``
    (in ``on_bus_message``) translates this to an
    ``RTVIUICommandFrame(command=command_name, payload=payload)`` and
    pushes it through the pipeline.

    Parameters:
        command_name: App-defined command name.
        payload: App-defined payload (already a plain dict by the time
            it lands on the bus).
    """

    command_name: str = ""
    payload: Any = None


# ---------------------------------------------------------------------------
# UI job-group lifecycle
# ---------------------------------------------------------------------------


@dataclass
class BusUIJobGroupStartedMessage(BusUIDataMessage):
    """A user-facing job group has been dispatched.

    Published by ``UIWorker.ui_job_group(...)`` on entry. ``PipelineWorker``
    forwards it to the client as a ``ui-job-group`` envelope with
    ``kind = "group_started"``.

    Parameters:
        job_id: Shared job-group identifier for the group.
        workers: Names of the workers the work was dispatched to.
        label: Optional human-readable label for the group.
        cancellable: Whether the client may request cancellation.
        at: Epoch milliseconds when the group started.
    """

    job_id: str = ""
    workers: list[str] | None = None
    label: str | None = None
    cancellable: bool = True
    at: int = 0


@dataclass
class BusUIJobUpdateMessage(BusUIDataMessage):
    """Per-worker progress for a user-facing job group.

    Forwarded by the ``UIWorker`` whenever a worker emits a
    ``BusJobUpdateMessage`` whose ``job_id`` matches a registered user
    job group. ``PipelineWorker`` forwards to the client as a ``ui-job-group``
    envelope with ``kind = "job_update"``.

    Parameters:
        job_id: The shared job-group identifier.
        worker_name: The worker that produced the update.
        data: The worker's update payload, forwarded verbatim.
        at: Epoch milliseconds when the update was emitted on the bus.
    """

    job_id: str = ""
    worker_name: str = ""
    data: Any = None
    at: int = 0


@dataclass
class BusUIJobCompletedMessage(BusUIDataMessage):
    """A worker in a user-facing job group has completed.

    Forwarded by the ``UIWorker`` whenever a worker's
    ``BusJobResponseMessage`` arrives for a registered user job group.
    ``PipelineWorker`` forwards to the client as a ``ui-job-group`` envelope with
    ``kind = "job_completed"``.

    Parameters:
        job_id: The shared job-group identifier.
        worker_name: The worker that produced the response.
        status: Completion status as a string (``JobStatus`` value).
        response: The worker's response payload.
        at: Epoch milliseconds when the response was received.
    """

    job_id: str = ""
    worker_name: str = ""
    status: str = ""
    response: Any = None
    at: int = 0


@dataclass
class BusUIJobGroupCompletedMessage(BusUIDataMessage):
    """A user-facing job group has completed.

    Published when ``UIWorker.ui_job_group(...)`` exits, after every
    worker has responded (or the group has been cancelled). ``PipelineWorker``
    forwards to the client as a ``ui-job-group`` envelope with
    ``kind = "group_completed"``.

    Parameters:
        job_id: The shared job-group identifier.
        at: Epoch milliseconds when the group completed.
    """

    job_id: str = ""
    at: int = 0
