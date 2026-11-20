#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Telling an RTVI client where a flow is.

:func:`announce_nodes` sends the client an RTVI server message each time a
:class:`~pipecat.flows.FlowManager` enters a node, so a client can follow a
conversation through its flow.
"""

from typing import TYPE_CHECKING, Any

from pipecat.flows.exceptions import FlowError
from pipecat.flows.types import NodeEntered

if TYPE_CHECKING:
    from pipecat.flows.manager import FlowManager


def announce_nodes(flow_manager: "FlowManager") -> None:
    """Send an RTVI server message to the client each time the flow enters a node.

    Call it once, after creating the flow manager. The message is the data of
    an RTVI ``server-message``::

        {
            "type": "flows",
            "event": "node",
            "node": "find_slot",
            "previous_node": "start",
            "function": "begin",
            "metadata": {"step": 2},
            "function_metadata": {"exit": "chosen"}
        }

    ``type``, ``event`` and ``node`` are always present. The other fields are
    those of :class:`~pipecat.flows.NodeEntered`, and one with no value is left
    out. Metadata is sent as is, so it must be JSON-serializable.

    Args:
        flow_manager: The flow manager whose nodes to announce.

    Raises:
        ~pipecat.flows.FlowError: If the flow manager's worker has RTVI disabled.
    """
    try:
        rtvi = flow_manager.worker.rtvi
    except Exception:
        raise FlowError("announce_nodes needs a worker with RTVI enabled") from None

    async def on_node_entered(flow_manager: "FlowManager", entered: NodeEntered) -> None:
        message: dict[str, Any] = {"type": "flows", "event": "node", "node": entered.node}
        for field in ("previous_node", "function", "metadata", "function_metadata"):
            value = getattr(entered, field)
            if value is not None:
                message[field] = value
        await rtvi.send_server_message(message)

    flow_manager.add_event_handler("on_node_entered", on_node_entered)
