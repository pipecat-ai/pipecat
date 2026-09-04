#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tools for the restaurant reservation flow defined in restaurant_reservation.yaml.

Each tool is a Flows direct function: its name, description, and parameters
come from the signature and docstring, and the body does the work. None of
them chooses the next node. They return ``(result, None)`` and the flow config
decides where each one leads. ``check_availability`` reports a ``status`` the
config branches on.
"""

import asyncio
from typing import Literal, TypedDict

from pipecat.flows import FlowManager


class MockReservationSystem:
    """Simulates a restaurant reservation system API."""

    def __init__(self):
        # Times that are fully booked.
        self.booked_times = {"7:00 PM", "8:00 PM"}

    async def check_availability(
        self, party_size: int, requested_time: str
    ) -> tuple[bool, list[str]]:
        """Check if a table is available for the given party size and time."""
        # Simulate API call delay
        await asyncio.sleep(0.5)

        is_available = requested_time not in self.booked_times

        alternatives = []
        if not is_available:
            base_times = ["5:00 PM", "6:00 PM", "7:00 PM", "8:00 PM", "9:00 PM", "10:00 PM"]
            alternatives = [t for t in base_times if t not in self.booked_times]

        return is_available, alternatives


reservation_system = MockReservationSystem()


class PartySizeResult(TypedDict):
    size: int
    status: str


class TimeResult(TypedDict):
    status: Literal["available", "unavailable"]
    time: str
    alternative_times: list[str]


async def collect_party_size(flow_manager: FlowManager, size: int) -> tuple[PartySizeResult, None]:
    """
    Record the number of people in the party.

    Args:
        size (int): Number of people in the party. Must be between 1 and 12.
    """
    flow_manager.state["party_size"] = size
    return PartySizeResult(size=size, status="success"), None


async def check_availability(
    flow_manager: FlowManager, time: str, party_size: int
) -> tuple[TimeResult, None]:
    """
    Check availability for requested time.

    Args:
        time (str): Requested reservation time in "HH:MM AM/PM" format. Must be between 5 PM and 10 PM.
        party_size (int): Number of people in the party.
    """
    is_available, alternative_times = await reservation_system.check_availability(party_size, time)

    return TimeResult(
        status="available" if is_available else "unavailable",
        time=time,
        alternative_times=alternative_times,
    ), None


async def end_conversation(flow_manager: FlowManager) -> tuple[None, None]:
    """End the conversation."""
    return None, None
