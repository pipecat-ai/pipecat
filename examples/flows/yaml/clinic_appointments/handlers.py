#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tools for the appointment flow in flow.yaml and its subflow, verification.yaml.

Each tool is a direct function that reports its outcome and returns
``(result, TRANSITION_IN_YAML)``; the YAML decides where the conversation goes.
``verify_birth_date`` is called from the subflow, so the same function serves
both nodes that run it.
"""

from pipecat.flows import TRANSITION_IN_YAML, FlowManager

# The record a real system would look up.
PATIENT = {"name": "Alex Morgan", "birth_date": "1990-04-12", "appointment_day": "Thursday"}


async def verify_birth_date(flow_manager: FlowManager, birth_date: str):
    """Check the caller's date of birth against the record on file.

    Args:
        birth_date: The caller's date of birth, as YYYY-MM-DD.
    """
    verified = birth_date == PATIENT["birth_date"]
    if verified:
        flow_manager.state["patient_name"] = PATIENT["name"]
        flow_manager.state["appointment_day"] = PATIENT["appointment_day"]
    return {"verified": verified}, TRANSITION_IN_YAML


async def book_appointment(flow_manager: FlowManager, day: str):
    """Book an appointment for the verified caller.

    Args:
        day: The day the caller wants to come in.
    """
    flow_manager.state["appointment_day"] = day
    return {"booked": True, "day": day}, TRANSITION_IN_YAML


async def cancel_appointment(flow_manager: FlowManager):
    """Cancel the verified caller's appointment."""
    day = flow_manager.state.pop("appointment_day", None)
    return {"cancelled": True, "day": day}, TRANSITION_IN_YAML
