#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tools for the patient intake flow defined in flow.yaml.

Each tool is a Flows direct function: its name, description, and parameters
come from the signature and docstring, and the body does the work. None of
them chooses the next node. They return ``(result, TRANSITION_IN_YAML)`` and the flow config
decides where each one leads.
"""

from typing import TypedDict

from pipecat.flows import TRANSITION_IN_YAML, FlowManager


class BirthdayVerificationResult(TypedDict):
    verified: bool


class PrescriptionRecordResult(TypedDict):
    count: int


class AllergyRecordResult(TypedDict):
    count: int


class ConditionRecordResult(TypedDict):
    count: int


class VisitReasonRecordResult(TypedDict):
    count: int


async def verify_birthday(flow_manager: FlowManager, birthday: str):
    """Verify the user has provided their correct birthday. Once confirmed, the next step is to record the user's prescriptions.

    Args:
        birthday (str): The user's birthdate (convert to YYYY-MM-DD format).
    """
    # In a real app, this would verify against patient records
    is_valid = birthday == "1983-01-01"

    flow_manager.state["birthday_verified"] = is_valid
    flow_manager.state["birthday"] = birthday

    return BirthdayVerificationResult(verified=is_valid), TRANSITION_IN_YAML


async def record_prescriptions(flow_manager: FlowManager, prescriptions: list[dict]):
    """Record the user's prescriptions. Once confirmed, the next step is to collect allergy information.

    Args:
        prescriptions (list[dict]): List of prescription objects, each with "medication" (str, the medication's name) and "dosage" (str, the prescription's dosage).
    """
    flow_manager.state["prescriptions"] = prescriptions

    # In a real app, this would store in patient records
    return PrescriptionRecordResult(count=len(prescriptions)), TRANSITION_IN_YAML


async def record_allergies(flow_manager: FlowManager, allergies: list[dict]):
    """Record the user's allergies. Once confirmed, then next step is to collect medical conditions.

    Args:
        allergies (list[dict]): List of allergy objects, each with "name" (str, what the user is allergic to).
    """
    flow_manager.state["allergies"] = allergies

    # In a real app, this would store in patient records
    return AllergyRecordResult(count=len(allergies)), TRANSITION_IN_YAML


async def record_conditions(flow_manager: FlowManager, conditions: list[dict]):
    """Record the user's medical conditions. Once confirmed, the next step is to collect visit reasons.

    Args:
        conditions (list[dict]): List of condition objects, each with "name" (str, the user's medical condition).
    """
    flow_manager.state["conditions"] = conditions

    # In a real app, this would store in patient records
    return ConditionRecordResult(count=len(conditions)), TRANSITION_IN_YAML


async def record_visit_reasons(flow_manager: FlowManager, visit_reasons: list[dict]):
    """Record the reasons for their visit. Once confirmed, the next step is to verify all information.

    Args:
        visit_reasons (list[dict]): List of visit reason objects, each with "name" (str, the user's reason for visiting).
    """
    flow_manager.state["visit_reasons"] = visit_reasons

    # In a real app, this would store in patient records
    return VisitReasonRecordResult(count=len(visit_reasons)), TRANSITION_IN_YAML
