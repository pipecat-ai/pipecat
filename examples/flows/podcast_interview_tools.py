#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tools for the podcast interview flow defined in podcast_interview.yaml.

Each tool is a Flows direct function: its name, description, and parameters
come from the signature and docstring. None of them chooses the next node.
They return ``(result, TRANSITION_IN_YAML)`` and the flow config decides where each one
leads, including the interview node's transition back to itself.
"""

from typing import TypedDict

from pipecat.flows import TRANSITION_IN_YAML, FlowManager, TransitionInYaml


class ProceedToTopicResult(TypedDict):
    """Result type for proceed_to_topic function"""

    guest_summary: str


class StartInterviewResult(TypedDict):
    """Result type for start_interview function"""

    topic: str


async def proceed_to_topic(
    flow_manager: FlowManager, guest_summary: str
) -> tuple[ProceedToTopicResult, TransitionInYaml]:
    """Use after the guest has introduced themselves.

    Args:
        guest_summary (str): A quick summary of who the guest is (name, role, area of expertise, etc.).
    """
    return ProceedToTopicResult(guest_summary=guest_summary), TRANSITION_IN_YAML


async def start_interview(
    flow_manager: FlowManager, topic: str
) -> tuple[StartInterviewResult, TransitionInYaml]:
    """Use this when the guest has shared a clear topic they want to explore.

    Args:
        topic (str): The topic the guest wants to discuss.
    """
    return StartInterviewResult(topic=topic), TRANSITION_IN_YAML


async def next_question(flow_manager: FlowManager) -> tuple[None, TransitionInYaml]:
    """Use this after you've thoroughly explored the current aspect with multiple questions and follow-ups."""
    return None, TRANSITION_IN_YAML


async def wrap_up(flow_manager: FlowManager) -> tuple[None, TransitionInYaml]:
    """Use this when you've gathered substantial insights and are ready to wrap up."""
    return None, TRANSITION_IN_YAML


async def end_interview(flow_manager: FlowManager) -> tuple[None, TransitionInYaml]:
    """Use this after the guest has shared their final thoughts."""
    return None, TRANSITION_IN_YAML
