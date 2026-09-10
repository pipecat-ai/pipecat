#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The one tool for flow.yaml.

A tool is a direct function: its name, description, and parameters come from
the signature and docstring. It returns ``(result, TRANSITION_IN_YAML)``; flow.yaml
decides that the conversation moves to the end node afterwards.
"""

from pipecat.flows import TRANSITION_IN_YAML, FlowManager


async def record_favorite_color(flow_manager: FlowManager, color: str):
    """Record the color the user said is their favorite.

    Here "record" means print to the console, but any logic could go here:
    write to a database, make an API call, etc.

    Args:
        color: The user's favorite color.
    """
    print(f"Your favorite color is: {color}")
    return color, TRANSITION_IN_YAML
