#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Vonage session configuration utilities.

This module extracts the necessary parameters to connect to a Vonage Video session.

Required environment variables:

- VONAGE_APPLICATION_ID - Vonage application ID
- VONAGE_SESSION_ID - Vonage session ID
- VONAGE_TOKEN - Vonage token

Example:
    from pipecat.runner.vonage import configure

    application_id, session_id, token = await configure()
"""

import os


async def configure() -> tuple[str, str, str]:
    """Configure Vonage application ID, session ID and token from environment.

    Returns:
        Tuple containing the server application_id, session_id and token.

    Raises:
        Exception: If required Vonage configuration is not provided.
    """
    application_id = os.getenv("VONAGE_APPLICATION_ID")
    session_id = os.getenv("VONAGE_SESSION_ID")
    token = os.getenv("VONAGE_TOKEN")

    if not application_id:
        raise Exception(
            "No Vonage application ID specified. Use set VONAGE_APPLICATION_ID in your environment."
        )

    if not session_id:
        raise Exception(
            "No Vonage Session ID specified. Use set VONAGE_SESSION_ID in your environment."
        )

    if not token:
        raise Exception("No Vonage token specified. Use set VONAGE_TOKEN in your environment.")

    return (application_id, session_id, token)
