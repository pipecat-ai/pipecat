#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Utility functions for Google services."""

from typing import Any, Dict, Optional, Union

from pipecat import version as pipecat_version


def update_google_client_http_options(http_options: Optional[Union[Dict[str, Any], Any]]) -> Any:
    """Updates http_options with the x-goog-api-client header.

    Args:
        http_options: The existing http_options, which can be None, a dictionary,
                      or an object with a 'headers' attribute.

    Returns:
        The updated http_options.
    """
    client_header = {"x-goog-api-client": f"pipecat/{pipecat_version()}"}

    if http_options is None:
        http_options = {"headers": client_header}
    elif isinstance(http_options, dict):
        # Create a copy to avoid modifying the original dictionary if it's reused elsewhere
        http_options = http_options.copy()
        if "headers" in http_options:
            http_options["headers"].update(client_header)
        else:
            http_options["headers"] = client_header
    elif hasattr(http_options, "headers"):
        # We can't easily copy an arbitrary object, so we modify it in place.
        # This assumes the object is mutable and it's safe to do so.
        if http_options.headers is None:
            http_options.headers = client_header
        else:
            http_options.headers.update(client_header)

    return http_options
