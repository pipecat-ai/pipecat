#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Classification of the errors Soniox reports in-stream."""

from typing import Any

from pipecat.utils.errors import ErrorCategory, classify_http_status_code


def classify_error_code(error_code: Any) -> ErrorCategory | None:
    """Classify the ``error_code`` of a Soniox error message.

    Soniox's STT and TTS websockets report an error as a message carrying an
    HTTP status code: 402 when the account is out of credit, 401 for a
    rejected key, and so on. Returns None when the code is missing or not a
    number, which the error frame reports as unknown.

    Args:
        error_code: The ``error_code`` field of the message.

    Returns:
        The matching category, or None when there is nothing to classify.
    """
    if isinstance(error_code, int) and not isinstance(error_code, bool):
        return classify_http_status_code(error_code)
    return None
