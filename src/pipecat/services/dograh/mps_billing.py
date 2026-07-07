"""Helpers for attaching MPS billing metadata to Dograh service requests."""

from collections.abc import Mapping
from typing import Any

MPS_CORRELATION_ID_METADATA_KEY = "mps_correlation_id"
MPS_BILLING_VERSION_KEY = "mps_billing_version"
MPS_BILLING_VERSION_V2 = "2"


def get_correlation_id(
    *,
    explicit_correlation_id: str | None,
    start_metadata: Mapping[str, Any] | None,
) -> str | None:
    """Return the MPS-minted correlation id for this run, if any."""
    if explicit_correlation_id:
        return explicit_correlation_id

    if not start_metadata:
        return None

    correlation_id = start_metadata.get(MPS_CORRELATION_ID_METADATA_KEY)
    if correlation_id is None:
        return None

    return str(correlation_id)
