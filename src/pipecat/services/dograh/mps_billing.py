from typing import Any, Mapping

MPS_CORRELATION_ID_METADATA_KEY = "mps_correlation_id"
MPS_BILLING_VERSION_KEY = "mps_billing_version"
MPS_BILLING_VERSION_V2 = "2"
WORKFLOW_RUN_ID_METADATA_KEY = "workflow_run_id"


def get_correlation_id(
    *,
    explicit_correlation_id: str | None,
    start_metadata: Mapping[str, Any] | None,
) -> str | None:
    if explicit_correlation_id:
        return explicit_correlation_id

    if not start_metadata:
        return None

    correlation_id = start_metadata.get(MPS_CORRELATION_ID_METADATA_KEY)
    if correlation_id is None:
        correlation_id = start_metadata.get(WORKFLOW_RUN_ID_METADATA_KEY)
    if correlation_id is None:
        return None

    return str(correlation_id)


def uses_mps_billing_v2(
    *,
    explicit_correlation_id: str | None,
    start_metadata: Mapping[str, Any] | None,
) -> bool:
    return bool(
        explicit_correlation_id
        or (start_metadata and start_metadata.get(MPS_CORRELATION_ID_METADATA_KEY))
    )
