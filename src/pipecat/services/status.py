#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Service health status.

Lets application code and connection-management logic ask whether a service is
usable, and — when it isn't — whether waiting and retrying could ever change
that.
"""

from enum import Enum

from pipecat.utils.errors import ErrorCategory

__all__ = ["ServiceStatus", "status_for_category"]


class ServiceStatus(Enum):
    """The health of a service.

    Parameters:
        UNKNOWN: The service has not reported its health. Services that don't
            classify their errors stay here for their whole lifetime.
        READY: The service is connected and working.
        DEGRADED: The service hit a failure and is retrying.
        UNAVAILABLE: The service gave up retrying, but the failure is one that
            could clear on its own, such as a provider outage.
        MISCONFIGURED: The service cannot work with its current configuration.
            Reconnecting will keep failing until credentials or settings change.
    """

    UNKNOWN = "unknown"
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    MISCONFIGURED = "misconfigured"

    @property
    def is_misconfigured(self) -> bool:
        """Whether the service needs reconfiguring before it can work again.

        The one status worth giving up on. A service that is merely degraded,
        or unavailable because the provider is having trouble, may start
        working again on its own, so work keeps flowing to it.
        """
        return self is ServiceStatus.MISCONFIGURED


def status_for_category(category: ErrorCategory) -> ServiceStatus | None:
    """Map an error category to the status it implies.

    Args:
        category: The category of the error the service reported.

    Returns:
        The implied status, or None when the category says nothing about the
        service's health — a rate limit or a one-off transient failure doesn't
        mean the service is unhealthy.
    """
    if category.is_configuration_error:
        return ServiceStatus.MISCONFIGURED
    return None
