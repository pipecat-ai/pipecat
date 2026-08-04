#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User-facing job group context (compatibility shim).

Client-visible job groups live on ``JobGroupContext`` itself: pass
``ui=UIJobGroupOptions(...)`` to ``BaseUIWorker.job(...)``,
``job_group(...)``, or ``request_job_group(...)``. This subclass keeps the
historical constructor signature for code that imported it directly.
"""

from pipecat.pipeline.job_context import JobGroupContext
from pipecat.utils.deprecation import deprecated
from pipecat.workers.base_ui_worker import UIJobGroupOptions
from pipecat.workers.base_worker import BaseWorker


@deprecated(
    "`UIJobGroupContext` is deprecated since 1.8.0 and will be removed in 2.0.0. "
    "Use `JobGroupContext` with `ui=UIJobGroupOptions(...)` instead."
)
class UIJobGroupContext(JobGroupContext):
    """Deprecated alias for a client-visible :class:`JobGroupContext`.

    .. deprecated:: 1.8.0
        Use :class:`~pipecat.pipeline.job_context.JobGroupContext` with
        ``ui=UIJobGroupOptions(...)`` instead (or the ``ui=`` parameter on
        ``BaseUIWorker.job_group``). Will be removed in 2.0.0.
    """

    def __init__(
        self,
        worker: BaseWorker,
        worker_names: tuple[str, ...],
        *,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
        label: str | None = None,
        cancellable: bool = True,
    ):
        """Initialize with the historical UIJobGroupContext signature."""
        super().__init__(
            worker,
            worker_names,
            name=name,
            payload=payload,
            timeout=timeout,
            cancel_on_error=cancel_on_error,
            ui=UIJobGroupOptions(label=label, cancellable=cancellable),
        )

    @property
    def label(self) -> str | None:
        """The group's human-readable label."""
        return self._ui.label if self._ui else None

    @property
    def cancellable(self) -> bool:
        """Whether the client may request cancellation."""
        return self._ui.cancellable if self._ui else True
