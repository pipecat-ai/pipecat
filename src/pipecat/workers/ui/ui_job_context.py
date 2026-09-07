#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User-facing job group context (compatibility shim).

A group dispatched by a ``BaseUIWorker`` is client-visible, so a plain
``JobGroupContext`` covers what this subclass used to. It keeps the
historical constructor signature for code that imported it directly.
"""

from pipecat.pipeline.job_context import JobGroupContext, JobGroupParams
from pipecat.utils.deprecation import deprecated
from pipecat.workers.base_worker import BaseWorker


@deprecated(
    "`UIJobGroupContext` is deprecated since 1.8.0 and will be removed in 2.0.0. "
    "Use `JobGroupContext` instead."
)
class UIJobGroupContext(JobGroupContext):
    """Deprecated alias for a :class:`JobGroupContext` on a ``BaseUIWorker``.

    .. deprecated:: 1.8.0
        Use :class:`~pipecat.pipeline.job_context.JobGroupContext` instead;
        a group dispatched by a ``BaseUIWorker`` is client-visible either
        way. Will be removed in 2.0.0.
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
            params=JobGroupParams(
                name=name,
                payload=payload,
                timeout=timeout,
                cancel_on_error=cancel_on_error,
                label=label,
                cancellable=cancellable,
            ),
        )

    @property
    def label(self) -> str | None:
        """The group's human-readable label."""
        return self._params.label

    @property
    def cancellable(self) -> bool:
        """Whether the client may request cancellation."""
        return self._params.cancellable
