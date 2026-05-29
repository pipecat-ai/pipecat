#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User-facing job group context.

Wraps ``JobGroupContext`` so the work it dispatches is also surfaced
to the UI client through the UI Worker protocol. Apps reach this via
``UIWorker.ui_job_group(...)`` rather than constructing it directly.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pipecat.bus.ui.messages import (
    BusUIJobGroupCompletedMessage,
    BusUIJobGroupStartedMessage,
)
from pipecat.pipeline.job_context import JobGroupContext

if TYPE_CHECKING:
    from pipecat.workers.ui.ui_worker import UIWorker


class UIJobGroupContext(JobGroupContext):
    """Job group whose lifecycle is forwarded to the UI client.

    Behaves like ``JobGroupContext`` for the dispatching code, and additionally
    forwards the group's lifecycle -- start, per-worker progress, and completion
    -- to the UI client as ``ui-job-group`` envelopes, so the client can show a
    cancellable progress card. Workers need not know about the UI surface: any
    ``send_job_update`` they emit against the group's ``job_id`` is forwarded
    automatically.

    Example::

        async with self.ui_job_group(
            "researcher_a", "researcher_b",
            payload={"query": query},
            label=f"Research: {query}",
            cancellable=True,
        ) as tg:
            async for event in tg:
                ...
            results = tg.responses
    """

    def __init__(
        self,
        worker: UIWorker,
        worker_names: tuple[str, ...],
        *,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
        label: str | None = None,
        cancellable: bool = True,
    ):
        """Initialize the UIJobGroupContext.

        Args:
            worker: The parent ``UIWorker`` that owns this job group.
            worker_names: Names of the workers to send the job to.
            name: Optional job name for routing to named ``@job``
                handlers on the workers.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds covering both the
                ready-wait and job execution.
            cancel_on_error: Whether to cancel the group if a worker
                errors. Defaults to True.
            label: Optional human-readable label surfaced to the
                client (e.g. ``"Research: Radiohead"``). The client UI
                uses it to title the in-flight job-group card.
            cancellable: Whether the client may request cancellation
                of this group via the reserved ``__cancel_job_group`` event.
                Defaults to True.
        """
        super().__init__(
            worker,
            worker_names,
            name=name,
            payload=payload,
            timeout=timeout,
            cancel_on_error=cancel_on_error,
        )
        self._ui_worker = worker
        self._label = label
        self._cancellable = cancellable

    @property
    def label(self) -> str | None:
        """The group's human-readable label.

        Returns:
            The label surfaced to the client, or ``None`` if unset.
        """
        return self._label

    @property
    def cancellable(self) -> bool:
        """Whether the client may request cancellation.

        Returns:
            ``True`` if the client may cancel this group via the reserved
            ``__cancel_job_group`` event.
        """
        return self._cancellable

    async def __aenter__(self) -> UIJobGroupContext:
        await super().__aenter__()
        job_id = self.job_id
        self._ui_worker._register_ui_job_group(
            job_id=job_id,
            worker_names=list(self._worker_names),
            label=self._label,
            cancellable=self._cancellable,
        )
        await self._ui_worker.send_bus_message(
            BusUIJobGroupStartedMessage(
                source=self._ui_worker.name,
                target=None,
                job_id=job_id,
                workers=list(self._worker_names),
                label=self._label,
                cancellable=self._cancellable,
                at=int(time.time() * 1000),
            )
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        job_id = self._group.job_id if self._group else None
        try:
            return await super().__aexit__(exc_type, exc_val, exc_tb)
        finally:
            if job_id:
                self._ui_worker._unregister_ui_job_group(job_id)
                await self._ui_worker.send_bus_message(
                    BusUIJobGroupCompletedMessage(
                        source=self._ui_worker.name,
                        target=None,
                        job_id=job_id,
                        at=int(time.time() * 1000),
                    )
                )
