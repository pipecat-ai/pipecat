#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Observer reporting the failures a pipeline runs into.

Errors travel upstream from the processor that raised them, and not all of
them reach the end of that journey: a processor that answers for a failure
itself — a service switcher that fails over to its next service, say — stops
the error there. This observer reads each error where it is raised, so a
session's failure history holds the ones that were recovered from as well as
the ones that surfaced.
"""

import time
from collections.abc import Callable

from pydantic import BaseModel

from pipecat.frames.frames import ErrorFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.utils.errors import ErrorCategory


class ErrorEvent(BaseModel):
    """One failure, as the processor that raised it described it.

    Parameters:
        message: What went wrong, in the words of the processor that failed.
        category: Why it failed, drawn from :class:`ErrorCategory` and
            independent of the provider that failed: rejected credentials, an
            unreachable service, a malformed request and so on.
        exception_type: The name of the exception behind the failure, where one
            caused it. Failures group by this where a message, carrying the
            particulars of a single occurrence, is too specific to group by.
        processor: The name of the processor that raised the error.
        processor_usable: Whether that processor can still do its job. A
            processor that can't keeps failing for as long as it is given work,
            so this separates a bad minute from the end of a capability.
        timestamp: Unix timestamp of the failure.
    """

    message: str
    category: ErrorCategory
    exception_type: str | None = None
    processor: str
    processor_usable: bool
    timestamp: float


class ErrorObserver(BaseObserver):
    """Reports each error a pipeline raises, once, where it is raised.

    An error is reported at its origin rather than where it ends up, and named
    for the processor that raised it rather than the one that passed it along.

    Events:
        on_error(observer, event): Emitted for each error, as an
            :class:`ErrorEvent`.

    Example::

        observer = ErrorObserver()

        @observer.event_handler("on_error")
        async def on_error(observer, event):
            logger.info(event.model_dump_json())
    """

    def __init__(self, *, time_source: Callable[[], float] = time.time, **kwargs):
        """Initialize the error observer.

        Args:
            time_source: Reads the current time in seconds. Supplying one lets
                a test place failures without waiting.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._now = time_source
        self._reported: set[int] = set()

        self._register_event_handler("on_error")

    async def on_push_frame(self, data: FramePushed):
        """Report an error frame, the first time it is seen.

        An error is pushed again by every processor it travels through, and
        only the first of those pushes comes from the processor that failed.

        Args:
            data: Frame push event containing the frame and direction.
        """
        frame = data.frame
        if not isinstance(frame, ErrorFrame) or frame.id in self._reported:
            return

        self._reported.add(frame.id)

        # An error assembled by hand rather than reported through `push_error`
        # arrives without the processor and category that method settles, so
        # attribute it to the processor pushing it and report its cause as unknown.
        processor = frame.processor or data.source
        await self._call_event_handler(
            "on_error",
            ErrorEvent(
                message=frame.error,
                category=frame.category or ErrorCategory.UNKNOWN,
                exception_type=type(frame.exception).__name__ if frame.exception else None,
                processor=processor.name,
                processor_usable=processor.is_usable,
                timestamp=self._now(),
            ),
        )
