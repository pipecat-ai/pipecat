#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Selection of the incoming video frames an input transport passes on."""

import time

from pipecat.frames.frames import UserImageRequestFrame

# How early a frame may arrive and still count as due, at most half an interval.
# It covers jitter in the incoming stream, which is about the same at any framerate.
JITTER_TOLERANCE_SECS = 0.05


class _VideoInSampler:
    """Decides which frames of one incoming video source a transport passes on.

    A user's camera or screen share sends frames at its own rate, often 15-30 per
    second, while a bot usually wants fewer: a vision bot may want one a second,
    or only a frame when it asks for one. An input transport keeps one sampler per
    video source and calls :meth:`sample` for every frame that arrives; the
    sampler says whether to pass that frame on or drop it.

    The framerate decides which frames are passed on:

    - ``None``: every frame.
    - ``0``: only frames that answer an image request.
    - ``N > 0``: ``N`` frames per second, on the schedule described below.

    Whatever the framerate, an image request queued with :meth:`add_request` is
    answered by the next frame that arrives, which is passed on with the request.

    For ``N > 0``, frames are due on a fixed schedule, one interval (``1 / N``
    seconds) apart:

    - **On schedule.** A frame is due once its scheduled time comes, or up to a
      tolerance before it, since frames never arrive exactly on time. The next
      frame is scheduled one interval after this frame's *scheduled* time, not its
      arrival time, so the rate doesn't drift. A frame arriving late moves the
      schedule to its arrival, so a stream slower than the framerate passes every
      frame rather than falling further and further behind.
    - **Fresh start.** The first frame, a frame that answers a request, and the
      first frame after a stall (more than an interval past its scheduled time)
      start a new schedule from their arrival. After a stall this passes on one
      frame instead of a burst catching up on the missed ones.

    The tolerance is the smaller of :data:`JITTER_TOLERANCE_SECS` and half an
    interval. At low framerates the next frame after any frame is therefore close
    to a full interval later; at framerates near the stream's own rate it covers
    the stream's jitter, so every frame passes.
    """

    def __init__(self, framerate: int | None):
        """Initialize the sampler.

        Args:
            framerate: Frames per second to pass on. ``0`` passes on frames only
                to answer requests, and ``None`` passes on every frame.
        """
        self._framerate = framerate
        # When the next frame is due. Zero before the first frame, which then
        # counts as arriving after a stall and starts the schedule.
        self._next_time = 0.0
        self._requests: list[UserImageRequestFrame] = []

    @property
    def framerate(self) -> int | None:
        """The frames per second this sampler passes on."""
        return self._framerate

    @framerate.setter
    def framerate(self, framerate: int | None):
        self._framerate = framerate

    def add_request(self, request: UserImageRequestFrame):
        """Queue an image request for the next frame to answer.

        Args:
            request: The image request to answer.
        """
        self._requests.append(request)

    def sample(self) -> tuple[bool, UserImageRequestFrame | None]:
        """Decide whether to pass on the frame that just arrived.

        Returns:
            A tuple of ``(due, request)``: ``due`` is whether to pass the frame on, and
            ``request`` is the ``UserImageRequestFrame`` it answers, or ``None``.
        """
        # A waiting request is always answered by this frame.
        request = self._requests.pop(0) if self._requests else None

        if self._framerate is None:
            return True, request
        if self._framerate == 0:
            return request is not None, request

        now = time.time()
        interval = 1 / self._framerate
        tolerance = min(JITTER_TOLERANCE_SECS, interval / 2)

        if request is None and now < self._next_time - tolerance:
            # Too early: the next frame isn't due yet.
            return False, None

        if request or now - self._next_time > interval:
            # Fresh start (a request, the first frame, or a stall): the next
            # frame is due one interval from now.
            self._next_time = now + interval
        else:
            # On schedule: one interval after this frame's scheduled time, or
            # after its arrival if it came late.
            self._next_time = max(self._next_time, now - tolerance) + interval

        return True, request
