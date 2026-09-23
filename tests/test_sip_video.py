#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the SIP transport's video plane."""

import importlib.util
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

# The ``sip`` extra is optional; skip the whole module when baresip-python
# isn't installed, matching CI environments that don't pull it.
pytest.importorskip("baresip")

from pipecat.frames.frames import OutputImageRawFrame  # noqa: E402
from pipecat.transports.sip.connection import SIPConnection  # noqa: E402
from pipecat.transports.sip.transport import (  # noqa: E402
    SIPOutputTransport,
    SIPParams,
    SIPTransport,
    _i420_to_rgb,
    _rgb_to_i420,
)

# The RGB/I420 conversions live behind the ``sip-video`` extra; tests
# that exercise them skip when OpenCV isn't installed.
requires_cv2 = pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None, reason="requires the sip-video extra (cv2)"
)

WIDTH, HEIGHT = 64, 48


def make_connection(**kwargs):
    args = dict(user="1001", domain="example.com", password="secret")
    args.update(kwargs)
    connection = SIPConnection(**args)
    connection.connect = AsyncMock()
    connection.disconnect = AsyncMock()
    connection.request_keyframe = AsyncMock()
    return connection


def gradient_rgb() -> np.ndarray:
    rgb = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    rgb[:, :, 0] = np.linspace(0, 255, WIDTH, dtype=np.uint8)
    rgb[:, :, 1] = np.linspace(255, 0, HEIGHT, dtype=np.uint8)[:, None]
    rgb[:, :, 2] = 128
    return rgb


@requires_cv2
def test_i420_rgb_round_trip_within_chroma_tolerance():
    rgb = gradient_rgb()

    i420 = _rgb_to_i420(rgb.tobytes(), WIDTH, HEIGHT)
    assert len(i420) == WIDTH * HEIGHT * 3 // 2

    back = np.frombuffer(_i420_to_rgb(i420, WIDTH, HEIGHT), np.uint8).reshape(HEIGHT, WIDTH, 3)
    max_error = np.abs(back.astype(int) - rgb.astype(int)).max()
    assert max_error <= 8  # 4:2:0 chroma subsampling loss only


def test_video_geometry_aligned_from_params_before_connect():
    connection = make_connection()
    SIPTransport(
        connection,
        SIPParams(
            video_out_enabled=True,
            video_out_width=320,
            video_out_height=240,
            video_out_framerate=15,
        ),
    )
    assert connection._settings.video_size == (320, 240)
    assert connection._settings.video_fps == 15.0


def test_video_geometry_mismatch_after_connect_raises():
    connection = make_connection(video_size=(640, 480))
    connection._connected = True

    with pytest.raises(RuntimeError, match="video geometry is fixed"):
        SIPTransport(
            connection,
            SIPParams(video_out_enabled=True, video_out_width=320, video_out_height=240),
        )


def test_audio_only_transport_leaves_video_geometry_alone():
    connection = make_connection(video_size=(640, 480))
    SIPTransport(connection, SIPParams(audio_in_enabled=True, audio_out_enabled=True))
    assert connection._settings.video_size == (640, 480)


@requires_cv2
@pytest.mark.asyncio
async def test_write_video_frame_converts_rgb_to_i420():
    connection = make_connection()
    params = SIPParams(video_out_enabled=True, video_out_width=WIDTH, video_out_height=HEIGHT)
    transport = SIPTransport(connection, params)
    output = SIPOutputTransport(transport, connection, params)
    connection.write_video_frame = Mock(return_value=True)

    frame = OutputImageRawFrame(image=gradient_rgb().tobytes(), size=(WIDTH, HEIGHT), format="RGB")
    assert await output.write_video_frame(frame) is True

    written = connection.write_video_frame.call_args.args[0]
    assert len(written) == WIDTH * HEIGHT * 3 // 2


@pytest.mark.asyncio
async def test_write_video_frame_rejects_non_rgb():
    connection = make_connection()
    params = SIPParams(video_out_enabled=True)
    transport = SIPTransport(connection, params)
    output = SIPOutputTransport(transport, connection, params)
    connection.write_video_frame = Mock(return_value=True)

    frame = OutputImageRawFrame(image=b"\x00" * 16, size=(2, 2), format="RGBA")
    assert await output.write_video_frame(frame) is False
    connection.write_video_frame.assert_not_called()


@requires_cv2
@pytest.mark.asyncio
async def test_write_video_frame_geometry_refusal_returns_false():
    connection = make_connection()
    params = SIPParams(video_out_enabled=True, video_out_width=WIDTH, video_out_height=HEIGHT)
    transport = SIPTransport(connection, params)
    output = SIPOutputTransport(transport, connection, params)
    connection.write_video_frame = Mock(side_effect=ValueError("expected 460800 bytes"))

    frame = OutputImageRawFrame(image=gradient_rgb().tobytes(), size=(WIDTH, HEIGHT), format="RGB")
    assert await output.write_video_frame(frame) is False


@requires_cv2
@pytest.mark.asyncio
async def test_receive_video_pushes_rgb_frames():
    import asyncio

    from pipecat.transports.sip.transport import SIPInputTransport

    connection = make_connection()
    params = SIPParams(video_in_enabled=True)
    transport = SIPTransport(connection, params)
    input_transport = SIPInputTransport(transport, connection, params)
    input_transport.push_video_frame = AsyncMock()

    i420 = _rgb_to_i420(gradient_rgb().tobytes(), WIDTH, HEIGHT)
    video_frame = SimpleNamespace(data=i420, width=WIDTH, height=HEIGHT, timestamp_us=0)
    # One decoded frame, one empty poll (no video / nothing new), then
    # cancellation — which must escape the reader untouched.
    connection.read_video_frame = Mock(side_effect=[video_frame, None, asyncio.CancelledError()])

    with pytest.raises(asyncio.CancelledError):
        await input_transport._receive_video()

    pushed = input_transport.push_video_frame.await_args.args[0]
    assert pushed.size == (WIDTH, HEIGHT)
    assert pushed.format == "RGB"
    assert len(pushed.image) == WIDTH * HEIGHT * 3


@pytest.mark.asyncio
async def test_receive_video_surfaces_unexpected_errors():
    from pipecat.transports.sip.transport import SIPInputTransport

    connection = make_connection()
    params = SIPParams(video_in_enabled=True)
    transport = SIPTransport(connection, params)
    input_transport = SIPInputTransport(transport, connection, params)
    input_transport.push_error = AsyncMock()
    connection.read_video_frame = Mock(side_effect=RuntimeError("runtime is dead"))

    await input_transport._receive_video()

    input_transport.push_error.assert_awaited_once()
    assert "runtime is dead" in input_transport.push_error.await_args.args[0]


@pytest.mark.asyncio
async def test_request_keyframe_delegates():
    connection = make_connection()
    transport = SIPTransport(connection, SIPParams(video_in_enabled=True))

    await transport.request_keyframe()

    connection.request_keyframe.assert_awaited_once()
