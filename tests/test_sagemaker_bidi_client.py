#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SageMaker BiDi session start failures and how services report them."""

import asyncio

import pytest

pytest.importorskip("aws_sdk_sagemaker_runtime_http2")

from aws_sdk_sagemaker_runtime_http2.models import (  # noqa: E402
    InputValidationError,
    ModelError,
    ServiceUnavailableError,
)
from smithy_core.exceptions import CallError  # noqa: E402

import pipecat.services.aws.sagemaker.bidi_client as bidi_client  # noqa: E402
from pipecat.frames.frames import InputAudioRawFrame, TTSSpeakFrame  # noqa: E402
from pipecat.services.aws.sagemaker.bidi_client import (  # noqa: E402
    SageMakerBidiClient,
    SageMakerBidiSessionError,
    classify_sagemaker_bidi_error,
)
from pipecat.services.deepgram.flux.sagemaker.stt import (  # noqa: E402
    DeepgramFluxSageMakerSTTService,
)
from pipecat.services.deepgram.flux.sagemaker.tts import (  # noqa: E402
    DeepgramFluxSageMakerTTSService,
)
from pipecat.services.deepgram.sagemaker.stt import DeepgramSageMakerSTTService  # noqa: E402
from pipecat.services.deepgram.sagemaker.tts import DeepgramSageMakerTTSService  # noqa: E402
from pipecat.services.nvidia.sagemaker.stt import NvidiaSageMakerSTTService  # noqa: E402
from pipecat.services.nvidia.sagemaker.tts import NvidiaSageMakerTTSService  # noqa: E402
from pipecat.tests.utils import SleepFrame, run_test  # noqa: E402
from pipecat.utils.errors import ErrorCategory  # noqa: E402

OPERATION = "com.amazonaws.sagemakerruntimehttp2#InvokeEndpointWithBidirectionalStream"


def unmodeled_error(status: int, error_id: str) -> CallError:
    """An error shaped like the one the SDK raises for an unmodeled AWS error."""
    return CallError(
        message=(
            f"Unknown error for operation {OPERATION} - status: {status} - id: "
            f"com.amazonaws.sagemakerruntimehttp2#{error_id}"
        ),
        fault="client" if status < 500 else "server",
    )


def throttled() -> CallError:
    # SageMaker reports a throttled bidirectional stream with a 400.
    return unmodeled_error(400, "ThrottlingException")


class _InputStream:
    async def send(self, _event):
        pass

    async def close(self):
        pass


class _OutputStream:
    async def receive(self):
        await asyncio.sleep(3600)


class _Stream:
    input_stream = _InputStream()

    async def await_output(self):
        return (None, _OutputStream())


class FakeSDKClient:
    """Stands in for the SDK client, failing each session start with the next queued error."""

    errors: list[Exception] = []
    fail_forever: Exception | None = None
    attempts = 0

    def __init__(self, config=None):
        pass

    async def invoke_endpoint_with_bidirectional_stream(self, _input):
        FakeSDKClient.attempts += 1
        if FakeSDKClient.errors:
            raise FakeSDKClient.errors.pop(0)
        if FakeSDKClient.fail_forever is not None:
            raise FakeSDKClient.fail_forever
        return _Stream()


@pytest.fixture
def fake_sdk(monkeypatch):
    FakeSDKClient.errors = []
    FakeSDKClient.fail_forever = None
    FakeSDKClient.attempts = 0
    monkeypatch.setattr(bidi_client, "SageMakerRuntimeHTTP2Client", FakeSDKClient)
    monkeypatch.setattr(bidi_client, "_CONNECT_BACKOFF_MULTIPLIER", 0)
    return FakeSDKClient


@pytest.mark.parametrize(
    "exception,category",
    [
        (throttled(), ErrorCategory.RATE_LIMIT),
        (unmodeled_error(429, "TooManyRequests"), ErrorCategory.RATE_LIMIT),
        (unmodeled_error(503, "SomethingUnmodeled"), ErrorCategory.SERVER),
        (ServiceUnavailableError(message="unavailable"), ErrorCategory.SERVER),
        # Shapes AWS returned for a missing endpoint and for rejected credentials.
        (unmodeled_error(400, "ValidationError"), ErrorCategory.INVALID_REQUEST),
        (unmodeled_error(403, "InvalidSignatureException"), ErrorCategory.CONNECTIVITY),
        (unmodeled_error(403, "UnrecognizedClientException"), ErrorCategory.CONNECTIVITY),
        (InputValidationError(message="bad input"), ErrorCategory.INVALID_REQUEST),
        # The model container refused the stream (at capacity, or bad settings).
        (
            ModelError(
                message='Received server error (424) from primary with message "Failed to '
                'establish WebSocket connection".',
                original_status_code=424,
                error_code="INTERNAL_FAILURE_FROM_MODEL",
            ),
            ErrorCategory.SERVER,
        ),
        (ModelError(message="rejected", original_status_code=400), ErrorCategory.INVALID_REQUEST),
        (ModelError(message="overloaded", original_status_code=503), ErrorCategory.SERVER),
        (SageMakerBidiSessionError("x", error_id="ThrottlingException"), ErrorCategory.RATE_LIMIT),
        (RuntimeError("something else"), None),
    ],
)
def test_classify_sagemaker_bidi_error(exception, category):
    assert classify_sagemaker_bidi_error(exception) == category


def make_client(**kwargs) -> SageMakerBidiClient:
    return SageMakerBidiClient(
        endpoint_name="endpoint", region="us-east-2", model_invocation_path="v1/listen", **kwargs
    )


@pytest.mark.asyncio
async def test_start_session_retries_throttling(fake_sdk):
    fake_sdk.errors = [throttled(), throttled()]
    client = make_client()

    await client.start_session()

    assert fake_sdk.attempts == 3
    assert client.is_active


@pytest.mark.asyncio
async def test_start_session_gives_up_after_max_attempts(fake_sdk):
    fake_sdk.fail_forever = throttled()
    client = make_client(max_connect_attempts=3)

    with pytest.raises(SageMakerBidiSessionError) as raised:
        await client.start_session()

    assert fake_sdk.attempts == 3
    assert raised.value.error_id == "ThrottlingException"
    assert raised.value.http_status == 400
    assert raised.value.attempts == 3
    assert isinstance(raised.value.__cause__, CallError)
    assert not client.is_active


@pytest.mark.asyncio
async def test_start_session_does_not_retry_rejected_credentials(fake_sdk):
    fake_sdk.fail_forever = unmodeled_error(403, "InvalidSignatureException")
    client = make_client()

    with pytest.raises(SageMakerBidiSessionError) as raised:
        await client.start_session()

    assert fake_sdk.attempts == 1
    assert raised.value.http_status == 403


@pytest.mark.asyncio
async def test_start_session_does_not_retry_invalid_request(fake_sdk):
    fake_sdk.fail_forever = InputValidationError(message="bad input")
    client = make_client()

    with pytest.raises(SageMakerBidiSessionError) as raised:
        await client.start_session()

    assert fake_sdk.attempts == 1
    assert raised.value.error_id == "InputValidationError"


async def run_until_connect_fails(service, frames):
    categories = []

    @service.event_handler("on_error")
    async def on_error(_service, error):
        categories.append(error.category)

    await run_test(
        service,
        frames_to_send=[SleepFrame(sleep=0.2), *frames, SleepFrame(sleep=0.2)],
        expected_down_frames=None,
        expected_up_frames=None,
    )
    return categories


@pytest.mark.asyncio
async def test_stt_unusable_when_session_never_starts(fake_sdk):
    fake_sdk.fail_forever = throttled()
    stt = DeepgramSageMakerSTTService(endpoint_name="endpoint", region="us-east-2")
    audio = InputAudioRawFrame(audio=b"\x00" * 3200, sample_rate=16000, num_channels=1)

    categories = await run_until_connect_fails(stt, [audio])

    assert fake_sdk.attempts == 4
    assert categories == [ErrorCategory.RATE_LIMIT]
    assert not stt.is_usable


@pytest.mark.asyncio
async def test_stt_usable_after_throttled_attempt_recovers(fake_sdk):
    fake_sdk.errors = [throttled()]
    stt = DeepgramSageMakerSTTService(endpoint_name="endpoint", region="us-east-2")

    categories = await run_until_connect_fails(stt, [])

    assert fake_sdk.attempts == 2
    assert categories == []
    assert stt.is_usable


@pytest.mark.asyncio
async def test_flux_stt_unusable_when_session_never_starts(fake_sdk):
    fake_sdk.fail_forever = throttled()
    stt = DeepgramFluxSageMakerSTTService(endpoint_name="endpoint", region="us-east-2")

    categories = await run_until_connect_fails(stt, [])

    assert categories == [ErrorCategory.RATE_LIMIT]
    assert not stt.is_usable


@pytest.mark.asyncio
async def test_tts_unusable_when_session_never_starts(fake_sdk):
    fake_sdk.fail_forever = throttled()
    tts = DeepgramSageMakerTTSService(endpoint_name="endpoint", region="us-east-2")

    categories = await run_until_connect_fails(tts, [])

    assert categories == [ErrorCategory.RATE_LIMIT]
    assert not tts.is_usable


@pytest.mark.asyncio
async def test_flux_tts_stays_usable_and_retries_next_turn(fake_sdk):
    # Flux TTS starts a new session on each turn, so a failed start isn't permanent.
    fake_sdk.fail_forever = throttled()
    tts = DeepgramFluxSageMakerTTSService(endpoint_name="endpoint", region="us-east-2")

    categories = await run_until_connect_fails(tts, [TTSSpeakFrame(text="hello")])

    assert ErrorCategory.RATE_LIMIT in categories
    assert fake_sdk.attempts == 8
    assert tts.is_usable


@pytest.mark.asyncio
async def test_nvidia_stt_unusable_when_session_never_starts(fake_sdk):
    fake_sdk.fail_forever = throttled()
    stt = NvidiaSageMakerSTTService(endpoint_name="endpoint", region="us-east-2")
    audio = InputAudioRawFrame(audio=b"\x00" * 3200, sample_rate=16000, num_channels=1)

    categories = await run_until_connect_fails(stt, [audio])

    assert fake_sdk.attempts == 4
    assert categories == [ErrorCategory.RATE_LIMIT]
    assert not stt.is_usable


@pytest.mark.asyncio
async def test_nvidia_tts_stays_usable_and_retries_next_turn(fake_sdk):
    # NVIDIA TTS starts a new session on the next turn, so a failed start isn't permanent.
    fake_sdk.fail_forever = throttled()
    tts = NvidiaSageMakerTTSService(endpoint_name="endpoint", region="us-east-2")

    categories = await run_until_connect_fails(tts, [TTSSpeakFrame(text="hello")])

    assert ErrorCategory.RATE_LIMIT in categories
    assert fake_sdk.attempts == 8
    assert tts.is_usable


@pytest.mark.asyncio
async def test_nvidia_tts_treats_rejected_credentials_as_recoverable(fake_sdk):
    fake_sdk.fail_forever = unmodeled_error(403, "InvalidSignatureException")
    tts = NvidiaSageMakerTTSService(endpoint_name="endpoint", region="us-east-2")

    categories = await run_until_connect_fails(tts, [])

    assert fake_sdk.attempts == 1
    assert categories == [ErrorCategory.CONNECTIVITY]
    assert tts.is_usable
