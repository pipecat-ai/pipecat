#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for TwilioFrameSerializer.

deserialize() ignores malformed JSON and messages missing expected fields
rather than raising, since the transport's receive loop ends the call on any
exception. With auto_hang_up enabled, the constructor requires exactly one
credential type: auth_token or the api_key_sid + api_key_secret pair. The
hang-up request authenticates with the matching Basic Auth pair, while
account_sid stays in the URL.
"""

import base64
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import InputAudioRawFrame, InputDTMFFrame, OutputTransportMessageFrame
from pipecat.serializers.twilio import TwilioFrameSerializer
from tests.frame_processor_helpers import frame_processor_setup

STREAM_SID = "stream123"
SAMPLE_RATE = 8000
VALID_PAYLOAD = base64.b64encode(b"\xff" * 160).decode()

CALL_SID = "CAyyyy"
ACCOUNT_SID = "ACxxxx"
AUTH_TOKEN = "Auth_token"
API_KEY_SID = "SKxxxx"
API_KEY_SECRET = "secret"
BASE_URL = "https://api.example.test"
CREDS = dict(call_sid=CALL_SID, account_sid=ACCOUNT_SID)


def _make_client_session() -> MagicMock:
    response = MagicMock(status=200)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    session.post = MagicMock(return_value=response)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    return session


class TestTwilioDeserializeRobustness(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = TwilioFrameSerializer(
            stream_sid=STREAM_SID, params=TwilioFrameSerializer.InputParams(auto_hang_up=False)
        )
        await self.serializer.setup(
            frame_processor_setup(
                audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE
            )
        )

    async def test_malformed_json_is_ignored_not_raised(self):
        frame = await self.serializer.deserialize("not json")
        self.assertIsNone(frame)

    async def test_media_event_missing_payload_is_ignored_not_raised(self):
        frame = await self.serializer.deserialize(json.dumps({"event": "media", "media": {}}))
        self.assertIsNone(frame)

    async def test_message_missing_event_key_is_ignored_not_raised(self):
        frame = await self.serializer.deserialize(json.dumps({"foo": "bar"}))
        self.assertIsNone(frame)

    async def test_non_object_json_is_ignored_not_raised(self):
        for name, data in (
            ("list", "[]"),
            ("string", '"str"'),
            ("number", "123"),
            ("null", "null"),
        ):
            with self.subTest(name=name):
                self.assertIsNone(await self.serializer.deserialize(data))

    async def test_non_utf8_bytes_are_ignored_not_raised(self):
        self.assertIsNone(await self.serializer.deserialize(b"\xc3\x28"))

    async def test_media_event_with_malformed_media_is_ignored_not_raised(self):
        for name, media in (("null", None), ("string", "oops"), ("list", ["oops"])):
            with self.subTest(name=name):
                data = json.dumps({"event": "media", "media": media})
                self.assertIsNone(await self.serializer.deserialize(data))

    async def test_media_event_with_undecodable_payload_is_ignored_not_raised(self):
        for name, payload in (
            ("invalid_base64", "!!!not-base64"),
            ("bad_padding", "AAAAA"),
            ("non_ascii", "\u00e9\u00e9\u00e9\u00e9"),
            ("number", 123),
            ("list", ["AAAA"]),
        ):
            with self.subTest(name=name):
                data = json.dumps({"event": "media", "media": {"payload": payload}})
                self.assertIsNone(await self.serializer.deserialize(data))

    async def test_dtmf_event_with_malformed_dtmf_is_ignored_not_raised(self):
        for name, dtmf in (
            ("null", None),
            ("string", "5"),
            ("missing_digit", {}),
            ("null_digit", {"digit": None}),
            ("unknown_digit", {"digit": "x"}),
        ):
            with self.subTest(name=name):
                data = json.dumps({"event": "dtmf", "dtmf": dtmf})
                self.assertIsNone(await self.serializer.deserialize(data))

    async def test_valid_media_event_is_still_deserialized(self):
        data = json.dumps({"event": "media", "media": {"payload": VALID_PAYLOAD}})
        frame = await self.serializer.deserialize(data)
        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.sample_rate, SAMPLE_RATE)

    async def test_valid_dtmf_event_is_still_deserialized(self):
        data = json.dumps({"event": "dtmf", "dtmf": {"digit": "5"}})
        frame = await self.serializer.deserialize(data)
        self.assertIsInstance(frame, InputDTMFFrame)
        self.assertEqual(frame.button, KeypadEntry.FIVE)


class TestTwilioSerializeTransportMessages(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = TwilioFrameSerializer(
            stream_sid=STREAM_SID, params=TwilioFrameSerializer.InputParams(auto_hang_up=False)
        )
        await self.serializer.setup(
            frame_processor_setup(
                audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE
            )
        )

    async def test_non_dict_transport_message_does_not_raise(self):
        result = await self.serializer.serialize(OutputTransportMessageFrame(message=["a"]))
        self.assertEqual(json.loads(result), ["a"])

    async def test_rtvi_transport_message_is_ignored(self):
        message = {"label": "rtvi-ai", "type": "bot-ready"}
        result = await self.serializer.serialize(OutputTransportMessageFrame(message=message))
        self.assertIsNone(result)


class TestConstructorCredentialValidation(unittest.TestCase):
    def test_auth_token_alone_is_accepted(self):
        serializer = TwilioFrameSerializer(STREAM_SID, auth_token=AUTH_TOKEN, **CREDS)
        self.assertEqual(serializer._auth_token, AUTH_TOKEN)

    def test_api_key_pair_alone_is_accepted(self):
        for name, token_part in (
            ("token_omitted", {}),
            ("empty_token", {"auth_token": ""}),
        ):
            with self.subTest(name=name):
                serializer = TwilioFrameSerializer(
                    STREAM_SID,
                    api_key_sid=API_KEY_SID,
                    api_key_secret=API_KEY_SECRET,
                    **token_part,
                    **CREDS,
                )
                self.assertEqual(serializer._api_key_sid, API_KEY_SID)
                self.assertEqual(serializer._api_key_secret, API_KEY_SECRET)

    def test_api_key_sid_without_secret_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing: api_key_secret"):
            TwilioFrameSerializer(STREAM_SID, api_key_sid=API_KEY_SID, **CREDS)

    def test_api_key_secret_without_sid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing: api_key_sid"):
            TwilioFrameSerializer(STREAM_SID, api_key_secret=API_KEY_SECRET, **CREDS)

    def test_auth_token_with_any_api_key_part_is_rejected(self):
        for name, key_part in (
            ("sid_only", {"api_key_sid": API_KEY_SID}),
            ("secret_only", {"api_key_secret": API_KEY_SECRET}),
            ("full_pair", {"api_key_sid": API_KEY_SID, "api_key_secret": API_KEY_SECRET}),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "not both"):
                    TwilioFrameSerializer(STREAM_SID, auth_token=AUTH_TOKEN, **key_part, **CREDS)

    def test_no_credentials_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, r"missing required parameters: auth_token or api_key_sid \+ api_key_secret"
        ):
            TwilioFrameSerializer(STREAM_SID, **CREDS)


class TestHangUpBasicAuth(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.session = _make_client_session()

    async def test_basic_auth_with_auth_token(self):
        serializer = TwilioFrameSerializer(STREAM_SID, auth_token=AUTH_TOKEN, **CREDS)

        with patch("aiohttp.ClientSession", return_value=self.session):
            await serializer._hang_up_call()

        self.session.post.assert_called_once()

        auth = self.session.post.call_args.kwargs["auth"]
        self.assertEqual(auth.login, ACCOUNT_SID)
        self.assertEqual(auth.password, AUTH_TOKEN)

    async def test_basic_auth_with_api_key(self):
        serializer = TwilioFrameSerializer(
            STREAM_SID, api_key_sid=API_KEY_SID, api_key_secret=API_KEY_SECRET, **CREDS
        )

        with patch("aiohttp.ClientSession", return_value=self.session):
            await serializer._hang_up_call()

        self.session.post.assert_called_once()

        auth = self.session.post.call_args.kwargs["auth"]
        self.assertEqual(auth.login, API_KEY_SID)
        self.assertEqual(auth.password, API_KEY_SECRET)

        url = self.session.post.call_args.args[0]
        self.assertIn(f"/Accounts/{ACCOUNT_SID}/", url)

    async def test_base_url_working_with_api_key(self):
        serializer = TwilioFrameSerializer(
            STREAM_SID,
            api_key_sid=API_KEY_SID,
            api_key_secret=API_KEY_SECRET,
            base_url=BASE_URL,
            **CREDS,
        )

        with patch("aiohttp.ClientSession", return_value=self.session):
            await serializer._hang_up_call()

        self.session.post.assert_called_once()

        auth = self.session.post.call_args.kwargs["auth"]
        self.assertEqual(auth.login, API_KEY_SID)
        self.assertEqual(auth.password, API_KEY_SECRET)

        url = self.session.post.call_args.args[0]
        self.assertTrue(url.startswith(f"{BASE_URL}/"), url)


if __name__ == "__main__":
    unittest.main()
