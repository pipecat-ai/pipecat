#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for GenesysAudioHookSerializer wire format and protocol state."""

import json
import unittest
from datetime import timedelta

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
)
from pipecat.serializers.genesys import (
    AudioHookChannel,
    AudioHookMessageType,
    GenesysAudioHookSerializer,
)
from tests.frame_processor_helpers import frame_processor_setup

SAMPLE_RATE = 8000


def _open_message(seq: int = 1, channels: list[str] | None = None) -> str:
    """The handshake Genesys sends to start a session.

    The channel list here is authoritative: _handle_open negotiates the serializer's
    channel from it, overriding whatever InputParams asked for.
    """
    return json.dumps(
        {
            "version": "2",
            "type": "open",
            "seq": seq,
            "id": "conv-abc",
            "parameters": {
                "conversationId": "conv-abc",
                "participant": {"id": "p1", "ani": "+15551234567"},
                "media": [
                    {
                        "type": "audio",
                        "format": "PCMU",
                        "rate": SAMPLE_RATE,
                        "channels": channels or ["external"],
                    }
                ],
            },
        }
    )


class GenesysSerializerTestCase(unittest.IsolatedAsyncioTestCase):
    """Shared setup: a serializer wired to an 8 kHz pipeline."""

    params: GenesysAudioHookSerializer.InputParams | None = None

    async def asyncSetUp(self):
        self.serializer = GenesysAudioHookSerializer(params=self.params)
        await self.serializer.setup(
            frame_processor_setup(
                audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE
            )
        )

    async def _open_session(self, seq: int = 1, channels: list[str] | None = None):
        return await self.serializer.deserialize(_open_message(seq, channels))


class TestGenesysHandshake(GenesysSerializerTestCase):
    async def test_open_is_answered_with_opened(self):
        frame = await self._open_session()

        self.assertIsNotNone(frame)
        self.assertEqual(frame.message["type"], AudioHookMessageType.OPENED.value)
        self.assertTrue(self.serializer.is_open)

    async def test_open_records_the_conversation_and_participant(self):
        await self._open_session()

        self.assertEqual(self.serializer.conversation_id, "conv-abc")
        self.assertEqual(self.serializer.participant["ani"], "+15551234567")

    async def test_the_response_echoes_the_client_sequence_number(self):
        frame = await self._open_session(seq=7)

        self.assertEqual(frame.message["clientseq"], 7)

    async def test_the_server_sequence_number_increments_per_message(self):
        opened = await self._open_session()
        pong = await self.serializer.deserialize(
            json.dumps({"version": "2", "type": "ping", "seq": 2, "parameters": {}})
        )

        self.assertEqual(pong.message["seq"], opened.message["seq"] + 1)

    async def test_ping_is_answered_with_pong(self):
        await self._open_session()

        frame = await self.serializer.deserialize(
            json.dumps({"version": "2", "type": "ping", "seq": 2, "parameters": {}})
        )

        self.assertEqual(frame.message["type"], AudioHookMessageType.PONG.value)

    async def test_close_is_answered_with_closed_and_ends_the_session(self):
        await self._open_session()

        frame = await self.serializer.deserialize(
            json.dumps({"version": "2", "type": "close", "seq": 2, "parameters": {}})
        )

        self.assertEqual(frame.message["type"], AudioHookMessageType.CLOSED.value)
        self.assertFalse(self.serializer.is_open)


class TestGenesysAudio(GenesysSerializerTestCase):
    async def test_audio_round_trips_through_ulaw(self):
        audio = bytes(range(0, 256)) * 2
        await self._open_session()

        outbound = await self.serializer.serialize(
            OutputAudioRawFrame(audio=audio, sample_rate=SAMPLE_RATE, num_channels=1)
        )
        self.assertIsInstance(outbound, bytes)

        frame = await self.serializer.deserialize(outbound)
        self.assertEqual(frame.sample_rate, SAMPLE_RATE)
        self.assertEqual(frame.num_channels, 1)
        self.assertEqual(len(frame.audio), len(outbound) * 2)  # 8-bit ulaw -> 16-bit pcm

    async def test_audio_is_not_serialized_before_the_session_opens(self):
        frame = OutputAudioRawFrame(
            audio=b"\x00\x00" * 160, sample_rate=SAMPLE_RATE, num_channels=1
        )

        self.assertIsNone(await self.serializer.serialize(frame))

    async def test_audio_is_not_deserialized_before_the_session_opens(self):
        self.assertIsNone(await self.serializer.deserialize(b"\xff" * 160))

    async def test_a_paused_session_drops_audio_in_both_directions(self):
        await self._open_session()
        await self.serializer.deserialize(
            json.dumps({"version": "2", "type": "pause", "seq": 2, "parameters": {}})
        )

        self.assertTrue(self.serializer.is_paused)
        self.assertIsNone(await self.serializer.deserialize(b"\xff" * 160))
        self.assertIsNone(
            await self.serializer.serialize(
                OutputAudioRawFrame(
                    audio=b"\x00\x00" * 160, sample_rate=SAMPLE_RATE, num_channels=1
                )
            )
        )


class TestGenesysStereo(GenesysSerializerTestCase):
    params = GenesysAudioHookSerializer.InputParams(channel=AudioHookChannel.BOTH)

    async def test_the_external_channel_is_taken_from_interleaved_audio(self):
        """BOTH arrives interleaved as [L0, R0, L1, R1, ...]; only the left channel is kept."""
        await self._open_session(channels=["external", "internal"])
        interleaved = bytes([0x10, 0x20] * 80)

        frame = await self.serializer.deserialize(interleaved)

        # Half the bytes survive de-interleaving, then ulaw->pcm doubles them.
        self.assertEqual(len(frame.audio), len(interleaved))


class TestGenesysControlMessages(GenesysSerializerTestCase):
    async def test_dtmf_becomes_an_input_frame(self):
        await self._open_session()

        frame = await self.serializer.deserialize(
            json.dumps({"version": "2", "type": "dtmf", "seq": 2, "parameters": {"digit": "5"}})
        )

        self.assertIsInstance(frame, InputDTMFFrame)

    async def test_an_unparseable_message_is_ignored(self):
        self.assertIsNone(await self.serializer.deserialize("{not json"))

    async def test_an_unknown_message_type_is_ignored(self):
        await self._open_session()

        frame = await self.serializer.deserialize(
            json.dumps({"version": "2", "type": "not-a-real-type", "seq": 2, "parameters": {}})
        )

        self.assertIsNone(frame)

    async def test_position_from_the_client_is_tracked(self):
        await self._open_session()

        await self.serializer.deserialize(
            json.dumps(
                {"version": "2", "type": "ping", "seq": 2, "position": "PT2.500S", "parameters": {}}
            )
        )

        self.assertEqual(self.serializer._position, timedelta(seconds=2.5))

    async def test_a_malformed_position_falls_back_to_zero(self):
        await self._open_session()

        await self.serializer.deserialize(
            json.dumps(
                {"version": "2", "type": "ping", "seq": 2, "position": "garbage", "parameters": {}}
            )
        )

        self.assertEqual(self.serializer._position, timedelta(0))


class TestGenesysOutboundControl(GenesysSerializerTestCase):
    async def test_interruption_becomes_a_barge_in_event(self):
        await self._open_session()

        message = json.loads(await self.serializer.serialize(InterruptionFrame()))

        self.assertEqual(message["type"], AudioHookMessageType.EVENT.value)

    async def test_end_frame_becomes_a_disconnect(self):
        await self._open_session()

        message = json.loads(await self.serializer.serialize(EndFrame()))

        self.assertEqual(message["type"], AudioHookMessageType.DISCONNECT.value)

    async def test_cancel_frame_becomes_a_disconnect(self):
        await self._open_session()

        message = json.loads(await self.serializer.serialize(CancelFrame()))

        self.assertEqual(message["type"], AudioHookMessageType.DISCONNECT.value)

    async def test_output_variables_reach_the_disconnect_message(self):
        await self._open_session()
        self.serializer.set_output_variables({"outcome": "resolved"})

        message = json.loads(await self.serializer.serialize(EndFrame()))

        self.assertEqual(message["parameters"]["outputVariables"]["outcome"], "resolved")


if __name__ == "__main__":
    unittest.main()
