#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Native Daily audio compatibility without joining a room."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("user_tracks", [True, False])
def test_sequential_clients_can_set_up_and_write_native_audio(user_tracks):
    # Isolate Daily's process-wide initialization from tests that mock the SDK.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""\
                import asyncio
                import sys
                from types import SimpleNamespace

                from pipecat.frames.frames import OutputAudioRawFrame
                from pipecat.transports.daily.transport import DailyParams, DailyTransport
                from pipecat.utils.asyncio.task_manager import TaskManager

                async def main():
                    user_tracks = sys.argv[1] == "True"
                    for _ in range(3):
                        transport = DailyTransport(
                            "https://mock.daily.co/mock", None, "bot",
                            params=DailyParams(
                                audio_in_enabled=True, audio_out_enabled=True,
                                audio_in_user_tracks=user_tracks,
                            ),
                        )
                        client = transport._client
                        setup = SimpleNamespace(
                            task_manager=TaskManager(), audio_in_sample_rate=16000,
                            audio_out_sample_rate=24000,
                        )
                        await client.setup(setup)
                        await client.setup(setup)
                        try:
                            assert client._microphone_track is not None
                            assert (client._speaker is None) == user_tracks
                            assert await asyncio.wait_for(client.write_audio_frame(
                                OutputAudioRawFrame(
                                    audio=bytes(960), sample_rate=24000, num_channels=1,
                                )
                            ), timeout=5)
                        finally:
                            await client.cleanup()
                            await client.cleanup()

                asyncio.run(main())
                """),
            str(user_tracks),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
