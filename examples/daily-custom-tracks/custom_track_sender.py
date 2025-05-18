#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import time

from daily import CallClient, CustomAudioSource, Daily
from pydub import AudioSegment

parser = argparse.ArgumentParser(description="Daily AI SDK Bot Sample")
parser.add_argument("-u", "--url", type=str, required=True, help="URL of the Daily room to join")
parser.add_argument(
    "-i", "--input", type=str, required=True, help="Input audio file (needs 16000 sample rate)"
)

args, _ = parser.parse_known_args()

audio = AudioSegment.from_mp3(args.input)

raw_bytes = audio.raw_data
sample_rate = audio.frame_rate
channels = audio.channels

print(f"Length: {len(raw_bytes)} bytes")
print(f"Sample rate: {sample_rate}, Channels: {channels}")

# Initialize the Daily context & create call client
Daily.init()

client = CallClient()

# Join the room and indicate we have a custom track named "pipecat".
client.join(
    args.url,
    client_settings={
        "publishing": {
            "camera": False,
            "microphone": False,
            "customAudio": {"pipecat": True},
        },
    },
)

# Just sleep for a couple of seconds. To do this well we should really use
# completions.
time.sleep(2)

# Create the custom audio source. This is where we will write our audio.
audio_source = CustomAudioSource(sample_rate, channels)

# Create an audio track and assign it our audio source.
client.add_custom_audio_track("pipecat", audio_source)

# Just sleep for a second. To do this well we should really use completions.
time.sleep(1)

try:
    # Just write one second of audio until we have read all the file.
    chunk_size = sample_rate * channels * 2
    while len(raw_bytes) > 0:
        chunk = raw_bytes[:chunk_size]
        raw_bytes = raw_bytes[chunk_size:]
        audio_source.write_frames(chunk)

except KeyboardInterrupt:
    client.leave()

# Just sleep for a second. To do this well we should really use completions.
time.sleep(1)

client.release()
