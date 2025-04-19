import os
import wave

from PIL import Image

from pipecat.frames.frames import OutputAudioRawFrame, OutputImageRawFrame

script_dir = os.path.dirname(__file__)


def load_images(image_files):
    images = {}
    for file in image_files:
        # Build the full path to the image file
        full_path = os.path.join(script_dir, "../assets", file)
        # Get the filename without the extension to use as the dictionary key
        filename = os.path.splitext(os.path.basename(full_path))[0]
        # Open the image and convert it to bytes
        with Image.open(full_path) as img:
            images[filename] = OutputImageRawFrame(
                image=img.tobytes(), size=img.size, format=img.format
            )
    return images


def load_sounds(sound_files):
    sounds = {}

    for file in sound_files:
        # Build the full path to the sound file
        full_path = os.path.join(script_dir, "../assets", file)
        # Get the filename without the extension to use as the dictionary key
        filename = os.path.splitext(os.path.basename(full_path))[0]
        # Open the sound and convert it to bytes
        with wave.open(full_path) as audio_file:
            sounds[filename] = OutputAudioRawFrame(
                audio=audio_file.readframes(-1),
                sample_rate=audio_file.getframerate(),
                num_channels=audio_file.getnchannels(),
            )

    return sounds
