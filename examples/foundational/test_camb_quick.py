#!/usr/bin/env python3
"""Quick test script to verify Camb.ai TTS integration works.

Usage:
    export CAMB_API_KEY=your_api_key
    python test_camb_quick.py
"""

import asyncio
import os
import sys

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import aiohttp
from dotenv import load_dotenv

load_dotenv()


async def test_list_voices():
    """Test listing available voices."""
    from pipecat.services.camb.tts import CambTTSService

    api_key = os.getenv("CAMB_API_KEY")
    if not api_key:
        print("ERROR: CAMB_API_KEY environment variable not set!")
        return False

    print("\n1. Testing list_voices()...")
    async with aiohttp.ClientSession() as session:
        try:
            voices = await CambTTSService.list_voices(
                api_key=api_key,
                aiohttp_session=session,
            )
            print(f"   SUCCESS: Found {len(voices)} voices")
            if voices:
                print(f"   First voice: ID={voices[0]['id']}, Name={voices[0]['name']}")
            return True
        except Exception as e:
            print(f"   FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


async def test_tts_synthesis():
    """Test basic TTS synthesis."""
    from pipecat.services.camb.tts import CambTTSService
    from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame, ErrorFrame

    api_key = os.getenv("CAMB_API_KEY")
    if not api_key:
        print("ERROR: CAMB_API_KEY environment variable not set!")
        return False

    print("\n2. Testing TTS synthesis...")
    async with aiohttp.ClientSession() as session:
        tts = CambTTSService(
            api_key=api_key,
            aiohttp_session=session,
            voice_id=2681,  # Attic voice
            model="mars-8-flash",
        )

        # Manually set sample rate (normally done by StartFrame)
        tts._sample_rate = 24000

        text = "Hello! This is a test of the Camb.ai text to speech integration."
        print(f"   Synthesizing: '{text}'")

        audio_bytes = 0
        frames_received = []

        try:
            async for frame in tts.run_tts(text):
                frames_received.append(type(frame).__name__)
                if isinstance(frame, TTSAudioRawFrame):
                    audio_bytes += len(frame.audio)
                elif isinstance(frame, ErrorFrame):
                    print(f"   FAILED: {frame.error}")
                    return False

            print(f"   Frames received: {frames_received}")
            print(f"   Audio bytes received: {audio_bytes}")

            if audio_bytes > 0:
                print("   SUCCESS: TTS synthesis works!")

                # Optionally save and play audio
                save_audio = input("\n   Save audio to test_output.wav? (y/n): ").strip().lower()
                if save_audio == 'y':
                    await save_audio_to_file(tts, text)
                    # Try to play the audio
                    play_audio = input("   Play the audio? (y/n): ").strip().lower()
                    if play_audio == 'y':
                        play_wav_file("test_output.wav")

                return True
            else:
                print("   FAILED: No audio received")
                return False

        except Exception as e:
            print(f"   FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


async def save_audio_to_file(tts, text):
    """Save synthesized audio to a WAV file."""
    import wave
    from pipecat.frames.frames import TTSAudioRawFrame

    audio_data = bytearray()
    async for frame in tts.run_tts(text):
        if isinstance(frame, TTSAudioRawFrame):
            audio_data.extend(frame.audio)

    if audio_data:
        with wave.open("test_output.wav", "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)  # 24kHz
            wav_file.writeframes(bytes(audio_data))
        print("   Saved to test_output.wav")


def play_wav_file(filepath):
    """Play a WAV file using the system's default player."""
    import subprocess
    import platform

    system = platform.system()
    try:
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", filepath], check=True)
        elif system == "Linux":
            subprocess.run(["aplay", filepath], check=True)
        elif system == "Windows":
            subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{filepath}').PlaySync()"], check=True)
        else:
            print(f"   Unsupported platform: {system}. Please play {filepath} manually.")
    except Exception as e:
        print(f"   Could not play audio: {e}")


async def main():
    print("=" * 50)
    print("Camb.ai TTS Integration Test")
    print("=" * 50)

    results = []

    # Test 1: List voices
    results.append(await test_list_voices())

    # Test 2: TTS synthesis
    results.append(await test_tts_synthesis())

    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  List voices: {'PASS' if results[0] else 'FAIL'}")
    print(f"  TTS synthesis: {'PASS' if results[1] else 'FAIL'}")
    print("=" * 50)

    if all(results):
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
