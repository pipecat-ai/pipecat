#!/usr/bin/env python3
"""IBM Speech Services TTS test with text-to-speech synthesis"""
import asyncio
import os
import sys
import wave

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipecat.services.ibm.tts import IBMTTSService
from pipecat.transcriptions.language import Language


async def test_ibm_tts():
    """Test IBM TTS with text synthesis"""
    
    api_key = os.getenv("IBM_TTS_API_KEY")
    url = os.getenv("IBM_TTS_URL", "https://api.us-south.text-to-speech.watson.cloud.ibm.com")
    
    if not api_key:
        print("❌ IBM_TTS_API_KEY not set")
        print("\nSet it with:")
        print("export IBM_TTS_API_KEY='your-api-key'")
        return False
    
    print("=" * 70)
    print("IBM Speech Services TTS Audio Synthesis Test")
    print("=" * 70)
    print(f"✓ API Key: {api_key[:10]}...")
    print(f"✓ URL: {url}")
    print()
    
    # Test phrases
    test_phrases = [
        "Hello, this is a test of IBM Speech Services Text to Speech.",
        "The quick brown fox jumps over the lazy dog.",
        "Testing speech synthesis with multiple sentences. This is the second sentence.",
    ]
    
    # Allow user to provide custom text
    if len(sys.argv) > 1:
        test_phrases = [" ".join(sys.argv[1:])]
        print(f"📝 Using custom text: \"{test_phrases[0]}\"")
    else:
        print(f"📝 Using test phrase: \"{test_phrases[0]}\"")
    
    print()
    
    try:
        # Test different voices and configurations
        test_configs = [
            {
                "name": "US English Female (Ellie)",
                "voice": "en-US_EllieNatural",
                "language": Language.EN_US,
                "accept": "audio/l16;rate=24000",
            },
            {
                "name": "US English Male (Ethan)",
                "voice": "en-US_EthanNatural",
                "language": Language.EN_US,
                "accept": "audio/l16;rate=24000",
            },
            {
                "name": "UK English Female (Chloe)",
                "voice": "en-GB_ChloeNatural",
                "language": Language.EN_GB,
                "accept": "audio/l16;rate=24000",
            },
        ]
        
        # Let user choose voice or test first one
        config_idx = 0
        if len(sys.argv) > 1 and sys.argv[1].isdigit():
            config_idx = int(sys.argv[1]) - 1
            if config_idx < 0 or config_idx >= len(test_configs):
                config_idx = 0
        
        config = test_configs[config_idx]
        
        print(f"1. Testing voice: {config['name']}")
        print(f"   Voice ID: {config['voice']}")
        print(f"   Audio format: {config['accept']}")
        print()
        
        # Create TTS service
        print("2. Creating IBM TTS service...")
        tts = IBMTTSService(
            api_key=api_key,
            url=url,
            params=IBMTTSService.InputParams(
                voice=config['voice'],
                language=config['language'],
                accept=config['accept'],
            )
        )
        print(f"   ✓ Service created (sample rate: {tts._sample_rate} Hz)")
        
        # Synthesize speech
        print(f"\n3. Synthesizing speech...")
        print(f"   Text: \"{test_phrases[0]}\"")
        
        audio_chunks = []
        chunk_count = 0
        total_bytes = 0
        
        # Run TTS and collect audio
        async for frame in tts.run_tts(test_phrases[0], context_id="test-context"):
            from pipecat.frames.frames import (
                ErrorFrame,
                TTSAudioRawFrame,
                TTSStartedFrame,
                TTSStoppedFrame,
            )
            
            if isinstance(frame, TTSStartedFrame):
                print("   🎤 TTS started...")
            elif isinstance(frame, TTSAudioRawFrame):
                audio_chunks.append(frame.audio)
                chunk_count += 1
                total_bytes += len(frame.audio)
                if chunk_count % 5 == 0:
                    print(f"   📦 Received {chunk_count} audio chunks ({total_bytes:,} bytes)...")
            elif isinstance(frame, TTSStoppedFrame):
                print(f"   ✓ TTS completed")
            elif isinstance(frame, ErrorFrame):
                print(f"   ❌ Error: {frame.error}")
                return False
        
        # Combine audio chunks
        print(f"\n4. Processing audio...")
        print(f"   Total chunks: {chunk_count}")
        print(f"   Total bytes: {total_bytes:,}")
        
        if not audio_chunks:
            print("   ❌ No audio data received")
            return False
        
        combined_audio = b''.join(audio_chunks)
        duration = len(combined_audio) / (tts._sample_rate * 2)  # 16-bit = 2 bytes per sample
        print(f"   ✓ Combined audio: {len(combined_audio):,} bytes")
        print(f"   ✓ Duration: {duration:.2f} seconds")
        
        # Save to WAV file
        output_file = "test_tts_output.wav"
        print(f"\n5. Saving audio to {output_file}...")
        
        with wave.open(output_file, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(tts._sample_rate)
            wav_file.writeframes(combined_audio)
        
        file_size = os.path.getsize(output_file)
        print(f"   ✓ Saved {file_size:,} bytes to {output_file}")
        
        # Results
        print("\n" + "=" * 70)
        print("Test Results")
        print("=" * 70)
        print(f"✅ Voice: {config['name']}")
        print(f"✅ Audio chunks: {chunk_count}")
        print(f"✅ Audio size: {total_bytes:,} bytes")
        print(f"✅ Duration: {duration:.2f} seconds")
        print(f"✅ Sample rate: {tts._sample_rate} Hz")
        print(f"✅ Output file: {output_file}")
        print()
        print("💡 Play the audio with:")
        print(f"   afplay {output_file}  # macOS")
        print(f"   aplay {output_file}   # Linux")
        print(f"   start {output_file}   # Windows")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ Test FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n💡 USAGE:")
    print(f"   python3 {sys.argv[0]}                           # Use default test phrase")
    print(f"   python3 {sys.argv[0]} \"Your custom text here\"   # Synthesize custom text")
    print()
    print("   Available voices:")
    print("   1. US English Male (Michael)")
    print("   2. US English Female (Allison)")
    print("   3. UK English Female (Charlotte)")
    print()
    
    success = asyncio.run(test_ibm_tts())
    sys.exit(0 if success else 1)

