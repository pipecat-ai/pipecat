#!/usr/bin/env python3
"""IBM STT test with downloadable real speech audio"""
import asyncio
import os
import sys
import urllib.request

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipecat.services.ibm.stt import IBMSTTService
from pipecat.transcriptions.language import Language


def download_sample_audio():
    """Download a sample speech audio file for testing.
    
    Returns:
        str: Path to downloaded audio file, or None if download fails
    """
    # LibriVox public domain audio sample (16kHz, mono, WAV)
    sample_url = "https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav"
    output_file = "test_sample.wav"
    
    try:
        print("   📥 Downloading sample audio from LibriVox...")
        urllib.request.urlretrieve(sample_url, output_file)
        print(f"   ✓ Downloaded to {output_file}")
        return output_file
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return None

async def test_ibm_real_audio():
    """Test IBM STT with real speech audio"""
    
    api_key = os.getenv("IBM_STT_API_KEY")
    url = os.getenv("IBM_STT_URL", "https://api.us-south.speech-to-text.watson.cloud.ibm.com")
    
    if not api_key:
        print("❌ IBM_STT_API_KEY not set")
        return False
    
    print("=" * 70)
    print("IBM STT Real Speech Audio Test")
    print("=" * 70)
    print(f"✓ API Key: {api_key[:10]}...")
    print(f"✓ URL: {url}")
    print()
    
    try:
        # Get audio file
        print("1. Obtaining test audio...")
        
        # Check if user provided their own audio file
        if len(sys.argv) > 1:
            audio_file = sys.argv[1]
            if not os.path.exists(audio_file):
                print(f"   ❌ File not found: {audio_file}")
                return False
            print(f"   ✓ Using provided file: {audio_file}")
        else:
            # Download sample
            audio_file = download_sample_audio()
            if not audio_file:
                print("\n   💡 TIP: You can provide your own WAV file:")
                print(f"      python3 {sys.argv[0]} your_audio.wav")
                return False
        
        # Read audio
        print(f"\n2. Reading audio file...")
        import wave
        with wave.open(audio_file, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            duration = n_frames / sample_rate
            
            print(f"   ✓ Sample rate: {sample_rate} Hz")
            print(f"   ✓ Channels: {n_channels}")
            print(f"   ✓ Sample width: {sample_width} bytes")
            print(f"   ✓ Duration: {duration:.2f} seconds")
            print(f"   ✓ Frames: {n_frames:,}")
            
            # Convert to mono if stereo
            audio_data = wav_file.readframes(n_frames)
            if n_channels == 2:
                print("   ℹ️  Converting stereo to mono...")
                import struct
                samples = struct.unpack(f'<{n_frames * 2}h', audio_data)
                mono_samples = [int((samples[i] + samples[i+1]) / 2) 
                               for i in range(0, len(samples), 2)]
                audio_data = struct.pack(f'<{len(mono_samples)}h', *mono_samples)
                print(f"   ✓ Converted to mono ({len(audio_data)} bytes)")
        
        # Create IBM STT
        print("\n3. Creating IBM STT service...")
        stt = IBMSTTService(
            api_key=api_key,
            url=url,
            model="en-US",
            sample_rate=sample_rate,
            params=IBMSTTService.InputParams(
                language=Language.EN_US,
                interim_results=True,
                timestamps=True,
                smart_formatting=True,
                word_confidence=True
            )
        )
        print(f"   ✓ Service created (sample rate: {stt.sample_rate} Hz)")
        
        # Connect
        print("\n4. Connecting to IBM STT...")
        await stt._connect_websocket()
        print("   ✓ Connected")
        
        # Collect results
        transcriptions = []
        interim_count = 0
        
        async def collect_results():
            nonlocal interim_count
            try:
                async for message in stt._get_websocket():
                    import json
                    data = json.loads(message)
                    
                    if "state" in data:
                        state = data["state"]
                        if state == "listening":
                            print(f"   📡 IBM is listening...")
                    
                    if "results" in data:
                        for result in data["results"]:
                            if "alternatives" in result:
                                alt = result["alternatives"][0]
                                text = alt.get("transcript", "").strip()
                                is_final = result.get("final", False)
                                confidence = alt.get("confidence")
                                
                                if text:
                                    if is_final:
                                        transcriptions.append({
                                            "text": text,
                                            "confidence": confidence,
                                            "timestamps": alt.get("timestamps", [])
                                        })
                                        conf_str = f" [{confidence:.1%}]" if confidence else ""
                                        print(f"   ✅ FINAL: \"{text}\"{conf_str}")
                                    else:
                                        interim_count += 1
                                        if interim_count % 5 == 0:  # Show every 5th interim
                                            print(f"   💭 Interim ({interim_count}): \"{text[:50]}...\"")
            except Exception as e:
                if "close" not in str(e).lower():
                    print(f"   ⚠️  Receiver error: {e}")
        
        receive_task = asyncio.create_task(collect_results())
        
        # Send audio
        print("\n5. Streaming audio to IBM STT...")
        chunk_size = 16000  # ~0.5 seconds at 16kHz, 16-bit
        total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            await stt._websocket.send(chunk)
            
            # Show progress every 10 chunks
            chunk_num = i // chunk_size
            if chunk_num % 10 == 0:
                progress = (i / len(audio_data)) * 100
                print(f"   📤 Sent {chunk_num}/{total_chunks} chunks ({progress:.0f}%)")
            
            await asyncio.sleep(0.05)  # Small delay to simulate real-time
        
        print(f"   ✓ Sent all audio ({len(audio_data):,} bytes)")
        
        # Finalize
        print("\n6. Finalizing transcription...")
        import json
        await stt._websocket.send(json.dumps({"action": "stop"}))
        print("   ✓ Stop message sent")
        
        # Wait for final results
        print("\n7. Waiting for final transcriptions (5 seconds)...")
        await asyncio.sleep(5)
        
        # Cleanup
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass
        
        await stt._disconnect_websocket()
        print("   ✓ Disconnected")
        
        # Display results
        print("\n" + "=" * 70)
        print("TRANSCRIPTION RESULTS")
        print("=" * 70)
        print(f"Interim transcriptions: {interim_count}")
        print(f"Final transcriptions: {len(transcriptions)}")
        
        if transcriptions:
            print("\n📝 Full Transcription:")
            print("-" * 70)
            full_text = " ".join([t["text"] for t in transcriptions])
            print(full_text)
            print("-" * 70)
            
            print("\n📊 Detailed Results:")
            for i, result in enumerate(transcriptions, 1):
                conf = result["confidence"]
                conf_str = f" [{conf:.1%}]" if conf else ""
                print(f"\n{i}. {result['text']}{conf_str}")
                
                # Show first few word timestamps
                if result["timestamps"] and len(result["timestamps"]) > 0:
                    print(f"   Timestamps (first 3 words):")
                    for word, start, end in result["timestamps"][:3]:
                        print(f"     '{word}': {start:.2f}s - {end:.2f}s")
        else:
            print("\n⚠️  No transcriptions received")
            print("\nPossible reasons:")
            print("  • Audio format not compatible (needs 16kHz, mono, 16-bit PCM)")
            print("  • Audio too short or too quiet")
            print("  • Network issues")
            print(f"\n💡 Try with your own audio file:")
            print(f"   python3 {sys.argv[0]} your_speech.wav")
        
        print("\n" + "=" * 70)
        if transcriptions:
            print("✅ TEST PASSED - Transcriptions received!")
        else:
            print("⚠️  TEST COMPLETED - No transcriptions (check audio format)")
        print("=" * 70)
        
        # Cleanup downloaded file
        if audio_file == "test_sample.wav" and os.path.exists(audio_file):
            os.remove(audio_file)
        
        return len(transcriptions) > 0
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n💡 USAGE:")
    print(f"   python3 {sys.argv[0]}                    # Download sample audio")
    print(f"   python3 {sys.argv[0]} your_audio.wav     # Use your own audio\n")
    print("   Audio requirements: WAV format, 16kHz, mono, 16-bit PCM\n")
    
    success = asyncio.run(test_ibm_real_audio())
    sys.exit(0 if success else 1)

# Made with Bob
