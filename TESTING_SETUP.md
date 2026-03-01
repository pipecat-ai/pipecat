# AssemblyAI u3-rt-pro Testing Setup Guide

## Quick Start

### 1. Setup Environment

```bash
# Copy API keys
cp .env.testing .env

# Install dependencies
uv sync --group dev --all-extras --no-extra gstreamer --no-extra krisp

# Make test script executable
chmod +x test_assemblyai_u3pro.py
```

### 2. Ensure Audio Devices

Make sure you have:
- **Microphone** enabled and working
- **Speakers/headphones** connected
- Audio permissions granted (macOS will prompt on first run)

### 3. Run Tests

```bash
# Run a specific test
python test_assemblyai_u3pro.py --test basic

# Interactive mode (choose from menu)
python test_assemblyai_u3pro.py --interactive

# Run all tests sequentially
python test_assemblyai_u3pro.py --test all
```

---

## Available Tests

### Basic Configuration Tests
```bash
# Test 1: Default configuration (min=max=100ms)
python test_assemblyai_u3pro.py --test basic

# Test 2: Custom min_turn_silence
python test_assemblyai_u3pro.py --test custom_min

# Test 3: max_turn_silence warning (should be overridden)
python test_assemblyai_u3pro.py --test max_warning
```

### Prompting Tests
```bash
# Test 5: Custom prompt warning
python test_assemblyai_u3pro.py --test prompt_warning

# Test 6: Prompt + keyterms conflict (should error)
python test_assemblyai_u3pro.py --test prompt_keyterms_conflict

# Test 7: Basic keyterms prompting
python test_assemblyai_u3pro.py --test keyterms
```

### Diarization Tests
```bash
# Test 10: Diarization without formatting
python test_assemblyai_u3pro.py --test diarization

# Test 11: Diarization with XML formatting
python test_assemblyai_u3pro.py --test diarization_xml
```

### Dynamic Updates Tests
```bash
# Test 13: Dynamic keyterms (multi-stage)
python test_assemblyai_u3pro.py --test dynamic_keyterms

# Test 15: Dynamic silence parameters
python test_assemblyai_u3pro.py --test dynamic_silence

# Test 17: Multiple parameters at once
python test_assemblyai_u3pro.py --test multi_param
```

---

## Test Execution Flow

### For Each Test:

1. **Start the test script**
   ```bash
   python test_assemblyai_u3pro.py --test <test_name>
   ```

2. **Wait for "started" message** indicating the bot is ready

3. **Speak into your microphone** to test - the bot will:
   - Transcribe your speech (you'll see `📝 TRANSCRIPTION:` logs)
   - Process through the LLM
   - Respond with voice through your speakers

4. **Observe logs** for:
   - ✅ Success indicators
   - ⚠️ Warning messages
   - ❌ Error messages
   - 📝 Transcription output

5. **Verify expected behavior** against checklist

6. **Stop test** with Ctrl+C

---

## Expected Test Outcomes

### Should Pass (✅)
- Basic configuration creates service
- Custom parameters are applied
- Keyterms boost recognition
- Diarization shows speaker IDs
- Dynamic updates work without errors

### Should Warn (⚠️)
Check logs for warnings:
- "We recommend testing at first with no prompt"
- "max_turn_silence is not used in Pipecat mode"
- "Unknown setting for AssemblyAI STT service"

### Should Error (❌)
Should raise ValueError and fail to start:
- Both prompt and keyterms_prompt set at init
- Both prompt and keyterms_prompt in same update
- vad_force_turn_endpoint=False with universal-streaming

---

## Debugging Tips

### Check Logs
```bash
# Run with verbose logging
LOGURU_LEVEL=DEBUG python test_assemblyai_u3pro.py --test <test_name>
```

### Common Issues

**Issue: "WebSocket connection failed"**
- Check ASSEMBLYAI_API_KEY is correct
- Verify network connection
- Check firewall settings

**Issue: "No audio input/output"**
- Verify microphone permissions (System Preferences → Security & Privacy → Microphone)
- Check default audio devices in System Preferences → Sound
- Test microphone with another app first
- Make sure no other app is using the microphone

**Issue: "No transcriptions appearing"**
- Verify microphone permissions
- Check audio levels (speak louder or move closer to mic)
- Speak clearly and wait for VAD to detect
- Check if microphone is muted

**Issue: "Can't hear bot responses"**
- Check speaker/headphone volume
- Verify correct output device is selected
- Check terminal for TTS errors

**Issue: "Service fails to start"**
- Check all API keys in .env
- Run `uv sync` to ensure dependencies installed
- Check Python version (3.10+)

---

## Manual Testing Checklist

After running automated tests, manually verify:

### ✅ Audio Quality
- [ ] Transcriptions are accurate
- [ ] No distortion or dropouts
- [ ] Latency is acceptable

### ✅ Turn Detection
- [ ] Bot waits for user to finish speaking
- [ ] No premature cutoffs
- [ ] Handles natural pauses correctly

### ✅ Interruptions
- [ ] Can interrupt bot mid-sentence
- [ ] Interruption is smooth
- [ ] Bot stops speaking immediately

### ✅ Diarization (if enabled)
- [ ] Multiple speakers detected correctly
- [ ] Speaker IDs consistent
- [ ] Speaker formatting works

### ✅ Dynamic Updates
- [ ] Keyterms update without disconnection
- [ ] Turn detection timing changes work
- [ ] Updates logged correctly

---

## Test Results Recording

### Use this template:

```markdown
## Test Run: YYYY-MM-DD

| Test # | Test Name | Status | Notes |
|--------|-----------|--------|-------|
| 1 | basic | ✅ PASS | Transcriptions working |
| 2 | custom_min | ✅ PASS | Turn timing changed |
| 3 | max_warning | ✅ PASS | Warning logged |
| 5 | prompt_warning | ✅ PASS | Warning shown |
| 6 | prompt_keyterms_conflict | ✅ PASS | ValueError raised |
| 7 | keyterms | ✅ PASS | Terms boosted |
| 10 | diarization | ✅ PASS | Speaker IDs correct |
| 11 | diarization_xml | ✅ PASS | XML tags shown |
| 13 | dynamic_keyterms | ✅ PASS | Updates worked |
| 15 | dynamic_silence | ✅ PASS | Timing adjusted |
| 17 | multi_param | ✅ PASS | All params updated |

### Issues Found:
- None

### Notes:
- All tests passed successfully
- Latency is excellent (sub-300ms)
- Diarization accuracy is good
```

---

## Advanced Testing

### Custom Test Scenarios

Create custom tests by modifying `test_assemblyai_u3pro.py`:

```python
async def test_my_custom_scenario():
    """My custom test scenario."""
    logger.info("Testing my specific use case")

    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        # Your custom params here
    )

    task, transport = await create_basic_voice_agent(connection_params)

    # Your test logic here

    runner = PipelineRunner()
    await runner.run(task)
```

### Stress Testing

Test with:
- Multiple simultaneous speakers
- Long conversations (30+ minutes)
- Rapid speech
- Heavy accents
- Background noise
- Poor network conditions

---

## Reporting Issues

When reporting issues, include:

1. **Test name and number**
2. **Full error message and stack trace**
3. **Relevant log output** (use LOGURU_LEVEL=DEBUG)
4. **Configuration used** (connection_params)
5. **Expected vs actual behavior**
6. **Steps to reproduce**

---

## Next Steps

After testing:

1. ✅ Mark completed tests in `TESTING_CHECKLIST.md`
2. 📝 Document any issues found
3. 🐛 Create GitHub issues for bugs
4. ✨ Suggest improvements
5. 📊 Share results with team

---

## Contact

Questions? Issues?
- Check `TESTING_CHECKLIST.md` for detailed test descriptions
- Review logs with `LOGURU_LEVEL=DEBUG`
- Reach out to the team with your findings

Happy testing! 🎯
