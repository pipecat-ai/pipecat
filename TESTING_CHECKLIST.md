# AssemblyAI u3-rt-pro Testing Checklist

## Test Environment Setup
- [ ] Install dependencies: `uv sync --group dev --all-extras`
- [ ] Set up `.env` file with API keys
- [ ] Verify LiveKit connection
- [ ] Run basic voice agent test

---

## Feature Testing Checklist

### ✅ Basic Configuration Tests

#### Test 1: Default u3-rt-pro Configuration
- [ ] **Setup:** Create service with default params
- [ ] **Expected:** No errors, uses u3-rt-pro model with 100ms min/max
- [ ] **Verify:** Check logs for connection confirmation

#### Test 2: Custom min_turn_silence
- [ ] **Setup:** Set `min_turn_silence=200`
- [ ] **Expected:** Both min and max set to 200ms
- [ ] **Verify:** Speak short phrases, observe turn detection timing

#### Test 3: User sets max_turn_silence (Warning Test)
- [ ] **Setup:** Set `max_turn_silence=500` in connection params
- [ ] **Expected:** Warning logged, value overridden to match min
- [ ] **Verify:** Check logs for warning message

---

### ✅ Prompting Tests

#### Test 4: No Prompt (Default - Recommended)
- [ ] **Setup:** Don't set prompt parameter
- [ ] **Expected:** Uses default prompt, 88% accuracy, no warnings
- [ ] **Verify:** Transcription quality is good

#### Test 5: Custom Prompt (Warning Test)
- [ ] **Setup:** Set custom prompt in connection params
- [ ] **Expected:** Warning logged about testing without prompt first
- [ ] **Verify:** Check logs for prompt warning

#### Test 6: Prompt + Keyterms Conflict (Error Test)
- [ ] **Setup:** Set both `prompt` and `keyterms_prompt` at init
- [ ] **Expected:** ValueError raised with helpful error message
- [ ] **Verify:** Service fails to initialize with clear error

---

### ✅ Keyterms Prompting Tests

#### Test 7: Basic Keyterms at Init
- [ ] **Setup:** Set `keyterms_prompt=["Pipecat", "AssemblyAI", "Universal-3"]`
- [ ] **Expected:** Terms are boosted in recognition
- [ ] **Verify:** Say the boosted terms, check accuracy

#### Test 8: Empty Keyterms (No Boosting)
- [ ] **Setup:** Set `keyterms_prompt=[]`
- [ ] **Expected:** No boosting, default behavior
- [ ] **Verify:** Normal transcription

---

### ✅ Diarization Tests

#### Test 9: Diarization Disabled (Default)
- [ ] **Setup:** Don't set `speaker_labels` parameter
- [ ] **Expected:** No speaker info in transcripts
- [ ] **Verify:** TranscriptionFrame.user_id is default user_id

#### Test 10: Diarization Enabled (No Formatting)
- [ ] **Setup:** Set `speaker_labels=True`
- [ ] **Expected:** Speaker ID in user_id field, plain text
- [ ] **Verify:** Multiple speakers show different IDs (Speaker A, Speaker B)

#### Test 11: Diarization with XML Formatting
- [ ] **Setup:** Set `speaker_labels=True`, `speaker_format="<{speaker}>{text}</{speaker}>"`
- [ ] **Expected:** Text includes speaker tags: `<Speaker A>Hello</Speaker A>`
- [ ] **Verify:** Formatted text in transcript, speaker ID in user_id

#### Test 12: Diarization with Colon Prefix
- [ ] **Setup:** Set `speaker_labels=True`, `speaker_format="{speaker}: {text}"`
- [ ] **Expected:** Text includes prefix: `Speaker A: Hello`
- [ ] **Verify:** Formatted text, multiple speakers distinguishable

---

### ✅ Dynamic Updates Tests

#### Test 13: Dynamic Keyterms Update (Stage 1 → Stage 2)
- [ ] **Setup:** Start with empty keyterms, update mid-conversation
- [ ] **Expected:** New keyterms take effect immediately
- [ ] **Test Steps:**
  1. Start conversation with no keyterms
  2. Send update frame with `keyterms_prompt=["cardiology", "Dr. Smith"]`
  3. Say the new terms
- [ ] **Verify:** Improved recognition after update

#### Test 14: Clear Keyterms (Reset Context)
- [ ] **Setup:** Start with keyterms, clear them mid-stream
- [ ] **Expected:** Context biasing removed
- [ ] **Test Steps:**
  1. Start with `keyterms_prompt=["test", "words"]`
  2. Send update frame with `keyterms_prompt=[]`
- [ ] **Verify:** No more boosting after clear

#### Test 15: Dynamic Silence Parameters
- [ ] **Setup:** Update `max_turn_silence` mid-stream
- [ ] **Expected:** Turn detection timing changes
- [ ] **Test Steps:**
  1. Start with default (1200ms)
  2. Update to `max_turn_silence=5000` (for reading numbers)
  3. Pause longer between words
  4. Update back to `max_turn_silence=1200`
- [ ] **Verify:** Longer pauses tolerated when increased

#### Test 16: Dynamic Prompt Update
- [ ] **Setup:** Update prompt mid-stream
- [ ] **Expected:** New instructions take effect
- [ ] **Test Steps:**
  1. Start with default prompt
  2. Send update with custom prompt
- [ ] **Verify:** Behavior changes according to new prompt

#### Test 17: Multiple Parameters at Once
- [ ] **Setup:** Update keyterms, max_turn_silence, and min_end_of_turn together
- [ ] **Expected:** All parameters updated in single WebSocket message
- [ ] **Verify:** Check logs for single UpdateConfiguration message

#### Test 18: Dynamic Update - Prompt + Keyterms Conflict (Error)
- [ ] **Setup:** Try to update both prompt and keyterms_prompt in same update
- [ ] **Expected:** ValueError raised
- [ ] **Verify:** Update fails with clear error message

---

### ✅ Turn Detection Mode Tests

#### Test 19: Pipecat Mode (vad_force_turn_endpoint=True) - Default
- [ ] **Setup:** Use default settings (Pipecat mode)
- [ ] **Expected:**
  - ForceEndpoint sent on VAD stop
  - Smart Turn Analyzer makes decisions
  - min=max=100ms for u3-rt-pro
- [ ] **Verify:** Fast finals, Smart Turn handles completeness

#### Test 20: STT Mode (vad_force_turn_endpoint=False) - u3-rt-pro only
- [ ] **Setup:** Set `vad_force_turn_endpoint=False` with u3-rt-pro
- [ ] **Expected:**
  - AssemblyAI controls turn endings
  - SpeechStarted message triggers interruptions
  - UserStarted/StoppedSpeakingFrame emitted
- [ ] **Verify:** Turn detection from AssemblyAI model

#### Test 21: STT Mode with universal-streaming (Error Test)
- [ ] **Setup:** Set `vad_force_turn_endpoint=False` with universal-streaming
- [ ] **Expected:** ValueError raised (requires u3-rt-pro)
- [ ] **Verify:** Service fails with clear error

---

### ✅ Language Detection Tests (If Multilingual Model)

#### Test 22: Language Detection Enabled
- [ ] **Setup:** Use `universal-streaming-multilingual` with `language_detection=True`
- [ ] **Expected:** Language codes in transcripts
- [ ] **Verify:** Speak different languages, check language_code field

#### Test 23: Language Confidence Threshold
- [ ] **Setup:** Enable language detection
- [ ] **Expected:** High confidence (≥0.7) → detected language, Low → fallback to English
- [ ] **Verify:** Check logs for confidence warnings

---

### ✅ Edge Cases & Error Handling

#### Test 24: WebSocket Disconnect During Update
- [ ] **Setup:** Simulate disconnect, try update
- [ ] **Expected:** Error logged, update queued for reconnection
- [ ] **Verify:** Graceful handling, no crash

#### Test 25: Invalid Parameter Types
- [ ] **Setup:** Send update with wrong type (e.g., keyterms_prompt as string)
- [ ] **Expected:** Warning logged, parameter skipped
- [ ] **Verify:** Service continues, invalid param ignored

#### Test 26: Unknown Parameter in Update
- [ ] **Setup:** Send update with unsupported parameter (e.g., `language`)
- [ ] **Expected:** Warning logged about parameter
- [ ] **Verify:** Other valid params still updated

---

### ✅ Integration Tests

#### Test 27: Full Voice Agent Flow (Multi-Stage)
- [ ] **Setup:** Complete voice agent with stage transitions
- [ ] **Test Steps:**
  1. Greeting stage (general keyterms)
  2. Name collection stage (name keyterms)
  3. Account number stage (number keyterms, longer silence)
  4. Medical info stage (medical keyterms)
  5. Closing stage (goodbye keyterms)
- [ ] **Verify:** Each stage has appropriate keyterms and timing

#### Test 28: Diarization + Dynamic Updates
- [ ] **Setup:** Enable diarization, update keyterms mid-stream
- [ ] **Expected:** Both features work together
- [ ] **Verify:** Speaker IDs persist, keyterms update correctly

#### Test 29: Interruption Handling
- [ ] **Setup:** Bot speaking, user interrupts
- [ ] **Expected:**
  - Pipecat mode: VAD + Smart Turn handles
  - STT mode: SpeechStarted triggers interrupt
- [ ] **Verify:** Bot stops, user speech processed

---

## Testing Results Template

```
| Test # | Feature | Status | Notes |
|--------|---------|--------|-------|
| 1 | Default Config | ✅ PASS | |
| 2 | Custom min_silence | ✅ PASS | |
| 3 | max_silence Warning | ✅ PASS | |
| ... | ... | ... | ... |
```

---

## Expected Outcomes Summary

### ✅ Should Work (No Errors)
- Default configuration
- Custom min_turn_silence
- Keyterms prompting
- Diarization with/without formatting
- Dynamic updates (one parameter or multiple)
- Pipecat mode turn detection

### ⚠️ Should Warn (Logs Warning, Continues)
- Custom prompt set at init
- max_turn_silence set (overridden)
- Invalid parameter types in updates
- Language update attempted
- Prompt used with universal-streaming

### ❌ Should Error (Raises Exception, Stops)
- prompt + keyterms_prompt at init
- prompt + keyterms_prompt in same update
- vad_force_turn_endpoint=False with universal-streaming

---

## Quick Test Commands

```bash
# Run basic test
python test_assemblyai_u3pro.py --test basic

# Run specific test
python test_assemblyai_u3pro.py --test diarization

# Run all tests
python test_assemblyai_u3pro.py --test all

# Interactive mode
python test_assemblyai_u3pro.py --interactive
```
