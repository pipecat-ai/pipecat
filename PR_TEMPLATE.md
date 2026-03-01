# Add Onairos Personalization Integration

## Summary

Adds Onairos integration for deep personalization of voice agents using user personality traits, memories, and MBTI compatibility scores.

**Key Features:**
- `OnairosPersonaInjector` - Augments LLM prompts with user context
- `OnairosMemoryService` - Enhanced context management
- `OnairosContextAggregator` - Connection state handling
- Full documentation and working example

## Why?

Onairos provides the best onboarding and personalization for Pipecat users:
- Users feel truly remembered across sessions
- Agents adapt communication style based on preferences
- Faster onboarding with pre-known user context
- Privacy-first: user owns their data

## How It Works

```
Frontend onComplete → {apiUrl, accessToken}
                            ↓
Backend set_api_credentials(apiUrl, accessToken)
                            ↓
Backend calls Onairos API → Gets traits/memory/MBTI
                            ↓
LLM prompt augmented:
  [Base Prompt]
  Personality Traits: {"AI Interest": 80, "Coffee Lover": 95}
  Memory: Reads Daily Stoic every morning...
  MBTI: INFJ: 0.627, INTJ: 0.585
```

## How to Test

1. **Install dependencies:**
   ```bash
   pip install "pipecat-ai[onairos,daily,openai,elevenlabs,silero]"
   ```

2. **Set environment variables:**
   ```bash
   export ONAIROS_API_KEY=sk_test_...  # Get from dashboard.onairos.uk
   export DAILY_API_KEY=...
   export OPENAI_API_KEY=...
   export ELEVENLABS_API_KEY=...
   export DEEPGRAM_API_KEY=...
   ```

3. **Run example:**
   ```bash
   python examples/foundational/38-onairos.py
   ```

4. **What to observe:**
   - Agent receives augmented prompt with personality traits
   - Logs show "Loaded Onairos data: X traits, Y MBTI types"
   - Responses are personalized based on user context

5. **Run tests:**
   ```bash
   uv run pytest tests/test_onairos.py -v
   ```

## Files Changed

### New Files
- `src/pipecat/services/onairos/__init__.py` - Module exports
- `src/pipecat/services/onairos/persona.py` - Main augmentation service
- `src/pipecat/services/onairos/memory.py` - Memory service
- `src/pipecat/services/onairos/context.py` - Context aggregator
- `docs/ONAIROS_INTEGRATION.md` - Full integration guide
- `docs/DEVELOPER_QUICKSTART.md` - 5-minute setup
- `examples/foundational/38-onairos.py` - Working example
- `tests/test_onairos.py` - Unit tests
- `changelog/onairos.added.md` - Changelog entry

### Modified Files
- `README.md` - Updated with Onairos branding
- `pyproject.toml` - Added `onairos` optional dependency
- `env.example` - Added `ONAIROS_*` variables

## Checklist

- [x] Code follows Pipecat patterns (FrameProcessor, Pydantic models)
- [x] Google-style docstrings
- [x] Unit tests added
- [x] Documentation added
- [x] Example added
- [x] Changelog entry added
- [x] No linter errors
- [ ] Tests pass locally

## Future Plans

- Webhook handler for server-to-server flow
- Caching layer for reduced API calls
- More examples (different use cases)
- React component for Voice UI Kit integration

## Resources

- **Onairos Docs:** https://onairos.uk/docs
- **Onairos Dashboard:** https://dashboard.onairos.uk
