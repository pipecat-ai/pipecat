# API Key Validation Implementation Guide

This guide explains how to implement API key validation for Pipecat services to address issue #941.

## Overview

The API key validation feature raises exceptions early when API keys are missing or blank, improving developer experience by catching configuration errors during service initialization rather than at runtime.

## Implementation

### 1. Utility Function

The validation logic is centralized in `src/pipecat/utils/api_key_validator.py`:

```python
from pipecat.utils.api_key_validator import validate_api_key, APIKeyError
```

### 2. Function Signature

```python
validate_api_key(
    api_key: Optional[str],
    service_name: str,
    allow_none: bool = False,
    env_var_name: Optional[str] = None,
) -> None
```

**Parameters:**
- `api_key`: The API key to validate
- `service_name`: Name of the service (used in error messages)
- `allow_none`: If True, allows None values (for services that support environment variables)
- `env_var_name`: Optional environment variable name to suggest in error message

### 3. Adding Validation to Services

#### Step 1: Import the validator

Add to the service file imports:

```python
from pipecat.utils.api_key_validator import validate_api_key
```

#### Step 2: Add validation in `__init__`

Add the validation call right after `super().__init__()`:

**For services that REQUIRE an API key explicitly:**

```python
def __init__(self, *, api_key: str, **kwargs):
    super().__init__(**kwargs)

    # Validate API key (required)
    validate_api_key(api_key, "ServiceName", allow_none=False, env_var_name="SERVICE_API_KEY")

    # Rest of initialization...
```

**For services that allow environment variable fallback:**

```python
def __init__(self, *, api_key=None, **kwargs):
    super().__init__(**kwargs)

    # Validate API key (allow None since SDK uses env var)
    validate_api_key(api_key, "ServiceName", allow_none=True, env_var_name="SERVICE_API_KEY")

    # Rest of initialization...
```

## Examples

### Example 1: OpenAI (allows None for env var)

**File:** `src/pipecat/services/openai/base_llm.py`

```python
from pipecat.utils.api_key_validator import validate_api_key

class BaseOpenAILLMService(LLMService):
    def __init__(self, *, model: str, api_key=None, **kwargs):
        super().__init__(**kwargs)

        # Allow None since OpenAI SDK uses OPENAI_API_KEY env var
        validate_api_key(api_key, "OpenAI", allow_none=True, env_var_name="OPENAI_API_KEY")

        # ... rest of initialization
```

### Example 2: Anthropic (requires explicit API key)

**File:** `src/pipecat/services/anthropic/llm.py`

```python
from pipecat.utils.api_key_validator import validate_api_key

class AnthropicLLMService(LLMService):
    def __init__(self, *, api_key: str, **kwargs):
        super().__init__(**kwargs)

        # Required for Anthropic
        validate_api_key(api_key, "Anthropic", allow_none=False, env_var_name="ANTHROPIC_API_KEY")

        # ... rest of initialization
```

### Example 3: Deepgram STT (requires explicit API key)

**File:** `src/pipecat/services/deepgram/stt.py`

```python
from pipecat.utils.api_key_validator import validate_api_key

class DeepgramSTTService(STTService):
    def __init__(self, *, api_key: str, **kwargs):
        super().__init__(**kwargs)

        # Required for Deepgram
        validate_api_key(api_key, "Deepgram", allow_none=False, env_var_name="DEEPGRAM_API_KEY")

        # ... rest of initialization
```

## Error Messages

### When API key is None (and not allowed):

```
APIKeyError: API key for Anthropic is missing or empty. Set the ANTHROPIC_API_KEY
environment variable or pass it explicitly. Please provide a valid API key to use this service.
```

### When API key is blank/whitespace:

```
APIKeyError: API key for OpenAI is missing or empty. Set the OPENAI_API_KEY
environment variable or pass it explicitly. Please provide a valid API key to use this service.
```

## Testing

### Unit Tests

Test the validation function directly:

```python
from pipecat.utils.api_key_validator import APIKeyError, validate_api_key

def test_valid_api_key():
    validate_api_key("sk-test-key", "TestService")  # Should pass

def test_none_api_key():
    with pytest.raises(APIKeyError):
        validate_api_key(None, "TestService", allow_none=False)

def test_empty_api_key():
    with pytest.raises(APIKeyError):
        validate_api_key("", "TestService")
```

### Integration Tests

Test services with invalid keys:

```python
from pipecat.services.openai import OpenAILLMService

def test_openai_with_blank_key():
    with pytest.raises(APIKeyError):
        OpenAILLMService(api_key="", model="gpt-4.1")
```

## Services Already Updated

- ✅ **OpenAI** (LLM) - `src/pipecat/services/openai/base_llm.py`
- ✅ **Anthropic** (LLM) - `src/pipecat/services/anthropic/llm.py`
- ✅ **Deepgram** (STT) - `src/pipecat/services/deepgram/stt.py`

## Services To Update

The validation should be added to all services that require API keys:

### LLM Services
- [ ] AWS (Bedrock)
- [ ] Azure
- [ ] Cerebras
- [ ] DeepSeek
- [ ] Fireworks
- [ ] Gemini
- [ ] Grok
- [ ] Groq
- [ ] Mistral
- [ ] Perplexity
- [ ] Together AI
- [ ] etc.

### TTS Services
- [ ] ElevenLabs
- [ ] Cartesia
- [ ] Fish
- [ ] LMNT
- [ ] PlayHT
- [ ] Rime
- [ ] etc.

### STT Services
- [ ] AssemblyAI
- [ ] Gladia
- [ ] Soniox
- [ ] etc.

### Other Services
- [ ] HeyGen
- [ ] Tavus
- [ ] Simli
- [ ] etc.

## Rollout Strategy

1. **Phase 1** (Completed): Core implementation
   - Create utility function
   - Update 3 example services (OpenAI, Anthropic, Deepgram)
   - Add comprehensive tests

2. **Phase 2**: Update high-traffic services
   - Update most commonly used LLM services
   - Update popular TTS/STT services

3. **Phase 3**: Complete rollout
   - Update all remaining services
   - Add validation to new services going forward

## Benefits

1. **Better Developer Experience**: Errors caught immediately during initialization
2. **Clearer Error Messages**: Helpful messages with environment variable suggestions
3. **Consistent Behavior**: All services validate API keys the same way
4. **Easier Debugging**: No need to wait for API calls to fail
5. **Documentation**: Error messages guide users to fix configuration

## Notes

- Services that use environment variables should set `allow_none=True`
- Services that require explicit API keys should set `allow_none=False`
- Always provide the `env_var_name` parameter for better error messages
- The validator checks for both `None` and empty/whitespace-only strings
