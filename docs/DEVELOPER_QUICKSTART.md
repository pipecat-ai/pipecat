# Onairos + Pipecat Developer Quickstart

## 5-Minute Setup

### Step 1: Get Onairos Credentials
1. Go to https://dashboard.onairos.uk
2. Create account → Create app → Copy credentials

### Step 2: Frontend

**Install the SDK for your platform:**
```bash
# Web (React, Vue, vanilla JS)
npm install onairos

# React Native
npm install @onairos/react-native

# Swift and Flutter SDKs also available - see docs.onairos.uk
```

```jsx
import Onairos from 'onairos';

function ConnectButton({ websocket }) {
  return (
    <Onairos
      requestData={{
        Traits: { type: "Personality", size: "Large" }
      }}
      onComplete={(apiUrl, accessToken) => {
        // Send credentials to your Pipecat backend
        websocket.send(JSON.stringify({
          type: "onairos_credentials",
          apiUrl: apiUrl,
          accessToken: accessToken
        }));
      }}
    />
  );
}
```

### Step 3: Backend (Python/Pipecat)
```bash
pip install "pipecat-ai[onairos]"
```

```python
from pipecat.services.onairos import OnairosPersonaInjector

# Initialize
persona = OnairosPersonaInjector(user_id="user_123")

# When credentials arrive from frontend:
persona.set_api_credentials(api_url, access_token)

# Add to pipeline - it auto-fetches and augments LLM prompt
pipeline = Pipeline([...stt, persona, llm, tts...])
```

### That's It!
Your LLM now receives augmented prompts with user personality traits, memories, and MBTI preferences.

---

## Full Example

See `examples/foundational/38-onairos.py`
