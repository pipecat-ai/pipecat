# Onairos + Pipecat Developer Quickstart

## 5-Minute Setup

### Step 1: Get Onairos Credentials
1. Go to https://dashboard.onairos.uk
2. Create account → Create app → Copy your publishable key

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
Your LLM now receives augmented prompts with user personality traits, archetype, user summary, and MBTI alignment scores.

### What the LLM Sees

Once Onairos data is loaded, the LLM receives an additional system message:

```
Positive Traits of User:
Stoic Wisdom Interest: 80, AI Enthusiasm: 75, Coffee Lover: 95

Areas to Improve:
Social Media Engagement: 35, Public Speaking Confidence: 40

User Summary:
You are drawn to deep philosophical thinking and have a strong interest
in Stoic philosophy...

Archetype: The Strategic Explorer

MBTI Alignment (Personalities User Likes):
INFJ: 0.627, INTJ: 0.585, ENFJ: 0.580, ISFJ: 0.580, INFP: 0.511

Critical Instruction:
Always check context before asking. Use this information to personalize.
```

---

## Full Example

See `examples/foundational/38-onairos.py`
