# Onairos + Pipecat Developer Quickstart

## 5-Minute Setup

### Step 1: Get Onairos Credentials
1. Go to https://dashboard.onairos.uk
2. Create account → Create app → Copy credentials

### Step 2: Frontend (React)
```bash
npm install @onairos/sdk
```

```jsx
import Onairos from 'onairos';

function ConnectButton({ onCredentials }) {
  const handleConnect = () => {
    const onairos = new Onairos({
      requestData: {
        Traits: { type: "Personality", size: "Large" }
      },
      onComplete: (data) => {
        // Send to your Pipecat backend
        websocket.send(JSON.stringify({
          type: "onairos_credentials",
          apiUrl: data.apiUrl,
          accessToken: data.accessToken
        }));
      }
    });
  };

  return <button onClick={handleConnect}>Connect Onairos</button>;
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
