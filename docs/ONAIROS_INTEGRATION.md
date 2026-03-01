# Onairos Integration Guide

Complete guide to integrating Onairos personalization with Pipecat voice agents.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Backend Setup (Python/Pipecat)](#backend-setup-pythonpipecat)
4. [Frontend Setup (npm)](#frontend-setup-npm)
5. [API Reference](#api-reference)
6. [How Persona Data Augments Your Agent](#how-persona-data-augments-your-agent)
7. [Examples](#examples)

---

## Overview

Onairos is a privacy-first personalization platform that enables your voice agents to:

- **Remember users** across sessions with persistent persona data
- **Personalize responses** based on communication style, interests, and preferences
- **Seamless onboarding** with user-controlled data sharing
- **Respect privacy** with user-owned data architecture

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Your Application                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐  │
│  │   Frontend  │     │    Pipecat      │     │    Onairos API   │  │
│  │  (React/RN) │────▶│  Voice Agent    │────▶│ api.onairos.uk   │  │
│  │             │     │                 │     │                  │  │
│  │ @onairos/sdk│     │ OnairosMemory   │     │ GET /personas    │  │
│  │             │     │ OnairosPersona  │     │ POST /connections│  │
│  └─────────────┘     └─────────────────┘     └──────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### 1. Onairos Developer Account

1. Go to [https://dashboard.onairos.uk/](https://dashboard.onairos.uk/)
2. Create a developer account
3. Create a new application
4. Get your API credentials:
   - **Secret Key** (`sk_live_*` or `sk_test_*`) - for server-side
   - **Publishable Key** (`pk_*`) - for client-side
   - **App ID** - your application identifier

### 2. Environment Variables

```bash
# Add to your .env file
ONAIROS_API_KEY=sk_live_your_secret_key
ONAIROS_APP_ID=app_your_app_id
ONAIROS_PUBLISHABLE_KEY=pk_your_publishable_key
```

---

## Backend Setup (Python/Pipecat)

The Pipecat backend calls the Onairos REST API directly using `aiohttp`. No npm/JavaScript needed on the backend.

### Installation

```bash
# Install pipecat with Onairos support
pip install "pipecat-ai[onairos]"

# Or with uv
uv add "pipecat-ai[onairos]"

# This installs aiohttp for REST API calls to api.onairos.uk
```

### Basic Integration

```python
import os
from pipecat.services.onairos import (
    OnairosMemoryService,
    OnairosPersonaInjector,
    OnairosContextAggregator,
)

# Initialize services
memory = OnairosMemoryService(
    api_key=os.getenv("ONAIROS_API_KEY"),
    app_id=os.getenv("ONAIROS_APP_ID"),
    user_id="user_123"
)

persona = OnairosPersonaInjector(
    api_key=os.getenv("ONAIROS_API_KEY"),
    app_id=os.getenv("ONAIROS_APP_ID"),
    user_id="user_123"
)
```

### Pipeline Integration

```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.onairos import OnairosMemoryService, OnairosPersonaInjector

pipeline = Pipeline([
    transport.input(),           # Audio/video input
    stt,                         # Speech-to-text
    persona,                     # ← Inject user persona early
    user_aggregator,
    memory,                      # ← Enhance with persona data
    llm,                         # Your LLM (OpenAI, Anthropic, etc.)
    tts,                         # Text-to-speech
    transport.output(),
    assistant_aggregator,
])
```

---

## Frontend Setup (npm) - Optional

> **Note:** This section is for your **separate frontend application** (React, React Native, etc.), not this Pipecat repo. The frontend handles user authentication with Onairos and passes the data to your Pipecat backend.

For web and mobile applications, use the Onairos SDK to enable user connections.

### Web (React)

```bash
npm install @onairos/sdk @onairos/react
```

```jsx
import { OnairosProvider, useOnairos, usePersona } from '@onairos/react';

function App() {
  return (
    <OnairosProvider 
      apiKey={process.env.ONAIROS_PUBLISHABLE_KEY}
      appId={process.env.ONAIROS_APP_ID}
    >
      <VoiceAgent />
    </OnairosProvider>
  );
}

function VoiceAgent() {
  const { connect, isConnected } = useOnairos();
  const { persona, isLoading } = usePersona();

  if (!isConnected) {
    return (
      <button onClick={() => connect({ 
        permissions: ['preferences', 'interests'] 
      })}>
        Connect Onairos for Personalization
      </button>
    );
  }

  return (
    <div>
      <p>Welcome! Your interests: {persona?.traits.interests.join(', ')}</p>
      {/* Your Pipecat voice UI here */}
    </div>
  );
}
```

### React Native

```bash
npm install @onairos/react-native
```

```jsx
import { OnairosProvider, useOnairos } from '@onairos/react-native';

function App() {
  return (
    <OnairosProvider apiKey={ONAIROS_PUBLISHABLE_KEY}>
      <VoiceAgent />
    </OnairosProvider>
  );
}
```

### Vanilla JavaScript

```bash
npm install @onairos/sdk
```

```javascript
import { OnairosClient } from '@onairos/sdk';

const onairos = new OnairosClient({
  apiKey: process.env.ONAIROS_API_KEY,
  appId: process.env.ONAIROS_APP_ID,
  environment: 'production'
});

// Create connection URL for user to connect
const connectionUrl = await onairos.connections.create({
  userId: 'user_123',
  redirectUrl: 'https://yourapp.com/callback',
  permissions: ['preferences', 'interests', 'traits']
});

// After connection, fetch persona
const persona = await onairos.personas.get('user_123');
console.log(persona.preferences.communicationStyle);
```

---

## API Reference

### Onairos REST API

**Base URL:** `https://api.onairos.uk/v1`

#### Get Persona

```
GET /personas/:userId
Authorization: Bearer sk_live_your_key

Response:
{
  "id": "persona_abc123",
  "userId": "user_123",
  "preferences": {
    "contentTopics": ["technology", "music"],
    "communicationStyle": "casual",
    "timezone": "America/New_York"
  },
  "traits": {
    "openness": 0.8,
    "interests": ["AI", "privacy", "music"]
  },
  "connectedAt": "2025-01-10T12:00:00Z"
}
```

#### Create Connection

```
POST /connections
Authorization: Bearer sk_live_your_key
Content-Type: application/json

{
  "userId": "user_123",
  "redirectUrl": "https://yourapp.com/callback",
  "permissions": ["preferences", "interests", "traits"]
}

Response:
{
  "connectionId": "conn_xyz789",
  "connectionUrl": "https://connect.onairos.uk/c/conn_xyz789",
  "expiresAt": "2025-01-15T12:00:00Z"
}
```

---

## How Onairos Augments Your Agent Prompts

### The Augmentation Pattern

Onairos augments your **base prompt** with rich user context from the `onComplete` callback:

**Base Prompt (without Onairos):**
```
You're onboarding a user for dating matches. Have a genuine conversation
to understand them. Ask whatever feels relevant based on what they share.
Trust your judgment on what matters.
Once you genuinely feel you know enough to match them well, wrap up.

Critical Instruction:
Always check context before asking. Complete onboarding ASAP.
```

**Augmented Prompt (with Onairos):**
```
You're onboarding a user for dating matches. Have a genuine conversation
to understand them. Ask whatever feels relevant based on what they share.
Trust your judgment on what matters.
Once you genuinely feel you know enough to match them well, wrap up.

Personality Traits of User:
{"Stoic Wisdom Interest": 80, "Daily Stoic Engagement": 90,
 "Philosophical Discussions": 85, "AI and ML Enthusiasm": 40,
 "Coffee Lover": 95, "Morning Person": 70}

Memory of User:
Reads Daily Stoic every morning. Prefers small coffee shop meetups.
Works in tech, interested in philosophy and AI ethics. Has a golden retriever.

MBTI (Personalities User Likes):
INFJ: 0.627, INTJ: 0.585, ENFJ: 0.580, INFP: 0.550, ENTP: 0.520

Critical Instruction:
Always check context before asking. Complete onboarding ASAP.
```

### How Onairos Data Flows

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   FRONTEND (React/JS)                    BACKEND (Pipecat/Python)        │
│                                                                          │
│   1. User clicks                                                         │
│      "Connect Onairos"                                                   │
│            │                                                             │
│            ▼                                                             │
│   2. Onairos popup                                                       │
│      User authorizes                                                     │
│            │                                                             │
│            ▼                                                             │
│   3. onComplete returns:        ──────▶  4. Backend receives             │
│      { apiUrl, accessToken }             { apiUrl, accessToken }         │
│      (NOT the actual data!)              via WebSocket/RTVI              │
│                                                │                         │
│                                                ▼                         │
│                                          5. Backend calls                │
│                                             POST apiUrl                  │
│                                             with accessToken             │
│                                                │                         │
│                                                ▼                         │
│                                          6. Onairos returns:             │
│                                             { traits, memory, mbti }     │
│                                                │                         │
│                                                ▼                         │
│                                          7. Augment LLM prompt           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key Insight:** The frontend only receives `apiUrl` and `accessToken`. The actual user data (traits, memory, MBTI) stays secure on the backend.

### The onComplete Response

When `onComplete` fires in the frontend:

```javascript
// Frontend receives this (NOT the actual data):
{
  "apiUrl": "https://api2.onairos.uk/inferenceNoProof",
  "accessToken": "eyJhbGciOiJIUzI1NiIs..."  // JWT token
}
```

### The Backend API Call

The backend uses these credentials to fetch the actual data:

```
POST https://api2.onairos.uk/inferenceNoProof
Authorization: Bearer {accessToken}
Content-Type: application/json

{
  "accessToken": "{accessToken}",
  "inputData": []
}

Response:
{
  "InferenceResult": {
    "output": {
      "personality_traits": {
        "Stoic Wisdom Interest": 80,
        "AI Enthusiasm": 40
      },
      "memory": "Reads Daily Stoic every morning...",
      "mbti": {
        "INFJ": 0.627,
        "INTJ": 0.585
      }
    }
  }
}
```

### Using OnairosPersonaInjector

```python
from pipecat.services.onairos import OnairosPersonaInjector, OnairosUserData

# Initialize without credentials (will be set when frontend sends them)
persona = OnairosPersonaInjector(
    user_id="user_123",
    params=OnairosPersonaInjector.InputParams(
        include_personality_traits=True,
        include_memory=True,
        include_mbti=True,
        top_mbti_count=5,
        critical_instruction="Always check context before asking.",
    ),
)

# When frontend sends onComplete data via WebSocket/RTVI:
@transport.event_handler("on_client_message")
async def on_client_message(transport, message):
    if message.get("type") == "onairos_credentials":
        persona.set_api_credentials(
            api_url=message["apiUrl"],
            access_token=message["accessToken"]
        )

# The persona injector will automatically call the Onairos API
# and augment the LLM prompt when processing frames
```

### Alternative: Pre-loaded Data

If you already have the data (e.g., from a webhook):

```python
persona = OnairosPersonaInjector(
    user_id="user_123",
    user_data=OnairosUserData(
        personality_traits={"Stoic Wisdom Interest": 80, "AI Enthusiasm": 40},
        memory="Reads Daily Stoic every morning. Prefers coffee shop meetups.",
        mbti={"INFJ": 0.627, "INTJ": 0.585, "ENFJ": 0.580},
    ),
)
```

### Real Impact Examples

**Without Onairos (generic onboarding):**
> "Hi! Tell me about yourself. What are your hobbies?"

**With Onairos (knows user loves Stoic philosophy + coffee):**
> "Hey! I see you're really into Stoic philosophy - that's fascinating. 
> Do you have a favorite Stoic author, or is it more the daily practice that resonates with you?
> Also, I noticed you're a coffee person - me too! Any favorite spots?"

The agent **already knows** key information, so it can:
- Skip basic questions (saves time)
- Reference interests naturally (builds rapport)
- Adapt communication style (feels personal)

---

## Examples

### Full Voice Agent Example

See `examples/foundational/38-onairos.py` for a complete working example.

```python
from pipecat.services.onairos import OnairosMemoryService, OnairosPersonaInjector

# Services
memory = OnairosMemoryService(
    api_key=os.getenv("ONAIROS_API_KEY"),
    user_id=USER_ID,
    params=OnairosMemoryService.InputParams(
        system_prompt="User Profile:\n",
        include_preferences=True,
        include_traits=True,
    )
)

persona = OnairosPersonaInjector(
    api_key=os.getenv("ONAIROS_API_KEY"),
    user_id=USER_ID,
)

# Event handlers
@persona.event_handler("on_persona_loaded")
async def on_persona_loaded(persona_data):
    logger.info(f"Loaded persona with interests: {persona_data.get('traits', {}).get('interests')}")

@persona.event_handler("on_connection_required")
async def on_connection_required(user_id):
    logger.info(f"User {user_id} needs to connect Onairos")
```

### With Voice UI Kit + RTVI

The recommended pattern is to pass Onairos data from frontend to backend via RTVI config:

**Frontend (React):**
```jsx
import { RTVIClient } from '@pipecat-ai/client-js';
import { useOnairos } from '@onairos/react';

function VoiceAgent() {
  const { persona, isConnected } = useOnairos();
  
  const startSession = async () => {
    const client = new RTVIClient({
      transport: new DailyTransport(),
      params: {
        baseUrl: 'https://your-server.com',
        // Pass Onairos data in config
        config: [
          {
            service: 'onairos',
            options: [
              { name: 'user_id', value: 'user_123' },
              { name: 'personality_traits', value: persona?.personality_traits || {} },
              { name: 'memory', value: persona?.memory || '' },
              { name: 'mbti', value: persona?.mbti || {} },
            ]
          }
        ]
      }
    });
    
    await client.connect();
  };

  return (
    <button onClick={startSession}>
      {isConnected ? 'Start Personalized Call' : 'Connect Onairos First'}
    </button>
  );
}
```

**Backend (Python) - Receiving Onairos Data:**
```python
from pipecat.services.onairos import OnairosPersonaInjector, OnairosUserData

async def run_bot(transport, runner_args):
    # Get Onairos data from RTVI config (passed by frontend)
    onairos_config = runner_args.config.get('onairos', {})
    
    user_id = onairos_config.get('user_id', 'anonymous')
    
    # Create persona injector with frontend data
    persona = OnairosPersonaInjector(
        user_id=user_id,
        user_data=OnairosUserData(
            personality_traits=onairos_config.get('personality_traits', {}),
            memory=onairos_config.get('memory', ''),
            mbti=onairos_config.get('mbti', {}),
        ) if onairos_config.get('personality_traits') else None,
        # Fallback to API if no frontend data
        api_key=os.getenv("ONAIROS_API_KEY"),
    )
    
    # ... rest of pipeline
```

### Alternative: Onairos onComplete Webhook

For server-to-server flow, Onairos can send data via webhook when a user connects:

```python
from fastapi import FastAPI, Request
from pipecat.services.onairos import OnairosUserData

app = FastAPI()

# Store user data (use Redis/DB in production)
user_data_cache = {}

@app.post("/onairos/webhook")
async def onairos_webhook(request: Request):
    """Receive Onairos onComplete data via webhook."""
    data = await request.json()
    
    user_id = data.get('user_id')
    user_data_cache[user_id] = OnairosUserData(
        personality_traits=data.get('personality_traits', {}),
        memory=data.get('memory', ''),
        mbti=data.get('mbti', {}),
        raw_data=data,
    )
    
    return {"status": "ok"}
```

---

## Troubleshooting

### "ONAIROS_API_KEY not set"

Set your API key in environment variables:
```bash
export ONAIROS_API_KEY=sk_live_your_key
```

### "No persona found for user"

The user hasn't connected their Onairos account yet. Use `OnairosContextAggregator` to prompt them to connect.

### Rate Limiting

| Plan       | Requests/min | Requests/day |
|------------|-------------|--------------|
| Free       | 60          | 1,000        |
| Startup    | 300         | 50,000       |
| Enterprise | Unlimited   | Unlimited    |

---

## Resources

- **Onairos Docs:** https://onairos.uk/docs
- **Onairos Dashboard:** https://dashboard.onairos.uk
- **API Reference:** https://onairos.uk/docs/api-endpoints/
- **JavaScript SDK:** https://onairos.uk/docs/javascript-sdk/
