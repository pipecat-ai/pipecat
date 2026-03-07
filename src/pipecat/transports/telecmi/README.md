# TeleCMI Transport for Pipecat

The `TelecmiTransport` provides an integration with [TeleCMI](https://telecmi.com/), allowing you to easily build scalable voice agents that can handle incoming phone calls and dial out to customers. It is built on a robust WebRTC stack for PSTN calls, ensuring ultra-low latency and production-readiness from day one.

## Installation

Make sure to install the TeleCMI extras when setting up Pipecat:
```bash
pip install "pipecat-ai[telecmi]"
```

## Getting Started

To use the TeleCMI transport with Pipecat, follow these steps to set up your agent and phone numbers using the Piopiy developer platform.

### 1. Setup Your Piopiy Agent
To get started, you will need a phone number and an AI Agent configured on the platform. 

Please follow the official [Piopiy Documentation](https://doc.piopiy.com/piopiy/docs/getting-started/introduction) to:
1. Buy a phone number.
2. Create and configure your Voice AI Agent.
3. Map your agent to the purchased phone number.
4. Retrieve your `AGENT_ID` and `AGENT_TOKEN` credentials for the agent.

### 2. Set Environment Variables
To run the agent locally, you must provide your agent credentials and standard API keys (for foundational examples). Create a `.env` file with the following variables:

```env
AGENT_ID=your_agent_id
AGENT_TOKEN=your_agent_token

# Depending on the AI services in your script (e.g. OpenAI, Cartesia, Deepgram)
OPENAI_API_KEY=your_openai_api_key
CARTESIA_API_KEY=your_cartesia_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
```

### 3. Run Your Pipecat Script

For an end-to-end implementation of an AI agent using TeleCMI, check out our foundational example:
```bash
python examples/foundational/57-transports-telecmi.py
```

### 4. Dial the Number
Once your script is running, dial your purchased TeleCMI phone number from a real phone. TeleCMI will hit your agent configuration and spawn a session, connecting the caller directly to your Pipecat agent.
