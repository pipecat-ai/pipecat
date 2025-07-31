# Client Server Web Example

Learn how to build web applications using Pipecat's client/server architecture. This approach separates your bot logic from your user interface, giving you full control over the client experience while maintaining real-time voice communication.

This example demonstrates:

- Server-side bot running with Pipecat
- React client using [Pipecat's client SDK](https://docs.pipecat.ai/client/introduction)
- Real-time voice communication between client and server
- UI components from [voice-ui-kit](https://github.com/pipecat-ai/voice-ui-kit) for common voice interface patterns

This is the recommended architecture for web applications that need custom interfaces or client-side functionality.

## Prerequisites

- Python 3.10+
- `npm` installed
- AI Service API keys for: [Deepgram](https://console.deepgram.com/signup), [OpenAI](https://auth.openai.com/create-account), and [Cartesia](https://play.cartesia.ai/sign-up)

## Setup

This example requires running both a server and client in **two separate terminal windows**.

### Terminal 1: Server Setup

1. Set up a virtual environment

From the `examples/client-server-web` directory, run:

```bash
cd server
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

> Using `uv`? Create your venv using: `uv venv && source .venv/bin/activate`.

2. Install dependencies

```bash
pip install -r requirements.txt
```

> Using `uv`? Install requirements using: `uv pip install -r requirements.txt`.

3. Configure environment variables

Create a `.env` file:

```bash
cp env.example .env
```

Then, add your API keys:

```
DEEPGRAM_API_KEY=your_deepgram_api_key
OPENAI_API_KEY=your_openai_api_key
CARTESIA_API_KEY=your_cartesia_api_key
```

4. Run the example

Run your bot using:

```bash
python bot.py
```

> Using `uv`? Run your bot using: `uv run bot.py`.

> 💡 First run note: The initial startup may take ~10 seconds as Pipecat downloads required models, like the Silero VAD model.

### Terminal 2: Client Setup

1. Open a new terminal window and navigate to the `client` folder:

From the `examples/client-server-web` directory, run:

```bash
cd client
```

2. Install dependencies:

```bash
npm i
```

3. Run the client:

```bash
npm run dev
```

4. **Open http://localhost:5173 in your browser** and click `Connect` to start talking to your bot.

> 💡 **Tip**: Check your server terminal for debug logs showing Pipecat's internal workings.

## Troubleshooting

- **Browser permissions**: Make sure to allow microphone access when prompted by your browser.
- **Connection issues**: If the WebRTC connection fails, first try a different browser. If that fails, make sure you don't have a VPN or firewall rules blocking traffic. WebRTC uses UDP to communicate.
- **Audio issues**: Check that your microphone and speakers are working and not muted.

## Next Steps

- **Explore the client SDK**: Learn more about [Pipecat's client SDKs](https://docs.pipecat.ai/client/introduction) for web, mobile, and other platforms
- **Learn about the voice-ui-kit**: Explore [voice-ui-kit](https://github.com/pipecat-ai/voice-ui-kit) to simplify your front end development
- **Advanced examples**: Check out [pipecat-examples](https://github.com/pipecat-ai/pipecat-examples) for more complex client/server applications
- **Join Discord**: Connect with other developers on [Discord](https://discord.gg/pipecat)
