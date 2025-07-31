# Phone Bot Twilio

Learn how to connect your Pipecat bot to a phone number so users can call and have voice conversations. This example shows the complete setup for telephone-based AI interactions using Twilio's telephony services. At the end, you'll be able to talk to your bot on the phone.

## Prerequisites

- Python 3.10+
- [ngrok](https://ngrok.com/docs/getting-started/) (for tunneling)
- [Twilio Account](https://www.twilio.com/login) and [phone number](https://help.twilio.com/articles/223135247-How-to-Search-for-and-Buy-a-Twilio-Phone-Number-from-Console)
- AI Service API keys for: [Deepgram](https://console.deepgram.com/signup), [OpenAI](https://auth.openai.com/create-account), and [Cartesia](https://play.cartesia.ai/sign-up)

## Setup

This example requires running both a server and ngrok tunnel in **two separate terminal windows**.

### Terminal 1: Start ngrok and Configure Twilio

1. Start ngrok:
   In a new terminal, start ngrok to tunnel the local server:

   ```bash
   ngrok http 7860
   ```

   > Want a fixed ngrok URL? Use the `--subdomain` flag:
   > `ngrok http --subdomain=your_ngrok_name 7860`

2. Update the Twilio Webhook:

   - Go to your Twilio phone number's configuration page
   - Under "Voice Configuration", in the "A call comes in" section:
     - Select "Webhook" from the dropdown
     - Enter your ngrok URL: `https://your-ngrok-url.ngrok.io`
     - Ensure "HTTP POST" is selected
   - Click Save at the bottom of the page

3. Configure streams.xml:
   - Copy the template file to create your local version:
     ```bash
     cp templates/streams.xml.template templates/streams.xml
     ```
   - In `templates/streams.xml`, replace `<your_server_url>` with your ngrok URL (without `https://`)
   - The final URL should look like: `wss://abc123.ngrok.io/ws`

### Terminal 2: Server Setup

1. Set up a virtual environment

   From the `examples/phone-bot-twilio` directory, run:

   ```bash
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

   > Optional: Add your `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` to enable auto-hangup.

4. Run the Application

   ```bash
   python server.py
   ```

### Test Your Phone Bot

**Call your Twilio phone number** to start talking with your AI bot! 🚀

> 💡 **Tip**: Check your server terminal for debug logs showing Pipecat's internal workings.

## Troubleshooting

- **Call doesn't connect**: Verify your ngrok URL is correctly set in both Twilio webhook and `streams.xml`
- **No audio or bot doesn't respond**: Check that all API keys are correctly set in your `.env` file
- **Webhook errors**: Ensure your server is running and ngrok tunnel is active before making calls
- **ngrok tunnel issues**: Free ngrok URLs change each restart - remember to update both Twilio and `streams.xml`

## Understanding the Call Flow

1. **Incoming Call**: User dials your Twilio number
2. **Webhook**: Twilio sends call data to your ngrok URL
3. **WebSocket**: Your server establishes real-time audio connection via Websocket and exchanges Media Streams with Twilio
4. **Processing**: Audio flows through your Pipecat Pipeline
5. **Response**: Synthesized speech streams back to caller

## Next Steps

- **Deploy to production**: Replace ngrok with a proper server deployment
- **Explore other telephony providers**: Try [Telnyx](https://github.com/pipecat-ai/pipecat-examples/tree/main/telnyx-chatbot) or [Plivo](https://github.com/pipecat-ai/pipecat-examples/tree/main/plivo-chatbot) examples
- **Advanced telephony features**: Check out [pipecat-examples](https://github.com/pipecat-ai/pipecat-examples) for call recording, transfer, and more
- **Join Discord**: Connect with other developers on [Discord](https://discord.gg/pipecat)
