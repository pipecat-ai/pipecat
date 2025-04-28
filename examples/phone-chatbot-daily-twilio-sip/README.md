# Daily + Twilio SIP Voice Bot

This project demonstrates how to create a voice bot that can receive phone calls via Twilio and use Daily's SIP capabilities to enable voice conversations.

## How It Works

1. Twilio receives an incoming call to your phone number
2. Twilio calls your webhook server (`/call` endpoint)
3. The server creates a Daily room with SIP capabilities
4. The server starts the bot process with the room details
5. The caller is put on hold with music
6. The bot joins the Daily room and signals readiness
7. Twilio forwards the call to Daily's SIP endpoint
8. The caller and bot are connected, and the bot handles the conversation

## Prerequisites

- A Daily account with an API key
- A Twilio account with a phone number that supports voice
- OpenAI API key for the bot's intelligence
- Cartesia API key for text-to-speech

## Setup

1. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Set up environment variables

Copy the example file and fill in your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Configure your Twilio webhook

In the Twilio console:

- Go to your phone number's configuration
- Set the webhook for "A Call Comes In" to your server's URL + "/call"
- For local testing, you can use ngrok to expose your local server

```bash
ngrok http 8000
# Then use the provided URL (e.g., https://abc123.ngrok.io/call) in Twilio
```

## Running the Server

Start the webhook server:

```bash
python server.py
```

## Testing

Call your Twilio phone number. The system should answer the call, put you on hold briefly, then connect you with the bot.

## Customizing the Bot

You can customize the bot's behavior by modifying the system prompt in `bot.py`.

## Troubleshooting

### Call is not being answered

- Check that your Twilio webhook is correctly configured
- Verify your Twilio account has sufficient funds
- Check the logs of both the server and bot processes

### Call connects but no bot is heard

- Ensure your Daily API key is correct and has SIP capabilities
- Check that the SIP endpoint is being correctly passed to the bot
- Verify that the ElevenLabs API key and voice ID are correct

### Bot starts but disconnects immediately

- Check the Daily and Twilio logs for any error messages
- Ensure your server has stable internet connectivity
