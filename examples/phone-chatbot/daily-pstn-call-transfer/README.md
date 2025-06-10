# Daily PSTN call transfer

A basic example of how to create a bot that handles the initial customer interaction and then transfers to a human operator when needed

## Architecture Overview

These examples use the following components:

- 🔁 **Transport**: Daily WebRTC
- 💬 **Speech-to-Text**: Deepgram via Daily transport
- 🤖 **LLMs**: Each example uses a specific LLM (OpenAI GPT-4o or Google Gemini)
- 🔉 **Text-to-Speech**: Cartesia

## Prerequisites

- A Daily account with an API key
- An OpenAI API key for the bot's intelligence
- A Cartesia API key for text-to-speech
- One phone to dial-in from and another phone to receive calls when escalating to a manager

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

- Note, please specify an OPERATOR_NUMBER so that the bot can ring a number when escalating to a manager

3. Buy a phone number

Instructions on how to do that can be found at this [docs link:](https://docs.daily.co/reference/rest-api/phone-numbers/buy-phone-number).

4. Set up the dial-in config

Instructions on how to do that can be found at this [docs link:](https://docs.daily.co/reference/rest-api/domainDialinConfig)

5. For local testing, use ngrok to expose your local server

```bash
ngrok http 7860
# Then use the provided URL (e.g., https://abc123.ngrok.io/start) in Twilio
```

## Running the Server

Start the webhook server:

```bash
python server.py
```

## Testing

Call the purchased phone number. The system should answer the call, put you on hold briefly, then connect you with the bot.
Have a short conversation with the bot, and then request to speak with a manager. The bot should then ring the manager. On your second phone, answer the call.

The bot will then summarise the conversation so far, and then silently listen to the conversation. You can now speak with the manager on the other phone.

When the manager hangs up the call, the bot will start speaking again. You can then ask the bot about the conversation with the manager, and it will have the context of the conversation.

## Customizing the Bot

You can customize the bot's behavior by modifying the system prompt in `bot.py`.

## Multiple SIP Endpoints

For PSTN calls, you only need one SIP endpoint.

## Daily SIP Configuration

The bot configures Daily rooms with SIP capabilities using these settings:

```python
sip_params = DailyRoomSipParams(
    display_name="phone-user",  # This will show up in the Daily UI; optional display the dialer's number
    video=False,                # Audio-only call
    sip_mode="dial-in",         # For receiving calls (vs. dial-out)
    num_endpoints=1,            # Number of SIP endpoints to create
)

properties = DailyRoomProperties(
        sip=sip_params,
        enable_dialout=True,  # Needed for outbound calls if you expand the bot
        enable_chat=False,  # No need for chat in a voice bot
        start_video_off=True,  # Voice only
)
```

## Troubleshooting

### Call is not being answered

- Check that your dial-in config is correctly configured to point towards your ngrok server and correct endpoint
- Make sure the server.py file is running
- Make sure ngrok is correctly setup and pointing to the correct port

### The bot does not escalate to the manager

- Check that your room has `enable_dialout=True` set
- Check that your meeting token is an owner token (The bot does this for you automatically)
- Check that the phone number you are trying to ring is correct, and is a US or Canadian number.

### Call connects but no bot is heard

- Ensure your Daily API key is correct and has SIP capabilities
- Verify that the Cartesia API key and voice ID are correct

### Bot starts but disconnects immediately

- Check the Daily logs for any error messages
- Ensure your server has stable internet connectivity
