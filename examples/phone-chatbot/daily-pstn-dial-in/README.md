<!-- @format -->

# Daily PSTN dial-in simple chatbot

This project demonstrates how to create a voice bot that can receive phone calls via Dailys PSTN capabilities to enable voice conversations.

## How It Works

1. Daily receives an incoming call to your phone number.
2. Daily calls your webhook server (`/start` endpoint).
3. The server creates a Daily room with dial-in capabilities
4. The server starts the bot process with the room details
5. The caller is put on hold with music
6. The bot joins the Daily room and signals readiness
7. Daily forwards the call to the Daily room
8. The caller and the bot are connected, and the bot handles the conversation

## Prerequisites

- A Daily account with an API key
- An OpenAI API key for the bot's intelligence
- A Cartesia API key for text-to-speech

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

3. Buy a phone number

Instructions on how to do that can be found at this [docs link:](https://docs.daily.co/reference/rest-api/phone-numbers/buy-phone-number)

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
```

## Troubleshooting

### Call is not being answered

- Check that your dial-in config is correctly configured to point towards your ngrok server and correct endpoint
- Make sure the server.py file is running
- Make sure ngrok is correctly setup and pointing to the correct port

### Call connects but no bot is heard

- Ensure your Daily API key is correct and has SIP capabilities
- Verify that the Cartesia API key and voice ID are correct

### Bot starts but disconnects immediately

- Check the Daily logs for any error messages
- Ensure your server has stable internet connectivity
