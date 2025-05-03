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

### Changing the Hold Music

To change the ringing sound or hold music that callers hear while waiting to be connected to the bot, update the URL in `server.py`:

```python
resp = VoiceResponse()
resp.play(
    url="https://your-custom-audio-file-url.mp3",
    loop=10,
)
```

> Read [Twilio's guide](https://www.twilio.com/en-us/blog/adding-mp3-to-voice-call-using-twilio) on how to set up an mp3 in a voice call.

## Handling Multiple SIP Endpoints

The bot is configured to handle multiple `on_dialin_ready` events that might occur with multiple SIP endpoints. It ensures that each call is only forwarded once using a simple flag:

```python
# Flag to track if call has been forwarded
call_already_forwarded = False

@transport.event_handler("on_dialin_ready")
async def on_dialin_ready(transport, cdata):
    nonlocal call_already_forwarded

    # Skip if already forwarded
    if call_already_forwarded:
        logger.info("Call already forwarded, ignoring this event.")
        return

    # ... forwarding code ...
    call_already_forwarded = True
```

Note that normally calls only require a single SIP endpoint. If you are planning to forward the call to a different number, you will need to set up 2 SIP endpoints: one for the initial call and one for the forwarded call. IMPORTANT: ensure that your `on_dialin_ready` handler only handles the first call.

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

- Check that your Twilio webhook is correctly configured
- Verify your Twilio account has sufficient funds
- Check the logs of both the server and bot processes

### Call connects but no bot is heard

- Ensure your Daily API key is correct and has SIP capabilities
- Check that the SIP endpoint is being correctly passed to the bot
- Verify that the Cartesia API key and voice ID are correct

### Bot starts but disconnects immediately

- Check the Daily and Twilio logs for any error messages
- Ensure your server has stable internet connectivity
