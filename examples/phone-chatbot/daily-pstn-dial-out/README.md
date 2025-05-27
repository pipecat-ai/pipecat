# Daily PSTN dial-out simple chatbot

This project demonstrates how to create a voice bot that uses Dailys PSTN capabilities to make calls to phone numbers.

## How it works

1. The server file receives a curl request with the phone number to dial out to
2. The server creates a Daily room with SIP capabilities
3. The server starts the bot process with the room details
4. When the bot has joined, it starts the dial-out process and rings the number provided in the curl request
5. The user then answers the phone and the user is brought into the call
6. The end user and bot are connected, and the bot handles the conversation

## Prerequisites

- A Daily account with an API key, and a phone number purchased through Daily
- A US phone number to ring
- dial-out must be enabled on your domain. Find out more by reading this [document and filling in the form](https://docs.daily.co/guides/products/dial-in-dial-out#main)
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

3. Buy a phone number

Instructions on how to do that can be found at this [docs link:](https://docs.daily.co/reference/rest-api/phone-numbers/buy-phone-number)

4. Request dial-out enablement

For compliance reasons, to enable dial-out for your Daily account, you must request enablement via the form. You can find out more about dial-out, and the form at the [link here:](https://docs.daily.co/guides/products/dial-in-dial-out#main)

## Running the Server

Start the webhook server:

```bash
python server.py
```

## Testing

With server.py running, send the following curl command from your terminal:

```bash
curl -X POST "http://127.0.0.1:7860/start" \
  -H "Content-Type: application/json" \
  -d '{
    "dialout_settings": {
      "phone_number": "+12345678910"
    }
  }'
```

The server should make a room. The bot will join the room and then ring the number provided. Answer the call to speak with the bot.

## Customizing the Bot

You can customize the bot's behavior by modifying the system prompt in `bot.py`.

## Multiple SIP Endpoints

For PSTN calls, you only need one SIP endpoint.

## Daily dial-out configuration

The bot configures the Daily rooms with dial-out capabilities using these settings. Note: You also need dial-out to be enabled on the domain, as mentioned earlier on in the README.

```python
properties = DailyRoomProperties(
        sip=sip_params,
        enable_dialout=True,  # Needed for outbound calls if you expand the bot
        enable_chat=False,  # No need for chat in a voice bot
        start_video_off=True,  # Voice only
)
```

## Troubleshooting

### I get an error about dial-out not being enabled

- Check that your room has `enable_dialout=True` set
- Check that your meeting token is an owner token (The bot does this for you automatically)
- Check that you have purchased a phone number to ring from
- Check that the phone number you are trying to ring is correct, and is a US or Canadian number.

### Call connects but no bot is heard

- Ensure your Daily API key is correct and has SIP capabilities
- Verify that the Cartesia API key and voice ID are correct

### Bot starts but disconnects immediately

- Check the Daily logs for any error messages
- Ensure your server has stable internet connectivity
