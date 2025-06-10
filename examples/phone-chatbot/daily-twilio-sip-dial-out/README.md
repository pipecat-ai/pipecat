<!-- @format -->

# Daily + Twilio SIP dial-out Voice Bot

This project demonstrates how to create a voice bot that can make phone calls via Twilio and use Daily's SIP capabilities to enable voice conversations.

## How it works

1. The server file receives a curl request with the SIP uri to dial out to
2. The server creates a Daily room with SIP capabilities
3. The server starts the bot process with the room details
4. When the bot has joined, it starts the dial-out process and dials out to the SIP uri provided in the curl request
5. Twilio receives the request, and the provided TWIML processes the SIP uri
6. Twilio then rings the number found within the SIP uri
7. When the user answers the phone, the user is brought into the call
8. The end user and the bot are connected, and the bot handles the conversation

## Prerequisites

- A Daily account with an API key
- A Twilio account with a phone number that supports voice and a correctly configured SIP domain
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

3. Create a TwiML Bin

Visit this link to create your [TwiML Bin](https://www.twilio.com/docs/serverless/twiml-bins)

- Login to the account that has your purchased Twilio phone number
- Press the plus button on the TwiML Bin dashboard to write a new TwiML that Twilio will host for you
- Give it a friendly name. For example "daily sip uri twiml bin"
- For the TWIML code, use something like:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Dial callerId="+1234567890">{{#e164}}{{To}}{{/e164}}</Dial>
</Response>
```

- callerId must be a valid number that you own on [Twilio](https://console.twilio.com/us1/develop/phone-numbers/manage/incoming)
- Save the file. We will use this when creating the SIP domain

4. Create and configure a programmable SIP domain

- Visit this link to [create a new SIP domain:](https://console.twilio.com/us1/develop/voice/manage/sip-domains?frameUrl=%2Fconsole%2Fvoice%2Fsip%2Fendpoints%3Fx-target-region%3Dus1)
- Press the plus button to create a new SIP domain
- Give the SIP domain a friendly name. For example "Daily SIP domain"
- Specify a SIP URI, for example "daily.sip.twilio.com"
- Under "Voice Authentication", press the plus button next to IP Access Control Lists. We are going to white list the entire IP spectrum
- Give it a friendly name such as "first half"
- For CIDR Network Address specify 0.0.0.0 and for the subnet specify 1
- Again, specify "first half" for the friendly name and click "Create ACL"
- Now let's do the same again and add another IP Access Control List by pressing the plus button
- Give it a friendly name such as "second half".
- For the CIDR Network Address specify 128.0.0.0 and for the subnet specify 1
- Lastly, specify the friendly name "second half" again
- Make sure both IP Access control list appears selected in the dropdown
- Under "Call Control Configuration", specify the following:
- Configure with: Webhooks, TwiML Bins, Functions, Studio, Proxy
- A call comes in: TwiML Bin > Select the name of the TwiML bin you made earlier
- Leave everything else blank and scroll to the bottom of the page. Click save

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
      "sip_uri": "sip:+1234567891@daily.sip.twilio.com"
    }
  }'
```

- Replace the phone number (Starting with +1) with the phone number you want to ring
- Replace daily with the SIP domain you configured previously

The server should make a room. The bot will join the room and then dial out to the SIP URI provided. Answer the call to speak with the bot.

## Customizing the Bot

You can customize the bot's behavior by modifying the system prompt in `bot.py`.

## Handling Multiple SIP Endpoints

Note that normally calls only require a single SIP endpoint. If you are planning to forward the call to a different number, you will need to set up 2 SIP endpoints: one for the initial call and one for the forwarded call.

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
- Check that the SIP URI is correct
- Check that the phone number you are trying to ring is correct

### I'm stuck setting up my Twilio account

- You can reference this [Notion doc](https://dailyco.notion.site/PUBLIC-Doc-Integration-Twilio-PSTN-Daily-s-SIP-Dialout-1cfdaed630f5458d9d4fc0e3f29ec559) to find more information on how to set up Twilio, as well as use webhooks instead of TwiML Bins

### Call connects but no bot is heard

- Ensure your Daily API key is correct and has SIP capabilities
- Verify that the Cartesia API key and voice ID are correct

### Bot starts but disconnects immediately

- Check the Daily logs for any error messages
- Ensure your server has stable internet connectivity
