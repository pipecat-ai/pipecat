<!-- @format -->

<div align="center">
 <img alt="pipecat" width="300px" height="auto" src="image.png">
</div>

# Phone Chatbot

Example project that demonstrates how to add phone funtionality to your Pipecat bots. We include examples for Daily (`bot_daily.py`) dial-in and dial-out, and Twilio (`bot_twilio.py`) dial-in, depending on who you want to use as a phone vendor.

- 🔁 Transport: Daily WebRTC
- 💬 Speech-to-Text: Deepgram via Daily transport
- 🤖 LLM: GPT4-o / OpenAI
- 🔉 Text-to-Speech: ElevenLabs

#### Should I use Daily or Twilio as a vendor?

If you're starting from scratch, using Daily to provision phone numbers alongside Daily as a transport offers some convenience (such as automatic call forwarding.)

If you already have Twilio numbers and workflows that you want to connect to your Pipecat bots, there is some additional configuration required (you'll need to create a `on_dialin_ready` and use the Twilio client to trigger the forward.)

You can read more about this, as well as see respective walkthroughs in our docs.

## Setup

1. Create and activate a virtual environment:
   ```shell
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install requirements:
   ```shell
   pip install -r requirements.txt
   ```
3. Copy env.example to .env and configure:
   ```shell
   cp env.example .env
   ```
4. Install [ngrok](https://ngrok.com/) so your local server can receive requests from Daily's servers.

## Using Daily numbers

### Running the example

To run either the dial-in or dial-out example, follow these steps to get started:

1. Run `bot_runner.py` to handle incoming HTTP requests:

   ```shell
   python bot_runner.py --host localhost
   ```

2. Start ngrok running in a terminal window:

   ```shell
   ngrok http --domain yourdomain.ngrok.app 8000
   ```

3. In a different terminal window, run the Daily bot file:
   ```shell
   python bot_daily.py
   ```

### Dial-in

To dial-in to the bot, you will need to enable dial-in for your Daily domain. Follow [this guide](https://docs.daily.co/guides/products/dial-in-dial-out/dialin-pinless#provisioning-sip-interconnect-and-pinless-dialin-workflow) to set up your domain.

Note: For the `room_creation_api` property, point at your ngrok hostname: `"room_creation_api": "https://yourdomain.ngrok.app/daily_start_bot"`.

Once your domain is configured, receiving a phone call at a number associated with your Daily account will result in a POST to the `/daily_start_bot` endpoint, which will start a bot session.

### Dial-out

For the bot to dial out to a number, make a POST request to `/daily_start_bot` and include the dial-out phone number in the body of the request as `dialoutNumber`.

For example:

```shell
curl -X "POST" "http://localhost:7860/daily_start_bot" \
     -H 'Content-Type: application/json; charset=utf-8' \
     -d $'{
  "dialoutNumber": "+12125551234"
}'
```

### Voicemail detection

To start the bot and test voicemail detection, send a POST request to /daily_start_bot with "detectVoicemail": true in the request body.

- If you only include `"detectVoicemail": true`, the bot will not dial out. Instead, you can test it in Daily Prebuilt by visiting the URL provided in the response.
- If you include both `"detectVoicemail": true` and a phone number under `"dialoutNumber"`, the bot will dial out to that number.

Example: Testing in Daily Prebuilt:

```shell
curl -X POST "http://localhost:7860/daily_start_bot" \                                                                                                        py pipecat
     -H "Content-Type: application/json" \
     -d '{"detectVoicemail": true}'
```

Example: Testing with Dial-Out:

```shell
curl -X POST "http://localhost:7860/daily_start_bot" \                                                                                                        py pipecat
     -H "Content-Type: application/json" \
     -d '{"dialoutNumber": "+18057145330", "detectVoicemail": true}'
```

### New! Using Gemini 2.0 Flash Lite with Daily

We have introduced support for Google's Gemini 2.0 Flash Lite model in this example. This lightweight model offers faster response times and reduced costs while maintaining good conversational capabilities.

**Quick Start**
To use the Gemini-based bot instead of OpenAI:

```shell
curl -X POST "http://localhost:7860/daily_gemini_start_bot" \                                                                                                        py pipecat
     -H "Content-Type: application/json" \
     -d '{"detectVoicemail": true}'
```

All request body parameters supported by /daily_start_bot (such as detectVoicemail, dialoutNumber, etc.) are also compatible with /daily_gemini_start_bot.

This example uses context switching to help steer the bot in the right direction. As Flash Lite is a smaller model, breaking the prompt down into smaller piece helps to improve the bot's accuracy.

For example, instead of giving one large prompt like:

```python
system_instruction="""You are a chatbot that needs to detect if you're talking to a voicemail system or human, then either leave a message or have a conversation. If it's voicemail, say "Hello, this is a message..." and hang up. If it's a human, introduce yourself and be helpful until they say goodbye."""
```

We break it into stages:

First prompt focuses only on detection: "Determine if this is voicemail or human"
After detection, we switch to a new context: either "Leave this specific voicemail message" or "Have a conversation with the human".

**Implementation Details**
The implementation is available in bot_daily_gemini.py and features:

- Staged prompting approach: Breaking down complex tasks into smaller, more focused prompts to improve the lightweight model's performance
- Dynamic context switching: The bot can change its behavior in real-time based on what it detects (voicemail vs. human caller)
- Function-based architecture: Uses function calling to trigger context switches and call termination

### More information

For more configuration options, please consult [Daily's API documentation](https://docs.daily.co).

## Using Twilio numbers

### Running the example

Follow these steps to get started:

1. Run `bot_runner.py` to handle incoming HTTP requests:

   ```shell
   python bot_runner.py --host localhost
   ```

2. Start ngrok running in a terminal window:

   ```shell
   ngrok http --domain yourdomain.ngrok.app 8000
   ```

3. In a different terminal window, run the Daily bot file:
   ```shell
   python bot_twilio.py
   ```

As above, but target the following URL:

`POST /twilio_start_bot`

For more configuration options, please consult Twilio's API documentation.

## Deployment example

A Dockerfile is included in this demo for convenience. Here is an example of how to build and deploy your bot to [fly.io](https://fly.io).

_Please note: This demo spawns agents as subprocesses for convenience / demonstration purposes. You would likely not want to do this in production as it would limit concurrency to available system resources. For more information on how to deploy your bots using VMs, refer to the Pipecat documentation._

### Build the docker image

`docker build -t tag:project .`

### Launch the fly project

`mv fly.example.toml fly.toml`

`fly launch` (using the included fly.toml)

### Setup your secrets on Fly

Set the necessary secrets (found in `env.example`)

`fly secrets set DAILY_API_KEY=... OPENAI_API_KEY=... ELEVENLABS_API_KEY=... ELEVENLABS_VOICE_ID=...`

If you're using Twilio as a number vendor:

`fly secrets set TWILIO_ACCOUNT_SID=... TWILIO_AUTH_TOKEN=...`

### Deploy!

`fly deploy`

## Need to do something more advanced?

This demo covers the basics of bot telephony. If you want to know more about working with PSTN / SIP, please ping us on [Discord](https://discord.gg/pipecat)!
