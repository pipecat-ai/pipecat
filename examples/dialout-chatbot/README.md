<div align="center">
 <img alt="pipecat" width="300px" height="auto" src="image.png">
</div>

# Dialout example

Example project that demonstrates how to add phone number dialout to your Pipecat bots. This example is based on the Dialin example, but only covers the Daily use case, not Twilio.

- 🔁 Transport: Daily WebRTC
- 💬 Speech-to-Text: Deepgram via Daily transport
- 🤖 LLM: GPT4-o / OpenAI
- 🔉 Text-to-Speech: ElevenLabs

## Setup

```shell
# Install the requirements
pip install -r requirements.txt

# Setup your env
mv env.example .env
```

## Using Daily numbers

Run `bot_runner.py` to handle incoming HTTP requests:

`python bot_runner.py --host localhost`

Then target the following URL:

`POST /outbound_call`

For more configuration options, please consult Daily's API documentation.


## Deployment example

A Dockerfile is included in this demo for convenience. Here is an example of how to build and deploy your bot to [fly.io](https://fly.io).

*Please note: This demo spawns agents as subprocesses for convenience / demonstration purposes. You would likely not want to do this in production as it would limit concurrency to available system resources. For more information on how to deploy your bots using VMs, refer to the Pipecat documentation.*

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

This demo covers the basics of bot telephony. If you want to know more about working with PSTN / SIP, please ping us on [Discord](https://discord.gg/pipecat).