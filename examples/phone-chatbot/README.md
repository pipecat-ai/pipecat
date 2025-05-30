<!-- @format -->

<div align="center">
 <img alt="pipecat" width="300px" height="auto" src="image.png">
</div>

# Pipecat Phone Chatbot

This repository contains examples for building intelligent phone chatbots using AI for various use cases including:

- **daily-pstn-dial-in**: Basic incoming call handling
- **daily-pstn-dial-out**: Basic outgoing call handling
- **daily-twilio-sip-dial-in**: Basic incoming call handling using Daily SIP + Twilio
- **daily-twilio-sip-dial-out**: Basic outgoing call handling using Daily SIP + Twilio
- **daily-pstn-simple-voicemail-detection**: Voicemail detection Bot. Bot calls a number, detects if it reaches voicemail or a human, and responds appropriately
- **daily-pstn-advanced-voicemail-detection**: A more advanced example of the voicemail detection bot. Utilises multiple pipelines. The first pipeline uses a much simpler, faster and cheaper LLM to detect the voicemail machine. Then switches to a more powerful LLM if it needs to have a conversation with a user. You should use this one if you want to use different LLMs for different tasks, it also shows how to do audio input for one LLM (multimodal LLM) and then STT for the other one. Switching out methods of sending data to the LLM
- **daily-pstn-simple-call-transfer**: Bot handles initial customer interaction and transfers to a human operator when needed

## Architecture Overview

These examples use the following components:

- 🔁 **Transport**: Daily WebRTC
- 💬 **Speech-to-Text**: Deepgram via Daily transport, or via separate Deepgram service
- 🤖 **LLMs**: Each example uses a specific LLM (OpenAI GPT-4o or Google Gemini)
- 📞 **SIP/PSTN**: Examples either use Daily PSTN or SIP with a SIP provider such as Twilio
- 🔉 **Text-to-Speech**: Cartesia

### Phone Number Provider: Daily vs Twilio

If you're starting from scratch, we recommend using Daily to provision phone numbers alongside Daily as a transport for simplicity (this provides automatic call forwarding).

If you already have Twilio numbers and workflows, you can connect them to your Pipecat bots with some additional configuration (`on_dialin_ready` and using the Twilio client to trigger forwarding).

The Twilio dial-out example shows you how to configure the SIP URI domain and TwiML bins.

## Deployment

See Pipecat Cloud deployment docs for how to deploy this example: https://docs.pipecat.daily.co/agents/deploy

We also have a great, easy to use quickstart guide here: https://docs.pipecat.daily.co/quickstart

## Using Different LLM Providers

Each example in this repository is implemented with a specific LLM provider:

- **daily-pstn-dial-in**: Uses OpenAI
- **daily-pstn-dial-out**: Uses OpenAI
- **daily-twilio-sip-dial-in**: Uses OpenAI
- **daily-twilio-sip-dial-out**: Uses OpenAI
- **daily-pstn-simple-voicemail-detection**: Uses Google Gemini Flash 2.0
- **daily-pstn-simple-voicemail-detection**: Uses Google Gemini Flash Lite 2.0 and Flash 2.0
- **daily-pstn-simple-call-transfer**: Uses OpenAI

If you want to implement one of these examples with a different LLM provider than what's provided:

- To implement **call_transfer** with **Gemini**, reference the `bot.py` file inside the voicemail detection example for how to structure LLM context, function calling, and other Gemini-specific implementations.
- To implement **voicemail_detection** with **OpenAI**, reference the `bot.py` file inside the call_transfer example for OpenAI-specific implementation details.

The key differences between implementations involve how context is managed, function calling syntax, and message formatting. Looking at both implementations side-by-side provides a good template for adapting any example to your preferred LLM provider.

## Customizing Bot Prompts

All examples include default prompts that work well for standard use cases.

## Advanced Usage

For more advanced phone integration scenarios using PSTN/SIP, please reach out on [Discord](https://discord.gg/pipecat).
