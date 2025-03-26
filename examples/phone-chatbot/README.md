<!-- @format -->

<div align="center">
 <img alt="pipecat" width="300px" height="auto" src="image.png">
</div>

# Pipecat Phone Chatbot

This repository contains examples for building intelligent phone chatbots using AI for various use cases including:

- **Voicemail detection**: Bot calls a number, detects if it reaches voicemail or a human, and responds appropriately
- **Call transfer**: Bot handles initial customer interaction and transfers to a human operator when needed

Each example is implemented for multiple LLM providers:

- OpenAI (GPT-4o)
- Google (Gemini Flash Lite 2.0 and Flash 2.0)

## Architecture Overview

These examples use the following components:

- 🔁 **Transport**: Daily WebRTC
- 💬 **Speech-to-Text**: Deepgram via Daily transport
- 🤖 **LLMs**: OpenAI GPT4-o / Google Gemini Flash Lite 2.0 and Flash 2.0
- 🔉 **Text-to-Speech**: Cartesia

## Getting Started

### Prerequisites

1. Create and activate a virtual environment:

   ```shell
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install requirements:

   ```shell
   pip install -r requirements.txt
   ```

3. Set up your environment variables:

   ```shell
   cp env.example .env
   ```

   Edit the `.env` file to include your API keys.

4. Install [ngrok](https://ngrok.com/) to make your local server accessible to external services.

### Phone Number Provider: Daily vs Twilio

If you're starting from scratch, we recommend using Daily to provision phone numbers alongside Daily as a transport for simplicity (this provides automatic call forwarding).

If you already have Twilio numbers and workflows, you can connect them to your Pipecat bots with some additional configuration (`on_dialin_ready` and using the Twilio client to trigger forwarding).

Most examples in this repository show how to use Daily for dial-in/dial-out operations.

## Running the Examples

### 1. Start the Bot Runner Service

The bot runner handles incoming requests and manages bot processes:

```shell
python bot_runner.py --host localhost
```

### 2. Create a Public Endpoint with ngrok

Start ngrok to create a public URL for your local server:

```shell
ngrok http --domain yourdomain.ngrok.app 7860
```

## Example 1: Voicemail Detection

This example demonstrates a bot that can dial out to a phone number, detect whether it reached a human or voicemail system, and respond appropriately.

### How It Works

1. Bot dials a phone number
2. Bot listens to determine if it's connected to a person or voicemail
3. If it detects voicemail, it leaves a predefined message and hangs up
4. If it detects a human, it engages in conversation

### Testing in Daily Prebuilt (No Actual Phone Calls)

To test without making actual phone calls:

```shell
curl -X POST "http://localhost:7860/start" \
     -H "Content-Type: application/json" \
     -d '{
         "config": {
            "llm": "openai",
            "voicemail_detection": {
               "testInPrebuilt": true
            }
         }
      }'
```

For testing with Gemini:

```shell
curl -X POST "http://localhost:7860/start" \
     -H "Content-Type: application/json" \
     -d '{
         "config": {
            "llm": "gemini",
            "voicemail_detection": {
               "testInPrebuilt": true
            }
         }
      }'
```

This will return a Daily room URL you can use to test the bot in the browser.

### Making Actual Phone Calls

To have the bot dial out to a real phone number:

```shell
curl -X POST "http://localhost:7860/start" \
     -H "Content-Type: application/json" \
     -d '{
         "config": {
            "llm": "openai",
            "dialout_settings": {
               "phoneNumber": "+12345678910"
            },
            "voicemail_detection": {
               "testInPrebuilt": false
            }
         }
      }'
```

> **Note:** To enable dial-out capabilities, you must first:
>
> 1. Contact [help@daily.co](mailto:help@daily.co) to enable dial-out for your domain
> 2. Purchase a phone number to dial out from
> 3. Ensure rooms have dial-out enabled (the bot runner handles this)
> 4. Use an owner token for the bot (also handled by the bot runner)

## Example 2: Call Transfer

This example demonstrates a bot that handles initial customer interaction and can transfer the call to a human operator when requested.

### How It Works

1. Customer calls in and speaks with the bot
2. When the customer asks for a supervisor/manager, the bot initiates a transfer
3. The bot dials out to an appropriate operator
4. When the operator joins, the bot summarizes the conversation
5. The bot remains silent while operator and customer talk
6. When the operator leaves, the bot resumes handling the call

### Testing in Daily Prebuilt (No Actual Phone Calls)

```shell
curl -X POST "http://localhost:7860/start" \
     -H "Content-Type: application/json" \
     -d '{
         "config": {
            "llm": "openai",
            "call_transfer": {
               "mode": "dialout",
               "speakSummary": true,
               "storeSummary": false,
               "operatorNumber": "+12345678910",
               "testInPrebuilt": true
            }
         }
      }'
```

This returns a Daily room URL. In the room, the expected flow is:

1. Join the room and speak with the bot
2. Ask to speak with a manager/supervisor
3. The bot will add the "operator" to the call
4. The bot will summarize the conversation and then go silent
5. To simulate the operator, you can mute yourself in Daily Prebuilt and speak as if you're the operator
6. When finished, have the "operator" leave the call
7. The bot will resume speaking and can recall details from the conversation
8. End the call by closing Daily Prebuilt or telling the bot you're done

### Using with Real Phone Calls

For incoming calls from customers, Daily will send a webhook to your `/start` endpoint. This webhook contains:

```json
{
	"From": "+CALLERS_PHONE",
	"To": "$PURCHASED_PHONE",
	"callId": "callid-read-only-string",
	"callDomain": "callDomain-read-only-string"
}
```

The system will:

1. Identify the customer based on their phone number
2. Determine the appropriate operator to contact
3. Customize the bot's behavior based on transfer settings

#### Operator Assignment

The `call_connection_manager.py` file contains mappings for:

1. `CUSTOMER_MAP`: Links phone numbers to customer names
2. `OPERATOR_CONTACT_MAP`: Contains operator contact information
3. `CUSTOMER_TO_OPERATOR_MAP`: Defines which operators should handle which customers

You can customize these mappings or integrate with your existing customer database.

## Configuration Options

### Request Body Structure

When making requests to the `/start` endpoint, the config object can include:

```json
{
	"config": {
		"llm": "openai",
		"prompts": [
			{
				"name": "call_transfer_initial_prompt",
				"text": "Your custom prompt here"
			},
			{
				"name": "call_transfer_prompt",
				"text": "Your custom prompt here"
			},
			{
				"name": "call_transfer_finished_prompt",
				"text": "Your custom prompt here"
			},
			{
				"name": "voicemail_detection_prompt",
				"text": "Your custom prompt here"
			},
			{
				"name": "voicemail_prompt",
				"text": "Your custom prompt here"
			},
			{
				"name": "human_conversation_prompt",
				"text": "Your custom prompt here"
			}
		],
		"dialin_settings": {
			"From": "+CALLERS_PHONE",
			"To": "$PURCHASED_PHONE",
			"callId": "callid-read-only-string",
			"callDomain": "callDomain-read-only-string"
		},
		"dialout_settings": {
			"phoneNumber": "+12345678910",
			"callerId": "caller-id-uuid",
			"sipUri": "sip:maria@example.com"
		},
		"call_transfer": {
			"mode": "dialout",
			"speakSummary": true,
			"storeSummary": false,
			"operatorNumber": "+12345678910",
			"testInPrebuilt": false
		},
		"voicemail_detection": {
			"testInPrebuilt": true
		}
	}
}
```

### Configuration Parameters

- `llm`: Specifies which LLM to use (`"openai"` or `"gemini"`)
- `prompts`: An array of objects containing prompts that you want the examples to use.
- `dialin_settings`: Information about incoming calls (typically from webhook)
- `dialout_settings`: For outbound calls:
  - `phoneNumber`: Number to dial
  - `callerId`: UUID of the number to display (optional)
  - `sipUri`: SIP URI to connect to (alternative to phoneNumber)
- `call_transfer`: For call transfer example:
  - `mode`: Currently only `"dialout"` is supported
  - `speakSummary`: Whether the bot should summarize the conversation for the operator
  - `storeSummary`: For future implementation
  - `operatorNumber`: Operator phone number
  - `testInPrebuilt`: Test without actual phone calls
- `voicemail_detection`: For voicemail detection example:
  - `testInPrebuilt`: Test without actual phone calls

## Using Twilio (Alternative)

To use Twilio for call handling:

1. Start the bot runner:

   ```shell
   python bot_runner.py --host localhost
   ```

2. Start ngrok:

   ```shell
   ngrok http --domain yourdomain.ngrok.app 7860
   ```

3. In another terminal, run the Twilio bot:
   ```shell
   python bot_twilio.py
   ```

Make requests to `/start_twilio_bot` for Twilio-specific functionality.

## Deployment

A Dockerfile is included for containerized deployment. Here's how to deploy to [fly.io](https://fly.io):

1. Build the Docker image:

   ```shell
   docker build -t tag:project .
   ```

2. Prepare the fly.toml file:

   ```shell
   mv fly.example.toml fly.toml
   ```

3. Launch the fly project:

   ```shell
   fly launch
   ```

4. Set secrets:

   ```shell
   fly secrets set DAILY_API_KEY=... OPENAI_API_KEY=... CARTESIA_API_KEY=... DEEPGRAM_API_KEY=...
   ```

   For Twilio:

   ```shell
   fly secrets set TWILIO_ACCOUNT_SID=... TWILIO_AUTH_TOKEN=...
   ```

5. Deploy:
   ```shell
   fly deploy
   ```

> **Note:** This demo spawns agents as subprocesses for demonstration purposes. For production, consider a different architecture to better handle concurrent calls.

## Customizing Bot Prompts

Both examples include default prompts that work well for standard use cases. However, you can customize how the bot behaves by providing your own prompts in the request body.

### Available Prompt Types

- `call_transfer_initial_prompt`: The initial prompt the bot uses when greeting a customer
- `call_transfer_prompt`: Instructions for the bot when summarizing the conversation for an operator
- `call_transfer_finished_prompt`: Instructions for when the operator leaves the call
- `voicemail_detection_prompt`: Instructions for detecting whether a call connected to voicemail
- `voicemail_prompt`: The message to leave when voicemail is detected
- `human_conversation_prompt`: Instructions for conversation when a human is detected

### Customization Example

```shell
curl -X POST "http://localhost:7860/start" \
     -H "Content-Type: application/json" \
     -d '{
         "config": {
            "llm": "openai",
            "prompts": [
               {
                  "name": "voicemail_prompt",
                  "text": "Hello, this is ACME Corporation calling. Please call us back at 555-123-4567 regarding your recent order. Thank you!"
               }
            ],
            "dialout_settings": {
               "phoneNumber": "+12345678910"
            },
            "voicemail_detection": {
               "testInPrebuilt": false
            }
         }
      }'
```

This example would use all default prompts except for the voicemail message, which would be replaced with your custom message.

### Template Variables

Some prompts support template variables that are automatically replaced:

- `{customer_name}`: Will be replaced with the customer's name if available

## Advanced Usage

For more advanced phone integration scenarios using PSTN/SIP, please reach out on [Discord](https://discord.gg/pipecat).
