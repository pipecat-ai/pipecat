<!-- @format -->

<div align="center">
 <img alt="pipecat" width="300px" height="auto" src="image.png">
</div>

# Pipecat Phone Chatbot

This repository contains examples for building intelligent phone chatbots using AI for various use cases including:

- **Simple dial-in**: Basic incoming call handling
- **Simple dial-out**: Basic outgoing call handling
- **Voicemail detection**: Bot calls a number, detects if it reaches voicemail or a human, and responds appropriately
- **Call transfer**: Bot handles initial customer interaction and transfers to a human operator when needed

## Architecture Overview

These examples use the following components:

- 🔁 **Transport**: Daily WebRTC
- 💬 **Speech-to-Text**: Deepgram via Daily transport
- 🤖 **LLMs**: Each example uses a specific LLM (OpenAI GPT-4o or Google Gemini)
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

## Example 1: Simple Dial-in

This example demonstrates basic handling of incoming calls without additional features like call transfer.

### Testing in Daily Prebuilt (No Actual Phone Calls)

```shell
curl -X POST "http://localhost:7860/start" \
	 -H "Content-Type: application/json" \
	 -d '{
		 "config": {
			"simple_dialin": {
			   "testInPrebuilt": true
			}
		 }
	  }'
```

This returns a Daily room URL where you can test the bot's basic conversation capabilities.

## Example 2: Simple Dial-out

This example demonstrates basic handling of outgoing calls without additional features like voicemail detection.

### Testing in Daily Prebuilt (No Actual Phone Calls)

```shell
curl -X POST "http://localhost:7860/start" \
	 -H "Content-Type: application/json" \
	 -d '{
		 "config": {
			"simple_dialout": {
			   "testInPrebuilt": true
			}
		 }
	  }'
```

This returns a Daily room URL where you can test the bot's basic conversation capabilities.

### Making Actual Phone Calls

```shell
curl -X POST "http://localhost:7860/start" \
	 -H "Content-Type: application/json" \
	 -d '{
		 "config": {
			"dialout_settings": [{
			   "phoneNumber": "+12345678910"
			}],
			"simple_dialout": {
			   "testInPrebuilt": false
			}
		 }
	  }'
```

## Example 3: Voicemail Detection

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
			"dialout_settings": [{
			   "phoneNumber": "+12345678910"
			}],
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

## Example 4: Call Transfer

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
    "dialout_settings": [
      {
        "phoneNumber": "+12345678910",
        "callerId": "caller-id-uuid",
        "sipUri": "sip:maria@example.com"
      }
    ],
    "call_transfer": {
      "mode": "dialout",
      "speakSummary": true,
      "storeSummary": false,
      "operatorNumber": "+12345678910",
      "testInPrebuilt": false
    },
    "voicemail_detection": {
      "testInPrebuilt": true
    },
    "simple_dialin": {
      "testInPrebuilt": true
    },
    "simple_dialout": {
      "testInPrebuilt": true
    }
  }
}
```

### Configuration Parameters

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
- `simple_dialin`: For simple dialin example:
  - `testInPrebuilt`: Test without actual phone calls
- `simple_dialout`: For simple dialout example:
  - `testInPrebuilt`: Test without actual phone calls

## Feature Compatibility

The following table shows which feature combinations are supported when making requests to the `/start` endpoint. The table is organized by use case to help you create the correct configuration.

| Use Case                                                        | `call_transfer` | `voicemail_detection` | `simple_dialin` | `simple_dialout` | `dialin_settings` | `dialout_settings` | `operatorNumber` | `testInPrebuilt` | Status           |
| --------------------------------------------------------------- | --------------- | --------------------- | --------------- | ---------------- | ----------------- | ------------------ | ---------------- | ---------------- | ---------------- |
| **Basic incoming call handling (simple_dialin)**                | ✗               | ✗                     | ✓               | ✗                | ✓                 | ✗                  | ✗                | ✗                | ✅ Supported     |
| **Test mode: Simple dialin in Daily Prebuilt**                  | ✗               | ✗                     | ✓               | ✗                | ✗                 | ✗                  | ✗                | ✓                | ✅ Supported     |
| **Basic outgoing call handling (simple_dialout)**               | ✗               | ✗                     | ✗               | ✓                | ✗                 | ✓                  | ✗                | ✗                | ✅ Supported     |
| **Test mode: Simple dialout in Daily Prebuilt**                 | ✗               | ✗                     | ✗               | ✓                | ✗                 | ✗                  | ✗                | ✓                | ✅ Supported     |
| **Standard call transfer (incoming call)**                      | ✓               | ✗                     | ✗               | ✗                | ✓                 | ✗                  | ✓/✗              | ✗                | ✅ Supported     |
| **Standard voicemail detection (outgoing call)**                | ✗               | ✓                     | ✗               | ✗                | ✗                 | ✓                  | ✗                | ✗                | ✅ Supported     |
| **Test mode: Call transfer in Daily Prebuilt**                  | ✓               | ✗                     | ✗               | ✗                | ✗                 | ✗                  | ✓                | ✓                | ✅ Supported     |
| **Test mode: Voicemail detection in Daily Prebuilt**            | ✗               | ✓                     | ✗               | ✗                | ✗                 | ✗                  | ✗                | ✓                | ✅ Supported     |
| Call transfer requires operatorNumber                           | ✓               | ✗                     | ✗               | ✗                | ✓                 | ✗                  | ✗                | ✓/✗              | ❌ Not Supported |
| Voicemail detection requires dialout_settings or testInPrebuilt | ✗               | ✓                     | ✗               | ✗                | ✓                 | ✗                  | ✗                | ✓/✗              | ❌ Not Supported |
| Cannot combine different bot types                              | ✓               | ✓                     | ✗               | ✗                | ✓                 | ✓                  | ✓                | ✓/✗              | ❌ Not Supported |
| Call_transfer needs dialin_settings in non-test mode            | ✓               | ✗                     | ✗               | ✗                | ✗                 | ✗                  | ✓                | ✗                | ❌ Not Supported |
| Voicemail_detection needs dialout_settings in non-test mode     | ✗               | ✓                     | ✗               | ✗                | ✗                 | ✗                  | ✗                | ✗                | ❌ Not Supported |
| Insufficient configuration                                      | ✗               | ✗                     | ✗               | ✗                | ✗                 | ✗                  | ✗                | ✓/✗              | ❌ Not Supported |

### Legend:

- ✓: Required
- ✗: Not allowed
- ✓/✗: Optional
- ✅: Supported
- ❌: Not Supported

### Notes:

- `dialin_settings` is typically populated automatically from webhook data for incoming calls
- `dialout_settings` must be specified manually for outgoing calls
- `operatorNumber` is specified within the `call_transfer` object (`"call_transfer": {"operatorNumber": "+1234567890", ...}`)
- `testInPrebuilt` is specified within the bot type object (e.g., `"call_transfer": {"testInPrebuilt": true, ...}`)
- For call transfers, `operatorNumber` must be provided to specify which operator to dial. If it is not provided, we will base it off of the operator map in call_connection_manager.py
- In test mode (`testInPrebuilt: true`), some requirements are relaxed to allow testing in Daily Prebuilt
- Multiple customers to dial out to can be specified by providing an array of objects in `dialout_settings`
- Bot types are mutually exclusive - you cannot combine multiple bot types in a single configuration

### Configuration Examples

#### Standard call transfer (incoming call):

```json
{
  "config": {
    "dialin_settings": {
      "from": "+12345678901",
      "to": "+19876543210",
      "call_id": "call-id-string",
      "call_domain": "domain-string"
    },
    "call_transfer": {
      "mode": "dialout",
      "speakSummary": true,
      "operatorNumber": "+12345678910"
    }
  }
}
```

#### Test mode: Call transfer in Daily Prebuilt:

```json
{
  "config": {
    "call_transfer": {
      "mode": "dialout",
      "speakSummary": true,
      "operatorNumber": "+12345678910",
      "testInPrebuilt": true
    }
  }
}
```

#### Test mode: Voicemail detection in Daily Prebuilt:

```json
{
  "config": {
    "voicemail_detection": {
      "testInPrebuilt": true
    }
  }
}
```

#### Standard voicemail detection:

```json
{
  "config": {
    "dialout_settings": [
      {
        "phoneNumber": "+12345678910"
      }
    ],
    "voicemail_detection": {
      "testInPrebuilt": false
    }
  }
}
```

#### Simple dialin (incoming call):

```json
{
  "config": {
    "dialin_settings": {
      "from": "+12345678901",
      "to": "+19876543210",
      "call_id": "call-id-string",
      "call_domain": "domain-string"
    },
    "simple_dialin": {}
  }
}
```

#### Test mode: Simple dialin in Daily Prebuilt:

```json
{
  "config": {
    "simple_dialin": {
      "testInPrebuilt": true
    }
  }
}
```

#### Simple dialout (outgoing call):

```json
{
  "config": {
    "dialout_settings": [
      {
        "phoneNumber": "+12345678910"
      }
    ],
    "simple_dialout": {}
  }
}
```

#### Test mode: Simple dialout in Daily Prebuilt:

```json
{
  "config": {
    "simple_dialout": {
      "testInPrebuilt": true
    }
  }
}
```

## Deployment

See Pipecat Cloud deployment docs for how to deploy this example: https://docs.pipecat.daily.co/agents/deploy

We also have a great, easy to use quickstart guide here: https://docs.pipecat.daily.co/quickstart

## Using Different LLM Providers

Each example in this repository is implemented with a specific LLM provider:

- **Simple dial-in**: Uses OpenAI
- **Simple dial-out**: Uses OpenAI
- **Voicemail detection**: Uses Google Gemini
- **Call transfer**: Uses OpenAI

If you want to implement one of these examples with a different LLM provider than what's provided:

- To implement **call_transfer** with **Gemini**, reference the `voicemail_detection.py` file for how to structure LLM context, function calling, and other Gemini-specific implementations.
- To implement **voicemail_detection** with **OpenAI**, reference the `call_transfer.py` file for OpenAI-specific implementation details.

The key differences between implementations involve how context is managed, function calling syntax, and message formatting. Looking at both implementations side-by-side provides a good template for adapting any example to your preferred LLM provider.

## Customizing Bot Prompts

All examples include default prompts that work well for standard use cases. However, you can customize how the bot behaves by providing your own prompts in the request body.

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
			"prompts": [
			   {
				  "name": "voicemail_prompt",
				  "text": "Hello, this is ACME Corporation calling. Please call us back at 555-123-4567 regarding your recent order. Thank you!"
			   }
			],
			"dialout_settings": [{
			   "phoneNumber": "+12345678910"
			}],
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
