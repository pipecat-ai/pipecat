# Handling PSTN/SIP Dial-in on Pipecat Cloud

This repository contains two server implementations for handling
the pinless dial-in workflow in Pipecat Cloud. This is the companion to the
Pipecat Cloud [pstn_sip starter image](https://github.com/daily-co/pipecat-cloud-images/tree/main/pipecat-starters/pstn_sip).
In addition you can use `/api/dial` to trigger dial-out, and
eventually, call-transfers.

1. [FastAPI Server](fastapi-webhook-server/README.md) -
   A FastAPI implementation that handles PSTN (Public Switched Telephone
   Network) and SIP (Session Initiation Protocol) calls using the Daily API.

2. [Next.js Serverless](nextjs-webhook-server/README.md) -
   A Next.js API implementation designed for deployment on Vercel's
   serverless platform.

Both implementations provide:

- HMAC signature validation for pinless webhook
- Structured logging
- Support for dial-in and dial-out settings
- Voicemail detection and call transfer functionality (coming soon)
- Test request handling

## Choosing an Implementation

- Use the **FastAPI Server** if you:

  - Need a standalone server
  - Prefer Python and FastAPI
  - Want to deploy to traditional hosting platforms

- Use the **Next.js Serverless** implementation if you:
  - Want serverless deployment
  - Prefer JavaScript/TypeScript
  - Already use Next.js and Vercel for other projects
  - Need quick scaling and zero maintenance

## Prerequisites

### Environment Variables

Both implementations require similar environment variables:

- `PIPECAT_CLOUD_API_KEY`: Pipecat Cloud API Key, begins with pk\_\*
- `AGENT_NAME`: Your Daily agent name
- `PINLESS_HMAC_SECRET`: Your HMAC secret for request verification
- `LOG_LEVEL`: (Optional) Logging level (defaults to 'info')

See the individual README files in each implementation directory for
specific setup instructions.

### Phone number setup

You can buy a phone number through the Pipecat Cloud Dashboard:

1. Go to `Settings` > `Telephony`
2. Follow the UI to purchase a phone number
3. Configure the webhook URL to receive incoming calls (e.g. `https://my-webhook-url.com/api/dial`)

Or purchase the number using Daily's
[PhoneNumbers API](https://docs.daily.co/reference/rest-api/phone-numbers).

```bash
curl --request POST \
--url https://api.daily.co/v1/domain-dialin-config \
--header 'Authorization: Bearer $TOKEN' \
--header 'Content-Type: application/json' \
--data-raw '{
	"type": "pinless_dialin",
	"name_prefix": "Customer1",
    "phone_number": "+1PURCHASED_NUM",
	"room_creation_api": "https://example.com/api/dial",
    "hold_music_url": "https://example.com/static/ringtone.mp3",
	"timeout_config": {
		"message": "No agent is available right now"
	}
}'
```

The API will return a static SIP URI (`sip_uri`) that can be called
from other SIP services.

### `room_creation_api`

To make and receive calls currently you have to host a server that
handles incoming calls. In the coming weeks, incoming calls will be
directly handled within Daily and we will expose an endpoint similar
to `{service}/start` that will manage this for you.

In the meantime, the server described below serves as the webhook
handler for the `room_creation_api`. Configure your pinless phone
number or SIP interconnect to the `ngrok` tunnel or
the actual server URL, append `/api/dial` to the webhook URL.

## Example curl commands

Note: Replace `http://localhost:3000` with your actual server URL and
phone numbers with valid values for your use case.

### Dialin Request

The server will receive a request when a call is received from Daily.

### Dialout Request

Dial a number, will use any purchased number

```bash
curl -X POST http://localhost:3000/api/dial \
  -H "Content-Type: application/json" \
  -d '{
    "dialout_settings": [
      {
        "phoneNumber": "+1234567890",
      }
    ]
  }'
```

Dial a number with callerId, which is the UUID of a purchased number.

```bash
curl -X POST http://localhost:3000/api/dial \
  -H "Content-Type: application/json" \
  -d '{
    "dialout_settings": [
      {
        "phoneNumber": "+1234567890",
        "callerId": "purchased_phone_uuid"
      }
    ]
  }'
```

Dial a number

```bash
curl -X POST http://localhost:3000/api/dial \
  -H "Content-Type: application/json" \
  -d '{
    "dialout_settings": [
      {
        "phoneNumber": "+1234567890",
        "callerId": "purchased_phone_uuid"
      }
    ]
  }'
```

### Advanced Request with Voicemail Detection

```bash
curl -X POST http://localhost:3000/api/dial \
  -H "Content-Type: application/json" \
  -d '{
    "To": "+1234567890",
    "From": "+1987654321",
    "callId": "call-uuid-123",
    "callDomain": "domain-uuid-456",
    "dialout_settings": [
      {
        "phoneNumber": "+1234567890",
        "callerId": "purchased_phone_uuid"
      }
    ],
    "voicemail_detection": {
      "testInPrebuilt": true
    },
    "call_transfer": {
      "mode": "dialout",
      "speakSummary": true,
      "storeSummary": true,
      "operatorNumber": "+1234567890",
      "testInPrebuilt": true
    }
  }'
```
