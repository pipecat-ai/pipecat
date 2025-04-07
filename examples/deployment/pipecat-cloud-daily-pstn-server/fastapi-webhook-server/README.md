# FastAPI server for handling Daily PSTN/SIP Webhook

A FastAPI server that handles PSTN (Public Switched Telephone Network) and SIP (Session Initiation Protocol) calls using the Daily API.

## Setup

1. Clone the repository

2. Navigate to the `fastapi-webhook-server` directory:

   ```bash
   cd fastapi-webhook-server
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy `env.example` to `.env`:

   ```bash
   cp env.example .env
   ```

5. Update `.env` with your credentials:

   - `AGENT_NAME`: Your Daily agent name
   - `PIPECAT_CLOUD_API_KEY`: Your Daily API key
   - `PINLESS_HMAC_SECRET`: Your HMAC secret for request verification

## Running the Server

Start the server:

```bash
python server.py
```

The server will run on `http://localhost:7860` and you can expose it via ngrok for testing:

```bash
`ngrok http 7860`
```

> Tip: Use a subdomain for a consistent URL (e.g. `ngrok http -subdomain=mydomain http://localhost:7860`)

## API Endpoints

### GET /

Health check endpoint that returns a "Hello, World!" message.

### POST /api/dial

Initiates a PSTN/SIP call with the following request body format:

```json
{
  "To": "+14152251493",
  "From": "+14158483432",
  "callId": "string-contains-uuid",
  "callDomain": "string-contains-uuid",
  "dialout_settings": [
    {
      "phoneNumber": "+14158483432",
      "callerId": "+14152251493"
    }
  ],
  "voicemail_detection": {
    "testInPrebuilt": true
  },
  "call_transfer": {
    "mode": "dialout",
    "speakSummary": true,
    "storeSummary": true,
    "operatorNumber": "+14152250006",
    "testInPrebuilt": true
  }
}
```

#### Response

Returns a JSON object containing:

- `status`: Success/failure status
- `data`: Response from Daily API
- `room_properties`: Properties of the created Daily room

## Error Handling

- 401: Invalid signature
- 400: Invalid authorization header (e.g. missing Daily API key in bot.py)
- 405: Method not allowed (e.g. incorrect route on the webhook URL)
- 500: Server errors (missing API key, network issues)
- Other status codes are passed through from the Daily API
