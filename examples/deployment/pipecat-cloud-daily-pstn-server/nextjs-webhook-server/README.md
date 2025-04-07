# Next.js server for handling Daily PSTN/SIP Webhook

Next.js API routes for handling Daily PSTN/SIP Pipecat requests.

## Features

- API endpoint for handling Daily PSTN/SIP Pipecat requests
- HMAC signature validation
- Structured logging with Pino
- Support for dial-in and dial-out settings
- Voicemail detection and call transfer functionality
- Test request handling

## Setup

1. Clone the repository

2. Navigate to the `nextjs-webhook-server` directory:

   ```bash
   cd nextjs-webhook-server
   ```

3. Install dependencies:

   ```bash
   npm install
   ```

4. Create `.env.local` file with your credentials:

   ```bash
   cp env.local.example .env.local
   ```

5. Update your `.env` with your secrets:

   ```bash
   PIPECAT_CLOUD_API_KEY=pk_*
   AGENT_NAME=my-first-agent
   PINLESS_HMAC_SECRET=your_hmac_secret
   LOG_LEVEL=info
   ```

### Running the server

Run the development server:

```bash
npm run dev
```

The server will run on `http://localhost:7860` and you can expose it via ngrok for testing:

```bash
`ngrok http 7860`
```

> Tip: Use a subdomain for a consistent URL (e.g. `ngrok http -subdomain=mydomain http://localhost:7860`)

## API Endpoints

### GET /api

Returns a simple "Hello, World!" message with a cute cat emoji to verify the server is running.

### POST /api/dial

Handles dial-in and dial-out requests for Pipecat Cloud.

#### Test Requests

The endpoint handles test requests when a webhook is configured. Send a request with `"Test": "test"` to verify your setup:

```json
{
  "Test": "test"
}
```

#### Production Request Format

```json
{
  // for dial-in from webhook
  "To": "+14152251493",
  "From": "+14158483432",
  "callId": "string-contains-uuid",
  "callDomain": "string-contains-uuid",
  // for making a dial out to a phone or SIP
  "dialout_settings": [
    { "phoneNumber": "+14158483432", "callerId": "purchased_phone_uuid" },
    { "sipUri": "sip:username@sip.hostname.com" }
  ]
}
```

## Deployment

The application is configured for Vercel deployment:

1. Push your code to a Git repository
2. Import your project in Vercel dashboard
3. Configure environment variables:
   - `PIPECAT_CLOUD_API_KEY`
   - `AGENT_NAME`
   - `PINLESS_HMAC_SECRET`
   - `LOG_LEVEL` (optional, defaults to 'info')
4. Deploy!

## Security

- HMAC signature validation for request authentication
- Environment variables for sensitive credentials
- Method validation (POST only for /dial)
