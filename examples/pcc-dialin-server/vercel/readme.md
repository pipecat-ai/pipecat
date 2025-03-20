# Daily API Backend

Next.js API routes for handling Daily Pipecat requests.

## Features

- API endpoint for handling Daily Pipecat requests
- HMAC signature validation
- Structured logging with Pino
- Support for dialin and dialout settings
- Voicemail detection and call transfer functionality
- Test request handling

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Create `.env.local` file with your credentials:
   ```
   DAILY_API_KEY=your_daily_api_key
   AGENT_NAME=my-first-agent
   PINLESS_HMAC_SECRET=your_hmac_secret
   LOG_LEVEL=info
   ```
4. Run the development server:
   ```bash
   npm run dev
   ```

## API Endpoints

### GET /api
Returns a simple "Hello, World!" message with a cute cat emoji to verify the server is running.

### POST /api/dial
Handles dialin and dialout requests for Daily Pipecat.

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
    {"phoneNumber": "+14158483432", "callerId": "purchased_phone_uuid"}, 
    {"sipUri": "sip:username@sip.hostname.com"}
  ],
}
```

## Deployment

The application is configured for Vercel deployment:

1. Push your code to a Git repository
2. Import your project in Vercel dashboard
3. Configure environment variables:
   - `DAILY_API_KEY`
   - `AGENT_NAME`
   - `PINLESS_HMAC_SECRET`
   - `LOG_LEVEL` (optional, defaults to 'info')
4. Deploy!

## Security

- HMAC signature validation for request authentication
- Environment variables for sensitive credentials
- Method validation (POST only for /dial)