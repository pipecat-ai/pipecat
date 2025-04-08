import { logger } from '../../lib/utils'; 
import axios from 'axios';
import crypto from 'crypto';

const validateSignature = (body, signature, timestamp, secret) => {
  // Skip if any required fields are missing
  if (!signature || !timestamp || !secret) {
    logger.warn('Missing required fields for HMAC validation');
    return true;
  }

  try {
    const decodedSecret = Buffer.from(secret, 'base64');
    const hmac = crypto.createHmac('sha256', decodedSecret);
    const signatureData = `${timestamp}.${body}`;
    const computedSignature = hmac.update(signatureData).digest('base64');
    
    logger.debug('Signature validation:', {
      timestamp,
      signatureData: signatureData.substring(0, 50) + '...',
      computedSignature,
      receivedSignature: signature
    });
    
    return computedSignature === signature;
  } catch (error) {
    logger.error('Error validating signature:', error);
    return true; // Allow request to proceed on error
  }
};

export default async function handler(req, res) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    logger.info('Incoming request to /api/dial:');
    logger.info(`Headers: ${JSON.stringify(req.headers)}`);
    
    const rawBody = JSON.stringify(req.body);
    logger.info(`Raw body: ${rawBody}`);

    const signature = req.headers['x-pinless-signature'];
    const timestamp = req.headers['x-pinless-timestamp'];
    
    if (signature && timestamp) {
      logger.info('Validating HMAC signature');
      if (!validateSignature(rawBody, signature, timestamp, process.env.PINLESS_HMAC_SECRET)) {
        logger.error('Invalid HMAC signature', { signature, timestamp });
        return res.status(401).json({ 
          error: 'Invalid signature',
          message: 'Invalid HMAC signature'
        });
      }
    } else {
      logger.info('Skipping HMAC validation - no signature headers present');
    }

    // Extract request data
    const {
      Test: test,
      To,
      From,
      callId,
      callDomain,
      dialout_settings,
      voicemail_detection,
      call_transfer
    } = req.body;

    // Handle test requests when a webhook is configured
    if (test === 'test') {
      logger.debug('Test request received');
      return res.status(200).json({ status: 'success', message: 'Test request received' });
    }

    // Process dialin settings
    let dialin_settings = null;
    const requiredFields = ['To', 'From', 'callId', 'callDomain'];
    
    if (requiredFields.every(field => req.body[field] !== undefined && req.body[field] !== null)) {
      dialin_settings = {
        // snake_case because pipecat expects this format
        From,
        To,
        call_id: callId, 
        call_domain: callDomain,
      };
      logger.debug(`Populated dialin_settings from request: ${JSON.stringify(dialin_settings)}`);
    }

    // Set up Daily room properties
    const daily_room_properties = {
      enable_dialout: dialout_settings !== undefined && dialout_settings !== null,
      exp: Math.floor(Date.now() / 1000) + (5 * 60), // 5 minutes from now
    };

    // Configure SIP if dialin settings are provided
    if (dialin_settings !== null) {
      const sip_config = {
        display_name: From,
        sip_mode: 'dial-in',
        num_endpoints: call_transfer !== null ? 2 : 1,
      };
      daily_room_properties.sip = sip_config;
    }

    // Prepare payload for {service}/start API call
    const payload = {
      createDailyRoom: true,
      dailyRoomProperties: daily_room_properties,
      body: {
        dialin_settings,
        dialout_settings,
        voicemail_detection,
        call_transfer,
      },
    };

    logger.debug(`Daily room properties: ${JSON.stringify(daily_room_properties)}`);

    // Get Daily API key and agent name from environment variables
    const pccApiKey = process.env.PIPECAT_CLOUD_API_KEY;
    const agentName = process.env.AGENT_NAME || 'my-first-agent';

    if (!pccApiKey) {
      throw new Error('PIPECAT_CLOUD_API_KEY environment variable is not set');
    }

    // Set up headers for Daily API call
    const headers = {
      'Authorization': `Bearer ${pccApiKey}`,
      'Content-Type': 'application/json',
    };

    const url = `https://api.pipecat.daily.co/v1/public/${agentName}/start`;
    logger.debug(`Making API call to Daily: ${url} ${JSON.stringify(headers)} ${JSON.stringify(payload)}`);
    
    try {
      const response = await axios.post(url, payload, { headers });
      logger.debug(`Response: ${JSON.stringify(response.data)}`);
      
      return res.status(200).json({
        status: 'success',
        data: response.data,
        room_properties: daily_room_properties,
      });
    } catch (error) {
      if (error.response) {
        // Pass through status code and error details from the Daily API
        const statusCode = error.response.status;
        const errorDetail = error.response.data || error.message;
        logger.error(`HTTP error: ${JSON.stringify(errorDetail)}`);
        return res.status(statusCode).json(errorDetail);
      } else {
        logger.error(`Request error: ${error.message}`);
        return res.status(500).json({ error: error.message });
      }
    }    
  } catch (error) {
    logger.error(`Unexpected error: ${error.message}`);
    return res.status(500).json({ error: 'Internal server error', message: error.message });
  }
}

// Configure body parser to preserve raw body text
export const config = {
  api: {
    bodyParser: {
      sizeLimit: '1mb',
    },
  },
};