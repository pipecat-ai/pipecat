import { logger } from '../../lib/utils';

export default function handler(req, res) {
  logger.info('Received request to /api');
  res.status(200).json({ message: 'Hello, World! from ᓚᘏᗢ' });
}
