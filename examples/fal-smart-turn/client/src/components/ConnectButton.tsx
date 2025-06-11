import {
  usePipecatClient,
  usePipecatClientTransportState,
} from '@pipecat-ai/client-react';

// Get the API base URL from environment variables
// Default to "/api" if not specified
// "/api" is the default for Next.js API routes and used
// for the Pipecat Cloud deployed agent
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || '/api';

export function ConnectButton() {
  const client = usePipecatClient();
  const transportState = usePipecatClientTransportState();
  const isConnected = ['connected', 'ready'].includes(transportState);

  const handleClick = async () => {
    if (!client) {
      console.error('RTVI client is not initialized');
      return;
    }

    try {
      if (isConnected) {
        await client.disconnect();
      } else {
        await client.connect({
          endpoint: `${API_BASE_URL}/connect`,
          requestData: { foo: 'bar' },
        });
      }
    } catch (error) {
      console.error('Connection error:', error);
    }
  };

  return (
    <div className="controls">
      <button
        className={isConnected ? 'disconnect-btn' : 'connect-btn'}
        onClick={handleClick}
        disabled={
          !client || ['connecting', 'disconnecting'].includes(transportState)
        }>
        {isConnected ? 'Disconnect' : 'Connect'}
      </button>
    </div>
  );
}
