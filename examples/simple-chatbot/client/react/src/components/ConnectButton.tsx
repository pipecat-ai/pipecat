import {
  usePipecatClient,
  usePipecatClientTransportState,
} from '@pipecat-ai/client-react';

export function ConnectButton() {
  const client = usePipecatClient();
  const transportState = usePipecatClientTransportState();
  const isConnected = ['connected', 'ready'].includes(transportState);

  const handleClick = async () => {
    if (!client) {
      console.error('Pipecat client is not initialized');
      return;
    }

    try {
      if (isConnected) {
        await client.disconnect();
      } else {
        await client.connect({ endpoint: 'http://localhost:7860/connect' });
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
