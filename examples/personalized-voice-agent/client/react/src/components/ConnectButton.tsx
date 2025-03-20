import {
  useRTVIClient,
  useRTVIClientTransportState,
} from '@pipecat-ai/client-react';
import { InteractiveHoverButton } from "@/components/magicui/interactive-hover-button";
 

export function ConnectButton() {
  const client = useRTVIClient();
  const transportState = useRTVIClientTransportState();
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
        await client.connect();
      }
    } catch (error) {
      console.error('Connection error:', error);
    }
  };

  return (
      <InteractiveHoverButton
        onClick={handleClick}
        disabled={
          !client || ['connecting', 'disconnecting'].includes(transportState)
        }>
        {!client || ['connecting', 'disconnecting'].includes(transportState) ? 'Connecting...' : isConnected ? 'Disconnect' : 'Connect'}
      </InteractiveHoverButton>
  );
}
