import {
  useRTVIClient,
  useRTVIClientTransportState,
} from '@pipecat-ai/client-react';
import { RTVIEvent, Participant } from '@pipecat-ai/client-js';

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
        // Manually disconnect
        await client.disconnect(); // Simple disconnect without storing room URL
        
        // Create a mock participant object matching the required interface
        const mockParticipant: Participant = {
          id: 'bot',
          name: 'bot',
          local: false
        }; // Required param for the event
        
        // Trigger BotDisconnected event to show recording
        client.emit(RTVIEvent.BotDisconnected, mockParticipant); // Directly emit event without delay
        console.log('Manually triggered BotDisconnected event after user disconnect');
      } else {
        await client.connect();
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
