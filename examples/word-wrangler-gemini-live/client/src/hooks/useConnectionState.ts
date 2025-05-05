import { useEffect, useCallback } from 'react';
import {
  useRTVIClient,
  useRTVIClientTransportState,
} from '@pipecat-ai/client-react';
import { CONNECTION_STATES } from '@/constants/gameConstants';

export function useConnectionState(
  onConnected?: () => void,
  onDisconnected?: () => void
) {
  const client = useRTVIClient();
  const transportState = useRTVIClientTransportState();

  const isConnected = CONNECTION_STATES.ACTIVE.includes(transportState);
  const isConnecting = CONNECTION_STATES.CONNECTING.includes(transportState);
  const isDisconnecting =
    CONNECTION_STATES.DISCONNECTING.includes(transportState);

  // Handle connection changes
  useEffect(() => {
    if (isConnected && onConnected) {
      onConnected();
    }
    if (!isConnected && !isConnecting && onDisconnected) {
      onDisconnected();
    }
  }, [isConnected, isConnecting, onConnected, onDisconnected]);

  // Toggle connection state
  const toggleConnection = useCallback(async () => {
    if (!client) return;

    try {
      if (isConnected) {
        await client.disconnect();
      } else {
        await client.connect();
      }
    } catch (error) {
      console.error('Connection error:', error);
    }
  }, [client, isConnected]);

  return {
    isConnected,
    isConnecting,
    isDisconnecting,
    toggleConnection,
    transportState,
    client, // Expose the client for direct access when needed
  };
}
