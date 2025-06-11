import { useEffect, useCallback } from 'react';
import {
  usePipecatClient,
  usePipecatClientTransportState,
} from '@pipecat-ai/client-react';
import { CONNECTION_STATES } from '@/constants/gameConstants';
import { useConfigurationSettings } from '@/contexts/Configuration';

// Get the API base URL from environment variables
// Default to "/api" if not specified
// "/api" is the default for Next.js API routes and used
// for the Pipecat Cloud deployed agent
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || '/api';

console.log('Using API base URL:', API_BASE_URL);

export function useConnectionState(
  onConnected?: () => void,
  onDisconnected?: () => void
) {
  const client = usePipecatClient();
  const transportState = usePipecatClientTransportState();
  const config = useConfigurationSettings();

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
        await client.connect({
          endpoint: `${API_BASE_URL}/connect`,
          requestData: {
            personality: config.personality,
          },
        });
      }
    } catch (error) {
      console.error('Connection error:', error);
    }
  }, [client, config, isConnected]);

  return {
    isConnected,
    isConnecting,
    isDisconnecting,
    toggleConnection,
    transportState,
    client, // Expose the client for direct access when needed
  };
}
