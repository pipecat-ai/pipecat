// components/VoiceAssistantFixed.tsx
import { useEffect, useState } from 'react';
import ProjectsModal from './ProjectsModal';
import { TransportState } from '@pipecat-ai/client-js';
import {
  useRTVIClient,
  useRTVIClientTransportState,
  useRTVIClientMicControl,
  VoiceVisualizer,
  RTVIClientMicToggle,
} from '@pipecat-ai/client-react'; // using built-in visualizer

const ICE_SERVERS = [
  { urls: 'stun:96.242.77.142:3478' },
  {
    urls: ['turn:96.242.77.142:3478?transport=udp'],
    username: 'alex',
    credential: 'supersecret',
  },
];

export default function VoiceAssistant() {
  const client = useRTVIClient();
  const [modalOpen, setModalOpen] = useState(false);
  const transportState = useRTVIClientTransportState();
  const { enableMic, isMicEnabled } = useRTVIClientMicControl();

  const isConnecting =
    transportState === 'connecting' ||
    transportState === 'initializing' ||
    transportState === 'authenticating';
  const isConnected =
    transportState === 'connected' || transportState === 'ready';

  const connect = async () => {
    if (!client || isConnecting || isConnected) return;
    try {
      await client.initDevices();
      await client.connect();
    } catch (err) {
      console.error('[VoiceAssistant] connect failed', err);
    }
  };

  useEffect(() => {
    return () => {
      client?.disconnect();
    };
  }, [client]);

  return (
    <div className="w-full min-h-screen flex flex-col items-center justify-center bg-gray-900 text-white gap-6 p-4">
      <h1 className="text-2xl font-semibold">Alex Covo Voice Assistant</h1>

      {/* Built-in visualizer for the local participant */}
      <VoiceVisualizer participantType="local" />

      <p>Status: {transportState}</p>

      {!isConnected ? (
        <button
          onClick={connect}
          disabled={isConnecting}
          className="px-6 py-3 rounded-full bg-blue-600 disabled:bg-gray-500"
        >
          {isConnecting ? 'Connecting…' : 'Start'}
        </button>
      ) : (
        <button
          onClick={() => client?.disconnect()}
          className="px-6 py-3 rounded-full bg-red-600"
        >
          Disconnect
        </button>
      )}

      {isConnected && (
        <RTVIClientMicToggle>
          {({ isMicEnabled, onClick }) => (
            <button
              onClick={onClick}
              className="mt-2 px-4 py-2 rounded bg-gray-700"
            >
              {isMicEnabled ? 'Mute Mic' : 'Unmute Mic'}
            </button>
          )}
        </RTVIClientMicToggle>
      )}
      <button
        className="mt-4 px-4 py-2 bg-blue-700 rounded"
        onClick={() => setModalOpen(true)}
      >
        Browse Projects
      </button>

      <ProjectsModal
        clientId={'default'}
        open={modalOpen}
        onClose={() => setModalOpen(false)}
      />
    </div>
  );
}
