'use client';

import {
  RTVIClientAudio,
  RTVIClientVideo,
  useRTVIClientTransportState,
} from '@pipecat-ai/client-react';
import { ConnectButton } from '../components/ConnectButton';
import { StatusDisplay } from '../components/StatusDisplay';
import { DebugDisplay } from '../components/DebugDisplay';

function BotVideo() {
  const transportState = useRTVIClientTransportState();
  const isConnected = transportState !== 'disconnected';

  return (
    <div className="bot-container">
      <div className="video-container">
        {isConnected && <RTVIClientVideo participant="bot" fit="cover" />}
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <div className="app">
      <div className="status-bar">
        <StatusDisplay />
        <ConnectButton />
      </div>

      <div className="main-content">
        <BotVideo />
      </div>

      <DebugDisplay />
      <RTVIClientAudio />
    </div>
  );
}
