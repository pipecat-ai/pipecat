'use client';

import {
  PipecatClientAudio,
  PipecatClientVideo,
  usePipecatClientTransportState,
} from '@pipecat-ai/client-react';
import { ConnectButton } from '../components/ConnectButton';
import { StatusDisplay } from '../components/StatusDisplay';
import { DebugDisplay } from '../components/DebugDisplay';

function BotVideo() {
  const transportState = usePipecatClientTransportState();
  const isConnected = transportState !== 'disconnected';

  return (
    <div className="bot-container">
      <div className="video-container">
        {isConnected && <PipecatClientVideo participant="bot" fit="cover" />}
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
      <PipecatClientAudio />
    </div>
  );
}
