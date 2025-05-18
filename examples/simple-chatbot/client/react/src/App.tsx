import {
  RTVIClientAudio,
  RTVIClientVideo,
  useRTVIClientEvent,
  useRTVIClientTransportState,
} from '@pipecat-ai/client-react';
import { RTVIEvent } from '@pipecat-ai/client-js';
import { RTVIProvider } from './providers/RTVIProvider';
import { ConnectButton } from './components/ConnectButton';
import { StatusDisplay } from './components/StatusDisplay';
import { VideoReplay } from './components/VideoReplay';
import { TranscriptPanel } from './components/TranscriptPanel';
import { useState, useEffect } from 'react';
import './App.css';

function BotVideo() {
  const transportState = useRTVIClientTransportState();
  const isConnected = transportState !== 'disconnected';
  const [showReplay, setShowReplay] = useState(false);
  const [wasConnected, setWasConnected] = useState(false); // Track if user has connected at least once this session

  // Track if user has been connected during this session
  useEffect(() => {
    if (isConnected) {
      setWasConnected(true); // Set to true once connected, never resets during session
    }
  }, [isConnected]);

  // Show replay when transport state changes to disconnected, but only if previously connected
  useEffect(() => {
    if (!isConnected && wasConnected) {
      console.log('BotVideo: Was connected and now disconnected, showing replay');
      setShowReplay(true); // Only show replay after being connected and then disconnected
    } else if (isConnected) {
      // Reset replay when connecting
      setShowReplay(false); // Hide replay when connecting to show live video instead
    }
  }, [isConnected, wasConnected]);

  // Listen for bot disconnection to show replay
  useRTVIClientEvent(
    RTVIEvent.BotDisconnected,
    () => {
      // Only show replay if we were previously connected
      if (wasConnected) {
        console.log('BotVideo: BotDisconnected event received, showing replay');
        setShowReplay(true); // Show replay when bot disconnects, but only if we were connected first
      }
    }
  );

  console.log('BotVideo: Rendering with:', { isConnected, wasConnected, showReplay });
  return (
    <div className="bot-container">
      <div className="video-container">
        {isConnected && <RTVIClientVideo participant="bot" fit="cover" />} {/* Show live video when connected */}
        {!isConnected && showReplay && <VideoReplay />} {/* Show replay only when disconnected and replay should be shown */}
      </div>
    </div>
  );
}

function AppContent() {
  return (
    <div className="app">
      <div className="status-bar">
        <StatusDisplay />
        <ConnectButton />
      </div>

      <div className="main-content">
        <div className="content-layout">
          <BotVideo />
          <TranscriptPanel />
        </div>
      </div>

      <RTVIClientAudio />
    </div>
  );
}

function App() {
  return (
    <RTVIProvider>
      <AppContent />
    </RTVIProvider>
  );
}

export default App;
