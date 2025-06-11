import {
  PipecatClientAudio,
  PipecatClientVideo,
  usePipecatClientTransportState,
} from '@pipecat-ai/client-react';
import { PipecatProvider } from './providers/PipecatProvider';
import { ConnectButton } from './components/ConnectButton';
import { StatusDisplay } from './components/StatusDisplay';
import { DebugDisplay } from './components/DebugDisplay';
import './App.css';

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

function AppContent() {
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

function App() {
  return (
    <PipecatProvider>
      <AppContent />
    </PipecatProvider>
  );
}

export default App;
