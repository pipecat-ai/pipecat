// src/App.tsx
import { useState } from 'react';
import {
  RTVIClientAudio,
  RTVIClientVideo,
  useRTVIClientTransportState,
} from '@pipecat-ai/client-react';
import { RTVIProvider } from './providers/RTVIProvider';
import { ConnectButton } from './components/ConnectButton';
import { StatusDisplay } from './components/StatusDisplay';
import { TranscriptDisplay } from './components/TranscriptDisplay';
import { AudioRecorder } from './components/AudioRecorder';
import { AudioAnalysis } from './components/AudioAnalysis';
import { LatencyTracker } from './components/LatencyTracker';

import { Interval } from './components/LatencyTracker'
import './App.css';

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

function AppContent() {

  const [mixedUrl, setMixedUrl] = useState<string | null>(null);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [latencies, setLatencies] = useState<Interval[]>([]);
  
  return (
    <div className="app">
      <div className="status-bar">
        <StatusDisplay />
        <ConnectButton />
      </div>

      <div className="main-content">
        <BotVideo />
        <TranscriptDisplay />
      </div>

      <RTVIClientAudio />

      <AudioRecorder onStopRecording={(url, startTime) => {
        setMixedUrl(url)
        setStartTime(startTime)
      }} />

      <LatencyTracker onLatency={({start, end} ) => {
        setLatencies((prev) => [...prev, { start, end }])
      }} 
      />

  

      {mixedUrl && (
        <div style={{ marginTop: '1rem', width: '100%' }}>
          <AudioAnalysis
            playbackUrl={mixedUrl}
            startTime={startTime!}
            latencies={latencies}
          />

          {/* Download button */}
          <div style={{ marginTop: '0.5rem' }}>
            <a href={mixedUrl} download="call-recording.webm">
              <button type="button">Download Recording</button>
            </a>
          </div>
        </div>
      )}


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
