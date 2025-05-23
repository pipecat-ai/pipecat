import { useState, useEffect, useCallback } from 'react';
import {
  RTVIClientAudio,
  RTVIClientVideo,
  useRTVIClientTransportState,
  useRTVIClientEvent,
} from '@pipecat-ai/client-react';
import {
  RTVIEvent,
  Participant,
} from '@pipecat-ai/client-js';
import { RTVIProvider } from './providers/RTVIProvider';
import { ConnectButton } from './components/ConnectButton';
import { StatusDisplay } from './components/StatusDisplay';
import { DebugDisplay } from './components/DebugDisplay';
import CallRecording from './components/CallRecording';
import { fetchLatestSession } from './services/recordingService';
import './App.css';

// Recording state interface
interface RecordingState {
  hasRecording: boolean;
  recordingUrl: string | null;
  isLoadingRecording: boolean;
  recordingError: string | null;
  sessionId: string | null;
}

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
  // Step 1: Add Recording State Management
  const [recordingState, setRecordingState] = useState<RecordingState>({
    hasRecording: false,
    recordingUrl: null,
    isLoadingRecording: false,
    recordingError: null,
    sessionId: null
  });

  // Step 2: Add Bot Disconnection Detection
  const [botDisconnected, setBotDisconnected] = useState<boolean>(false);
  const [wasConnected, setWasConnected] = useState<boolean>(false);

  const transportState = useRTVIClientTransportState();

  // Step 3 & 4: Fetch session ID when bot disconnects (let CallRecording handle the retry logic)
  const fetchSessionData = useCallback(async () => {
    console.log('🔍 Step 3: Fetching latest session data...');
    
    setRecordingState(prev => ({
      ...prev,
      isLoadingRecording: true,
      recordingError: null
    }));

    try {
      // Step 3: Get session ID
      const sessionResponse = await fetchLatestSession();
      
      if (sessionResponse && sessionResponse.session_id) {
        console.log('✅ Step 3: Session ID fetched:', sessionResponse.session_id);
        
        // Set session ID and let CallRecording component handle the retry logic for full recording
        setRecordingState(prev => ({
          ...prev,
          sessionId: sessionResponse?.session_id || null,
          hasRecording: true,
          isLoadingRecording: false,
          recordingUrl: null,
          recordingError: null
        }));
      } else {
        const errorMsg = sessionResponse?.error || 'No session ID returned';
        console.log('❌ Step 3: Failed to get session ID:', errorMsg);
        
        setRecordingState(prev => ({
          ...prev,
          recordingError: errorMsg,
          isLoadingRecording: false
        }));
      }
    } catch (error) {
      console.error('❌ Step 3: Error fetching session:', error);
      
      setRecordingState(prev => ({
        ...prev,
        recordingError: `API Error: ${error}`,
        isLoadingRecording: false
      }));
    }
  }, []);

  // Step 2: Detect when bot disconnects
  useRTVIClientEvent(
    RTVIEvent.BotDisconnected,
    useCallback((participant?: Participant) => {
      console.log('🤖❌ Bot Disconnected Event:', participant);
      setBotDisconnected(true);
      
      // Step 3: Fetch session ID (CallRecording will handle waiting for full recording)
      fetchSessionData();
    }, [fetchSessionData])
  );

  // Step 2: Track when bot connects (to know when a session was active)
  useRTVIClientEvent(
    RTVIEvent.BotConnected,
    useCallback((participant?: Participant) => {
      console.log('🤖✅ Bot Connected Event:', participant);
      setWasConnected(true);
      setBotDisconnected(false);
      
      // Reset recording state when starting new session
      setRecordingState({
        hasRecording: false,
        recordingUrl: null,
        isLoadingRecording: false,
        recordingError: null,
        sessionId: null
      });
    }, [])
  );

  // Log state changes for testing
  useEffect(() => {
    console.log('📊 Recording State Updated:', recordingState);
  }, [recordingState]);

  // Log transport state changes
  useEffect(() => {
    console.log('🔌 Transport State Updated:', transportState);
    
    // Also detect disconnection via transport state as backup
    if (transportState === 'disconnected' && wasConnected && !botDisconnected) {
      console.log('🔌❌ Transport Disconnected (was previously connected)');
      setBotDisconnected(true);
      fetchSessionData();
    }
  }, [transportState, wasConnected, botDisconnected, fetchSessionData]);

  useEffect(() => {
    console.log('🤖 Bot Disconnection State:', { botDisconnected, wasConnected, transportState });
  }, [botDisconnected, wasConnected, transportState]);

  return (
    <div className="app">
      <div className="status-bar">
        <StatusDisplay />
        <ConnectButton />
      </div>

      <div className="main-content">
        <BotVideo />
        
        {/* Step 4: Recording section - Clean UI with single title */}
        {recordingState.hasRecording && recordingState.sessionId && (
          <div style={{ 
            marginTop: '20px',
            padding: '15px',
            border: '2px solid #28a745',
            borderRadius: '8px',
            backgroundColor: '#d4edda'
          }}>
            <h3>🎵 Full Call Recording</h3>
            <CallRecording sessionId={recordingState.sessionId} />
          </div>
        )}

        {recordingState.isLoadingRecording && (
          <div style={{ 
            marginTop: '20px',
            padding: '15px',
            border: '2px solid #ffc107',
            borderRadius: '8px',
            backgroundColor: '#fff3cd'
          }}>
            <h3>⏳ Loading Recording...</h3>
            <p>Please wait while we fetch your recording.</p>
          </div>
        )}

        {recordingState.recordingError && (
          <div style={{ 
            marginTop: '20px',
            padding: '15px',
            border: '2px solid #dc3545',
            borderRadius: '8px',
            backgroundColor: '#f8d7da'
          }}>
            <h3>❌ Recording Error</h3>
            <p>Error: {recordingState.recordingError}</p>
          </div>
        )}
      </div>

      <DebugDisplay />
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