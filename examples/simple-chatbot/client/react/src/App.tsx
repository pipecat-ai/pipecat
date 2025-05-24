import { useState, useEffect, useCallback } from 'react';
import {
  RTVIClientAudio,
  RTVIClientVideo,
  useRTVIClientTransportState,
  useRTVIClientEvent,
  useRTVIClient,
} from '@pipecat-ai/client-react';
import {
  RTVIEvent,
} from '@pipecat-ai/client-js';
import { RTVIProvider } from './providers/RTVIProvider';
import { ConnectButton } from './components/ConnectButton';
import { StatusDisplay } from './components/StatusDisplay';
import { DebugDisplay } from './components/DebugDisplay';
import { ClientMetricsDisplay } from './components/ClientMetricsDisplay';
import CallRecording from './components/CallRecording';
import { fetchLatestSession } from './services/recordingService';
import './App.css';
import { useLatencyMetrics } from './hooks/useLatencyMetrics';

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
  const client = useRTVIClient();

  // Step 3 & 4: Fetch session ID when bot disconnects (let CallRecording handle the retry logic)
  const [sessionId, setSessionId] = useState<string>('');
  const { logEvent, computedMetrics, sendMetricsOnDisconnect, startNewSession, endSession } = useLatencyMetrics(sessionId);

  const fetchSessionData = useCallback(async () => {    
    setRecordingState(prev => ({
      ...prev,
      isLoadingRecording: true,
      recordingError: null
    }));

    try {
      const sessionResponse = await fetchLatestSession();
      
      if (sessionResponse && sessionResponse.session_id) {
        setRecordingState(prev => ({
          ...prev,
          sessionId: sessionResponse?.session_id || null,
          hasRecording: true,
          isLoadingRecording: false,
          recordingUrl: null,
          recordingError: null
        }));
        setSessionId(sessionResponse.session_id);
        
        // Send metrics after session ID is set
        console.log("🚀 Session ID fetched, now sending metrics");
        setTimeout(() => {
          sendMetricsOnDisconnect();
        }, 100);
      } else {
        const errorMsg = sessionResponse?.error || 'No session ID returned';
        
        setRecordingState(prev => ({
          ...prev,
          recordingError: errorMsg,
          isLoadingRecording: false
        }));
      }
    } catch (error) {
      setRecordingState(prev => ({
        ...prev,
        recordingError: `API Error: ${error}`,
        isLoadingRecording: false
      }));
    }
  }, [sendMetricsOnDisconnect]);

  // Step 2: Detect when bot disconnects
  useRTVIClientEvent(
    RTVIEvent.BotDisconnected,
    useCallback(() => {
      console.log("🔌 Bot disconnected");
      setBotDisconnected(true);
      endSession();
      fetchSessionData();
    }, [fetchSessionData, endSession])
  );

  // Step 2: Track when bot connects (to know when a session was active)
  useRTVIClientEvent(
    RTVIEvent.BotConnected,
    useCallback(() => {
      setWasConnected(true);
      setBotDisconnected(false);
      
      setRecordingState({
        hasRecording: false,
        recordingUrl: null,
        isLoadingRecording: false,
        recordingError: null,
        sessionId: null
      });
      
      startNewSession();
    }, [startNewSession])
  );

  // Also detect disconnection via transport state as backup
  useEffect(() => {
    if (transportState === 'disconnected' && wasConnected && !botDisconnected) {
      console.log("🔌 Transport disconnected");
      setBotDisconnected(true);
      fetchSessionData(); // This will now send metrics after getting session ID
    }
  }, [transportState, wasConnected, botDisconnected, fetchSessionData]);

  // Add latency event tracking to RTVI events
  useEffect(() => {
    if (!client) return;

    const handleUserStartedSpeaking = () => {
      logEvent('user_started_speaking');
    };

    const handleUserStoppedSpeaking = () => {
      logEvent('user_stopped_speaking');
    };

    const handleBotStartedSpeaking = () => {
      logEvent('bot_started_speaking');
    };

    const handleBotStoppedSpeaking = () => {
      logEvent('bot_stopped_speaking');
    };

    const handleParticipantConnected = (participant: any) => {
      console.log("Participant connected:", participant);
      if (participant.session_id) {
        // Set session ID and let CallRecording component handle the retry logic for full recording
        setRecordingState(prev => ({
          ...prev,
          sessionId: participant.session_id,
          hasRecording: true,
          isLoadingRecording: false,
          recordingUrl: null,
          recordingError: null
        }));
        setSessionId(participant.session_id);
      }
    };

    const handleParticipantLeft = () => {
      console.log("🔌 Participant left");
      fetchSessionData(); // This will send metrics after getting session ID
    };

    // Add connection state handlers
    const handleDisconnected = () => {
      console.log("🔌 Client disconnected");
      fetchSessionData(); // This will send metrics after getting session ID
    };

    const handleError = (error: unknown) => {
      console.log("❌ Client error", error);
      fetchSessionData(); // This will send metrics after getting session ID
    };

    // Register RTVI event listeners
    client.on("userStartedSpeaking", handleUserStartedSpeaking);
    client.on("userStoppedSpeaking", handleUserStoppedSpeaking);
    client.on("botStartedSpeaking", handleBotStartedSpeaking);
    client.on("botStoppedSpeaking", handleBotStoppedSpeaking);
    client.on("participantConnected", handleParticipantConnected);
    client.on("participantLeft", handleParticipantLeft);
    client.on("disconnected", handleDisconnected);
    client.on("error", handleError);

    return () => {
      client.off("userStartedSpeaking", handleUserStartedSpeaking);
      client.off("userStoppedSpeaking", handleUserStoppedSpeaking);
      client.off("botStartedSpeaking", handleBotStartedSpeaking);
      client.off("botStoppedSpeaking", handleBotStoppedSpeaking);
      client.off("participantConnected", handleParticipantConnected);
      client.off("participantLeft", handleParticipantLeft);
      client.off("disconnected", handleDisconnected);
      client.off("error", handleError);
    };
  }, [client, logEvent]);

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

      <ClientMetricsDisplay 
        computedMetrics={computedMetrics} 
        showMetrics={true}
      />
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