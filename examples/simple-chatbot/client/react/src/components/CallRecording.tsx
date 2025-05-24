import React, { useState, useEffect } from 'react';
import { fetchSessionRecordings, RecordingData } from '../services/recordingService';

interface CallRecordingProps {
  sessionId: string;
  autoStart?: boolean;  // Optional prop to auto-start recording
}

// Simple styling
const styles = {
  container: {
    marginTop: '10px',
  },
  audioPlayer: {
    width: '100%',
    marginBottom: '10px',
  },
  loadingText: {
    fontStyle: 'italic',
    color: '#666',
  },
  errorText: {
    color: 'red',
  }
};

const API_BASE_URL = 'http://localhost:7860';

const CallRecording: React.FC<CallRecordingProps> = ({ sessionId }) => {
  const [recordings, setRecordings] = useState<RecordingData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState<number>(0);

  useEffect(() => {
    if (!sessionId) {
      setError('No session ID provided');
      setLoading(false);
      return;
    }

    const getRecordings = async () => {
      try {
        const data = await fetchSessionRecordings(sessionId);
        
        if (data && !data.error) {
          // Check if full recording is available
          const hasFullRecording = data.full && data.full.length > 0;
          
          if (hasFullRecording) {
            setRecordings(data);
            setLoading(false);
          } else {
            // Retry up to 10 times with increasing delays
            if (retryCount < 10) {
              setTimeout(() => {
                setRetryCount(prev => prev + 1);
              }, Math.min(2000 + (retryCount * 1000), 10000)); // 2s, 3s, 4s, ... up to 10s
            } else {
              setError('Full recording not available after multiple attempts');
              setLoading(false);
            }
          }
        } else {
          const errorMsg = data?.error || 'Failed to load recordings';
          setError(errorMsg);
          setLoading(false);
        }
      } catch (err) {
        setError('Error loading recordings');
        setLoading(false);
      }
    };

    getRecordings();
  }, [sessionId, retryCount]);

  if (loading) {
    return (
      <div style={styles.loadingText}>
        {retryCount === 0 
          ? 'Loading recordings...' 
          : `Waiting for full recording to be ready... (attempt ${retryCount + 1}/10)`
        }
      </div>
    );
  }

  if (error) {
    return <div style={styles.errorText}>Error: {error}</div>;
  }

  if (!recordings || !recordings.full || recordings.full.length === 0) {
    return <div>No full conversation recording available for this session.</div>;
  }

  // Extract filename and construct full URL
  const recordingPath = recordings.full[0];
  const audioUrl = `${API_BASE_URL}/${recordingPath}`;

  // Clean UI - just the audio player
  return (
    <div style={styles.container}>
      <audio 
        controls 
        style={styles.audioPlayer} 
        src={audioUrl}
        preload="metadata"
      />
    </div>
  );
};

export default CallRecording;