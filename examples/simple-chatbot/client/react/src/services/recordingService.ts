/**
 * Service for fetching call recordings and session data
 */

const API_BASE_URL = 'http://localhost:7860';

interface SessionResponse {
  session_id?: string;
  error?: string;
}

export interface RecordingData {
  user?: string[];
  bot?: string[];
  full?: string[];
  error?: string;
  [key: string]: string[] | string | undefined;
}

// Fetch the latest session ID
export const fetchLatestSession = async (): Promise<SessionResponse | null> => {
  try {
    console.log('Fetching latest session...');
    const response = await fetch(`${API_BASE_URL}/api/latest-session`);
    
    // Log raw response for debugging
    const text = await response.text();
    console.log('Raw API response:', text);
    
    // Try to parse as JSON
    try {
      const data = JSON.parse(text);
      console.log('Parsed session data:', data);
      return data;
    } catch (parseError) {
      console.error('Failed to parse response as JSON:', parseError);
      return null;
    }
  } catch (error) {
    console.error('Error fetching session:', error);
    return null;
  }
};

// Fetch recordings for a specific session
export const fetchSessionRecordings = async (sessionId: string): Promise<RecordingData | null> => {
  if (!sessionId) return null;
  
  try {
    console.log(`Fetching recordings for session: ${sessionId}`);
    const response = await fetch(`${API_BASE_URL}/api/recordings/${sessionId}`);
    
    // Log raw response for debugging
    const text = await response.text();
    console.log('Raw recordings API response:', text);
    
    // Try to parse as JSON
    try {
      const data = JSON.parse(text);
      console.log('Parsed recordings data:', data);
      return data;
    } catch (parseError) {
      console.error('Failed to parse recordings response as JSON:', parseError);
      return null;
    }
  } catch (error) {
    console.error('Error fetching recordings:', error);
    return null;
  }
};

// Poll for session ID with retry logic
export const pollForSessionId = async (
  callback: (sessionId: string) => void, 
  maxAttempts: number = 10, 
  interval: number = 1000
): Promise<boolean> => {
  let attempts = 0;
  
  const poll = async (): Promise<boolean> => {
    try {
      const session = await fetchLatestSession();
      if (session && session.session_id) {
        callback(session.session_id);
        return true;
      }
    } catch (error) {
      console.error('Error polling for session ID:', error);
    }
    
    attempts++;
    if (attempts >= maxAttempts) {
      console.error('Max polling attempts reached');
      return false;
    }
    
    return new Promise(resolve => {
      setTimeout(async () => {
        resolve(await poll());
      }, interval);
    });
  };
  
  return await poll();
};