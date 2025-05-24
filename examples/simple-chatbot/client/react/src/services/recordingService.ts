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
    const response = await fetch(`${API_BASE_URL}/api/latest-session`);
    const text = await response.text();
    
    try {
      const data = JSON.parse(text);
      return data;
    } catch (parseError) {
      return null;
    }
  } catch (error) {
    return null;
  }
};

// Fetch recordings for a specific session
export const fetchSessionRecordings = async (sessionId: string): Promise<RecordingData | null> => {
  if (!sessionId) return null;
  
  try {
    const response = await fetch(`${API_BASE_URL}/api/recordings/${sessionId}`);
    const text = await response.text();
    
    try {
      const data = JSON.parse(text);
      return data;
    } catch (parseError) {
      return null;
    }
  } catch (error) {
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
      // Error handling
    }
    
    attempts++;
    if (attempts >= maxAttempts) {
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