/**
 * Service for handling recording API calls
 */

interface RecordingData {
  recording_url: string;
  start_time: string;
  duration: number;
  room_name: string;
}

interface LatestRecordingData {
  download_link: string; // Simple interface that only needs the download URL
}

/**
 * Fetch the latest recording regardless of room
 * @returns Promise with the download link for the latest recording
 */
export async function getLatestRecording(): Promise<LatestRecordingData> {
  console.log('recordingService: Fetching latest recording from endpoint');
  try {
    const response = await fetch('http://localhost:7860/latest_recording/'); // Direct API call to the simplified endpoint
    
    console.log('recordingService: Response status:', response.status);
    console.log('recordingService: Response headers:', [...response.headers.entries()]);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('recordingService: Error response body:', errorText);
      throw new Error(`Failed to fetch latest recording: ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('recordingService: Received data:', data);
    
    return data; // Return the data with download_link directly
  } catch (error) {
    console.error('recordingService: Exception during fetch:', error);
    throw error;
  }
} 