import { useEffect, useState, useRef } from 'react';
import { getLatestRecording } from '../services/recordingService';

// Simplified component that doesn't require roomUrl prop
export function VideoReplay() {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    console.log('VideoReplay: Starting to fetch recording');
    
    const fetchRecording = async () => {
      setLoading(true);
      setError(null);
      try {
        console.log('VideoReplay: Calling getLatestRecording()');
        // Fetch recording data using the service
        const data = await getLatestRecording(); // Directly fetch latest recording from API
        console.log('VideoReplay: Received recording data:', data);
        setVideoUrl(data.download_link);
      } catch (err) {
        console.error('Error fetching recording:', err);
        setError('Could not load recording. It may take a few minutes to process.');
      } finally {
        setLoading(false);
      }
    };

    // Add a delay before fetching to allow server processing time
    const timer = setTimeout(() => {
      fetchRecording();
    }, 3000); // 3 second delay

    return () => clearTimeout(timer);
  }, [retryCount]); // Run on mount and when retry is triggered

  // Handle video errors and implement progressive downloading
  useEffect(() => {
    const videoElement = videoRef.current;
    if (!videoElement) return;

    const handleError = () => {
      console.error('Video playback error');
      setError('Error playing the video. The recording may not be ready yet.');
    };

    const handleWaiting = () => {
      console.log('Video is waiting for more data');
    };

    const handleProgress = () => {
      if (videoElement.buffered.length > 0) {
        const bufferedEnd = videoElement.buffered.end(0);
        const duration = videoElement.duration;
        console.log(`Loaded ${Math.round((bufferedEnd / duration) * 100)}% of the video`);
      }
    };

    videoElement.addEventListener('error', handleError);
    videoElement.addEventListener('waiting', handleWaiting);
    videoElement.addEventListener('progress', handleProgress);

    return () => {
      videoElement.removeEventListener('error', handleError);
      videoElement.removeEventListener('waiting', handleWaiting);
      videoElement.removeEventListener('progress', handleProgress);
    };
  }, [videoUrl]);

  const retryFetch = () => {
    setRetryCount(prevCount => prevCount + 1);
  };

  if (loading) {
    return (
      <div className="video-replay-loading">
        <p>Loading session recording...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="video-replay-error">
        <p>{error}</p>
        <button onClick={retryFetch}>Try Again</button>
      </div>
    );
  }

  if (!videoUrl) {
    console.log('VideoReplay: No video URL available yet');
    return null;
  }

  console.log('VideoReplay: Rendering video player with URL:', videoUrl);
  return (
    <div className="video-replay">
      <video 
        ref={videoRef}
        src={videoUrl} 
        controls 
        autoPlay 
        className="replay-video"
        preload="auto"
      />
    </div>
  );
} 