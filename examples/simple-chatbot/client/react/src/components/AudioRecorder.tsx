// src/components/AudioRecorder.tsx
import { useEffect, useRef } from 'react';
import { useRTVIClientMediaTrack, useRTVIClientTransportState } from '@pipecat-ai/client-react';



interface AudioRecorderProps {
    onStopRecording: (url: string, startTime: number) => void;
  }
/**
 * AudioRecorder mixes local and bot audio tracks,
 * records them, and provides a "Stop Call" button
 * to end recording and render a playback control.
 */
export function AudioRecorder({ onStopRecording }: AudioRecorderProps) {
    const transport = useRTVIClientTransportState();
    const localTrack = useRTVIClientMediaTrack('audio', 'local');
    const botTrack   = useRTVIClientMediaTrack('audio', 'bot');
    const startTime   = useRef<number>(0);
    
    const recorderRef = useRef<MediaRecorder>();
    const chunksRef   = useRef<Blob[]>([]);
    const audioCtxRef = useRef<AudioContext>();
    const destRef     = useRef<MediaStreamAudioDestinationNode>();


    useEffect(() => {
        if (!localTrack || !botTrack) return;

        // Create AudioContext and destination for mixing
        const audioCtx = new AudioContext();
        const dest     = audioCtx.createMediaStreamDestination();
        audioCtxRef.current = audioCtx;
        destRef.current     = dest;

        const localSrc = audioCtx.createMediaStreamSource(new MediaStream([localTrack]));
        localSrc.connect(dest);
        const botSrc   = audioCtx.createMediaStreamSource(new MediaStream([botTrack]));
        botSrc.connect(dest);

        const recorder = new MediaRecorder(dest.stream, { mimeType: 'audio/webm; codecs=opus' });
        chunksRef.current = [];

        recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
        };

        recorder.onstop = () => {
            const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
            const url  = URL.createObjectURL(blob);
            onStopRecording(url, startTime.current);
        };

        recorderRef.current = recorder;
        recorder.start(); // start capturing immediately

        // Cleanup on unmount
        return () => {
            recorder.stop();
            audioCtx.close();
        };
    }, [localTrack, botTrack]);

    useEffect(() => {
        const recorder = recorderRef.current;
        if (!recorder) return;
    
        if (transport === 'connected' && recorder.state === 'inactive') {
          recorder.start();
          startTime.current = performance.now();
        }
    
        if (transport === 'disconnected' && recorder.state === 'recording') {
          recorder.stop();
        }
      }, [transport]);




    return null;
}
