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
  const transport   = useRTVIClientTransportState();
  const localTrack  = useRTVIClientMediaTrack('audio','local');
  const botTrack    = useRTVIClientMediaTrack('audio','bot');
  const startTime   = useRef<number>(0);

  const recorderRef = useRef<MediaRecorder>();
  const chunksRef   = useRef<Blob[]>([]);
  const audioCtxRef = useRef<AudioContext>();
  const destRef     = useRef<MediaStreamAudioDestinationNode>();
  // keep track of last URL so we can revoke()
  const lastUrlRef  = useRef<string|null>(null);

  // only set up the recorder once, when both tracks exist
  useEffect(() => {
    if (!localTrack || !botTrack) return;

    const audioCtx = new AudioContext();
    const dest     = audioCtx.createMediaStreamDestination();
    audioCtxRef.current = audioCtx;
    destRef.current     = dest;

    const localSrc = audioCtx.createMediaStreamSource(new MediaStream([localTrack]));
    const botSrc   = audioCtx.createMediaStreamSource(new MediaStream([botTrack]));
    localSrc.connect(dest);
    botSrc.connect(dest);

    const recorder = new MediaRecorder(dest.stream, { mimeType:'audio/webm; codecs=opus' });

    recorder.ondataavailable = e => {
      if (e.data.size) chunksRef.current.push(e.data);
    };

    recorder.onstop = () => {
      // assemble & hand back to the parent
      const blob = new Blob(chunksRef.current, { type:'audio/webm' });
      const url  = URL.createObjectURL(blob);
      // clean up last URL
      if (lastUrlRef.current) {
        URL.revokeObjectURL(lastUrlRef.current);
      }
      lastUrlRef.current = url;
      onStopRecording(url, startTime.current);
    };

    recorderRef.current = recorder;

    return () => {
      // make sure we stop anything still running
      if (recorder.state === 'recording') recorder.stop();
      audioCtx.close();
      if (lastUrlRef.current) {
        URL.revokeObjectURL(lastUrlRef.current);
      }
    };
  }, [localTrack, botTrack]);


  // 2) drive start/stop (and reset) purely off transport changes
  useEffect(() => {
    const rec = recorderRef.current;
    if (!rec) return;

    if (transport === 'ready' && rec.state === 'inactive') {
      // wipe out any old data before we start
      chunksRef.current = [];
      // record a fresh time-stamp
      startTime.current = performance.now();
      // kick off recording
      rec.start();
    }

    if (transport === 'disconnected' && rec.state === 'recording') {
      rec.stop();
    }
  }, [transport]);

  return null;
}

