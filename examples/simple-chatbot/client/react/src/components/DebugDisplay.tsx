import { useRef, useCallback } from 'react';
import {
  Participant,
  RTVIEvent,
  TransportState,
  TranscriptData,
  BotLLMTextData,
} from '@pipecat-ai/client-js';
import { useRTVIClient, useRTVIClientEvent } from '@pipecat-ai/client-react';
import './DebugDisplay.css';

export function DebugDisplay() {
  const debugLogRef = useRef<HTMLDivElement>(null);
  const client = useRTVIClient();

  const log = useCallback((message: string) => {
    if (!debugLogRef.current) return;

    const entry = document.createElement('div');
    entry.textContent = `${new Date().toISOString()} - ${message}`;

    // Add styling based on message type
    if (message.startsWith('User: ')) {
      entry.style.color = '#2196F3'; // blue for user
    } else if (message.startsWith('Bot: ')) {
      entry.style.color = '#4CAF50'; // green for bot
    }

    debugLogRef.current.appendChild(entry);
    debugLogRef.current.scrollTop = debugLogRef.current.scrollHeight;
  }, []);

  // Log transport state changes
  useRTVIClientEvent(
    RTVIEvent.TransportStateChanged,
    useCallback(
      (state: TransportState) => {
        log(`Transport state changed: ${state}`);
      },
      [log]
    )
  );

  // Log bot connection events
  useRTVIClientEvent(
    RTVIEvent.BotConnected,
    useCallback(
      (participant?: Participant) => {
        log(`Bot connected: ${JSON.stringify(participant)}`);
      },
      [log]
    )
  );

  useRTVIClientEvent(
    RTVIEvent.BotDisconnected,
    useCallback(
      (participant?: Participant) => {
        log(`Bot disconnected: ${JSON.stringify(participant)}`);
      },
      [log]
    )
  );

  // Log track events
  useRTVIClientEvent(
    RTVIEvent.TrackStarted,
    useCallback(
      (track: MediaStreamTrack, participant?: Participant) => {
        log(
          `Track started: ${track.kind} from ${participant?.name || 'unknown'}`
        );
      },
      [log]
    )
  );

  useRTVIClientEvent(
    RTVIEvent.TrackedStopped,
    useCallback(
      (track: MediaStreamTrack, participant?: Participant) => {
        log(
          `Track stopped: ${track.kind} from ${participant?.name || 'unknown'}`
        );
      },
      [log]
    )
  );

  // Log bot ready state and check tracks
  useRTVIClientEvent(
    RTVIEvent.BotReady,
    useCallback(() => {
      log(`Bot ready`);

      if (!client) return;

      const tracks = client.tracks();
      log(
        `Available tracks: ${JSON.stringify({
          local: {
            audio: !!tracks.local.audio,
            video: !!tracks.local.video,
          },
          bot: {
            audio: !!tracks.bot?.audio,
            video: !!tracks.bot?.video,
          },
        })}`
      );
    }, [client, log])
  );

  // Log transcripts
  useRTVIClientEvent(
    RTVIEvent.UserTranscript,
    useCallback(
      (data: TranscriptData) => {
        // Only log final transcripts
        if (data.final) {
          log(`User: ${data.text}`);
        }
      },
      [log]
    )
  );

  useRTVIClientEvent(
    RTVIEvent.BotTranscript,
    useCallback(
      (data: BotLLMTextData) => {
        log(`Bot: ${data.text}`);
      },
      [log]
    )
  );

  return (
    <div className="debug-panel">
      <h3>Debug Info</h3>
      <div ref={debugLogRef} className="debug-log" />
    </div>
  );
}
