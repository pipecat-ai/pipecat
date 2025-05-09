// src/components/DebugDisplay.tsx
import { useRef, useCallback, useEffect } from 'react';
import {
  RTVIEvent,
  TranscriptData,
  BotLLMTextData,
} from '@pipecat-ai/client-js';
import { useRTVIClientEvent, useRTVIClientTransportState } from '@pipecat-ai/client-react';
import './TranscriptDisplay.css';

export function TranscriptDisplay() {
  const debugLogRef = useRef<HTMLDivElement>(null);
  const lastUserTextRef = useRef<string | null>(null);

  const transportState = useRTVIClientTransportState();

  useEffect(() => {
    if (transportState === 'connected') {
      // Clear the log when connected
      if (debugLogRef.current) {
        debugLogRef.current.innerHTML = '';
      }
    }
  }
  , [transportState]);
  
  const log = useCallback((message: string) => {
    if (!debugLogRef.current) return;

    const entry = document.createElement('div');
    entry.textContent = message

    if (message.startsWith('User: ')) {
      entry.style.color = '#2196F3'; // blue for user
    } else if (message.startsWith('Bot: ')) {
      entry.style.color = '#4CAF50'; // green for bot
    }

    debugLogRef.current.appendChild(entry);
    debugLogRef.current.scrollTop = debugLogRef.current.scrollHeight;
  }, []);

  // Log final user transcripts immediately
  useRTVIClientEvent(
    RTVIEvent.UserTranscript,
    useCallback((data: TranscriptData) => {
      if (!data.final) return;
      if (data.text === lastUserTextRef.current) {
        lastUserTextRef.current = null;  // reset
        return;  // duplicate—ignore
      }
      lastUserTextRef.current = data.text;
      log(`User: ${data.text}`);
    }, [log])
  );

  // Log bot transcripts as they arrive
  useRTVIClientEvent(
    RTVIEvent.BotTranscript,
    useCallback((data: BotLLMTextData) => {
      log(`Bot: ${data.text}`);
    }, [log])
  );

  

  return (
    <div className="transcript-panel">
      <div ref={debugLogRef} className="transcript-log" />
    </div>
  );
}
