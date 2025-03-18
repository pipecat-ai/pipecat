import { useRef, useCallback, useState, useEffect } from 'react';
import {
  Participant,
  RTVIEvent,
  TransportState,
  TranscriptData,
  BotLLMTextData,
} from '@pipecat-ai/client-js';
import { useRTVIClient, useRTVIClientEvent } from '@pipecat-ai/client-react';

interface Message {
  id: string;
  text: string;
  type: 'user' | 'assistant' | 'system';
  timestamp: Date;
  showMemoryRefresh?: boolean;
  memoryRefreshTime?: number;
  showMemoryIndicator?: boolean;
}

interface DebugDisplayProps {
  onNewMessage: (type: 'user' | 'assistant' | 'system') => void;
}

// Create a shared context for memory refresh
export const memoryRefreshTrigger = {
  value: 0,
  subscribers: new Set<(value: number) => void>(),
  increment() {
    this.value++;
    this.subscribers.forEach(callback => callback(this.value));
  }
};

export function DebugDisplay({ onNewMessage }: DebugDisplayProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const client = useRTVIClient();

  // Add effect to handle delayed appearance of memory refresh indicators
  useEffect(() => {
    const timeouts: NodeJS.Timeout[] = [];
    
    messages.forEach(message => {
      if (message.showMemoryRefresh && !message.showMemoryIndicator) {
        const timeout = setTimeout(() => {
          setMessages(prev => prev.map(msg => 
            msg.id === message.id 
              ? { ...msg, showMemoryIndicator: true }
              : msg
          ));
        }, 800);
        timeouts.push(timeout);
      }
    });

    return () => {
      timeouts.forEach(timeout => clearTimeout(timeout));
    };
  }, [messages]);

  const generateRandomRefreshTime = () => {
    return Number((Math.random() * (100 - 50) + 50).toFixed(2));
  };

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  const addMessage = useCallback((text: string, type: 'user' | 'assistant' | 'system') => {
    const newMessage: Message = {
      id: `${Date.now()}-${Math.random()}`,
      text,
      type,
      timestamp: new Date(),
      showMemoryRefresh: type === 'user',
      memoryRefreshTime: type === 'user' ? generateRandomRefreshTime() : undefined
    };

    setMessages(prev => [...prev, newMessage]);
    onNewMessage(type);
    if (type === 'assistant') {
      memoryRefreshTrigger.increment();
    }
    setTimeout(scrollToBottom, 100);
  }, [scrollToBottom, onNewMessage]);

  // Log transport state changes
  useRTVIClientEvent(
    RTVIEvent.TransportStateChanged,
    useCallback(
      (state: TransportState) => {
        addMessage(`Transport state changed: ${state}`, 'system');
      },
      [addMessage]
    )
  );

  // Log bot connection events
  useRTVIClientEvent(
    RTVIEvent.BotConnected,
    useCallback(
      (participant?: Participant) => {
        addMessage(`Bot connected: ${JSON.stringify(participant)}`, 'system');
      },
      [addMessage]
    )
  );

  useRTVIClientEvent(
    RTVIEvent.BotDisconnected,
    useCallback(
      (participant?: Participant) => {
        addMessage(`Bot disconnected: ${JSON.stringify(participant)}`, 'system');
      },
      [addMessage]
    )
  );

  // Log track events
  useRTVIClientEvent(
    RTVIEvent.TrackStarted,
    useCallback(
      (track: MediaStreamTrack, participant?: Participant) => {
        addMessage(
          `Track started: ${track.kind} from ${participant?.name || 'unknown'}`,
          'system'
        );
      },
      [addMessage]
    )
  );

  useRTVIClientEvent(
    RTVIEvent.TrackStopped,
    useCallback(
      (track: MediaStreamTrack, participant?: Participant) => {
        addMessage(
          `Track stopped: ${track.kind} from ${participant?.name || 'unknown'}`,
          'system'
        );
      },
      [addMessage]
    )
  );

  // Log bot ready state and check tracks
  useRTVIClientEvent(
    RTVIEvent.BotReady,
    useCallback(() => {
      addMessage('Bot ready', 'system');

      if (!client) return;

      const tracks = client.tracks();
      addMessage(
        `Available tracks: ${JSON.stringify({
          local: {
            audio: !!tracks.local.audio,
            video: !!tracks.local.video,
          },
          bot: {
            audio: !!tracks.bot?.audio,
            video: !!tracks.bot?.video,
          },
        })}`,
        'system'
      );
    }, [client, addMessage])
  );

  // Log transcripts
  useRTVIClientEvent(
    RTVIEvent.UserTranscript,
    useCallback(
      (data: TranscriptData) => {
        // Only log final transcripts
        if (data.final) {
          addMessage(data.text, 'user');
        }
      },
      [addMessage]
    )
  );

  useRTVIClientEvent(
    RTVIEvent.BotTranscript,
    useCallback(
      (data: BotLLMTextData) => {
        addMessage(data.text, 'assistant');
      },
      [addMessage]
    )
  );

  return (
    <div className="w-full h-full mx-auto p-6 bg-zinc-50">
      <div className="flex flex-col gap-4 pb-16">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`
              flex animate-message-pop
              ${message.type === 'system' ? 'justify-center mx-auto w-full' : 'max-w-[80%]'}
              ${message.type === 'user' ? 'justify-end ml-auto' : ''}
              ${message.type === 'assistant' ? 'justify-start mr-auto' : ''}
            `}
          >
            {message.type === 'system' ? (
              <div className="text-sm text-zinc-500 font-medium text-center">
                {message.text}
              </div>
            ) : (
              <div className="flex flex-col">
                {!message.showMemoryRefresh && message.showMemoryIndicator && (
                  <div className="text-xs bg-blue-100 text-blue-400 px-2 py-1 rounded-md self-end animate-memory-indicator overflow-hidden">
                    Memory refresh: {message.memoryRefreshTime}ms
                  </div>
                )}
                <div
                  className={`
                    px-4 py-3 rounded-2xl shadow-sm transition-all duration-300 hover:shadow-md
                    ${message.type === 'user' ? 'bg-zinc-800 text-white' : ''}
                    ${message.type === 'assistant' ? 'bg-white text-zinc-800' : ''}
                  `}
                >
                  <div className="break-words">{message.text}</div>
                  <div className="text-xs mt-1 opacity-70">
                    {message.timestamp.toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                      second: '2-digit'
                    })}
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}
