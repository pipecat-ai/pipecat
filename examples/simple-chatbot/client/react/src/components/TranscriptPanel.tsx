import { useState, useRef, useEffect } from 'react';
import { useRTVIClientEvent } from '@pipecat-ai/client-react';
import { RTVIEvent, TranscriptData, BotLLMTextData } from '@pipecat-ai/client-js';
import './TranscriptPanel.css';

// Define interface for TTS word data
interface BotTtsTextData {
  words?: Array<{
    text: string;
    active: boolean;
  }>;
  // Other possible properties from the API
  text?: string;
}

export function TranscriptPanel() {
  const [messages, setMessages] = useState<Array<{
    sender: 'user' | 'bot';
    text: string;
    timestamp: Date;
  }>>([]);
  const [currentWords, setCurrentWords] = useState<Array<{
    text: string;
    active: boolean;
  }>>([]);
  const transcriptRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when new message arrives
  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
    }
  }, [messages]);

  // Handle user transcripts
  useRTVIClientEvent(
    RTVIEvent.UserTranscript,
    (data: TranscriptData) => {
      // Only add final transcripts
      if (data.final) {
        setMessages(prev => [
          ...prev,
          {
            sender: 'user',
            text: data.text,
            timestamp: new Date()
          }
        ]);
      }
    }
  );

  // Handle bot transcripts
  useRTVIClientEvent(
    RTVIEvent.BotTranscript,
    (data: BotLLMTextData) => {
      setMessages(prev => [
        ...prev,
        {
          sender: 'bot',
          text: data.text,
          timestamp: new Date()
        }
      ]);
    }
  );

  // Handle bot TTS with word-level highlighting
  useRTVIClientEvent(
    RTVIEvent.BotTtsText,
    (data: BotTtsTextData) => {
      if (data.words && data.words.length > 0) {
        setCurrentWords(data.words);
      }
    }
  );

  return (
    <div className="transcript-panel">
      <h3>Conversation Transcript</h3>
      <div ref={transcriptRef} className="transcript-container">
        {messages.map((message, index) => (
          <div 
            key={index} 
            className={`transcript-message ${message.sender}-message`}
          >
            <div className="message-sender">{message.sender === 'user' ? 'You' : 'Bot'}</div>
            <div className="message-text">{message.text}</div>
            <div className="message-time">{message.timestamp.toLocaleTimeString()}</div>
          </div>
        ))}
      </div>
      
      {currentWords.length > 0 && (
        <div className="word-level-transcription">
          <h4>Current Speech</h4>
          <div>
            {currentWords.map((word, index) => (
              <span 
                key={index}
                className={word.active ? 'highlighted-word' : 'word'}
              >
                {word.text}{' '}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
} 