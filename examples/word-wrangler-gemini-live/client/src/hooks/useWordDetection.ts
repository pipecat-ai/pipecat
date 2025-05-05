import { useRef } from 'react';
import { useRTVIClientEvent } from '@pipecat-ai/client-react';
import { RTVIEvent } from '@pipecat-ai/client-js';
import { detectWordGuess } from '@/utils/wordDetection';
import { GAME_STATES, GameState } from '@/constants/gameConstants';

interface UseWordDetectionProps {
  gameState: GameState;
  currentWord: string;
  onCorrectGuess: () => void;
  onIncorrectGuess: () => void;
}

export function useWordDetection({
  gameState,
  currentWord,
  onCorrectGuess,
  onIncorrectGuess,
}: UseWordDetectionProps) {
  const lastProcessedMessageRef = useRef('');

  // Reset the last processed message
  const resetLastProcessedMessage = () => {
    lastProcessedMessageRef.current = '';
  };

  // Listen for bot transcripts to detect correct answers
  useRTVIClientEvent(RTVIEvent.BotTranscript, (data) => {
    if (gameState !== GAME_STATES.ACTIVE) {
      return;
    }

    if (!currentWord) {
      return;
    }

    if (!data.text) {
      return;
    }

    // Skip if this is a repeat of the same transcript
    if (data.text === lastProcessedMessageRef.current) {
      return;
    }

    lastProcessedMessageRef.current = data.text;

    // Use the utility function to detect word guesses
    const result = detectWordGuess(data.text, currentWord);

    if (result.isCorrect) {
      onCorrectGuess();
    } else if (result.isExplicitGuess) {
      onIncorrectGuess();
    } else {
    }
  });

  return {
    resetLastProcessedMessage,
  };
}
