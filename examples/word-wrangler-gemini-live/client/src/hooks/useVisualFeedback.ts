import { useState, useRef, useCallback } from 'react';
import { clearTimer } from '@/utils/timerUtils';
import { GAME_CONFIG } from '@/constants/gameConstants';

export function useVisualFeedback() {
  // Visual feedback state
  const [showAutoDetected, setShowAutoDetected] = useState(false);
  const [showIncorrect, setShowIncorrect] = useState(false);
  const autoDetectTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Reset all visual states
  const resetVisuals = useCallback(() => {
    setShowAutoDetected(false);
    setShowIncorrect(false);
    autoDetectTimerRef.current = clearTimer(autoDetectTimerRef.current);
  }, []);

  // Show correct animation
  const showCorrect = useCallback((onComplete?: () => void) => {
    // Clear any existing animation
    autoDetectTimerRef.current = clearTimer(autoDetectTimerRef.current);

    // Show correct animation
    setShowAutoDetected(true);
    setShowIncorrect(false);

    // Set timeout to hide animation
    autoDetectTimerRef.current = setTimeout(() => {
      setShowAutoDetected(false);
      if (onComplete) onComplete();
    }, GAME_CONFIG.ANIMATION_DURATION);
  }, []);

  // Show incorrect animation
  const showIncorrectAnimation = useCallback(() => {
    // Clear any existing animation
    autoDetectTimerRef.current = clearTimer(autoDetectTimerRef.current);

    // Show incorrect animation
    setShowIncorrect(true);
    setShowAutoDetected(false);

    // Set timeout to hide animation
    autoDetectTimerRef.current = setTimeout(() => {
      setShowIncorrect(false);
    }, GAME_CONFIG.ANIMATION_DURATION);
  }, []);

  // Clean up function
  const cleanup = useCallback(() => {
    autoDetectTimerRef.current = clearTimer(autoDetectTimerRef.current);
  }, []);

  return {
    showAutoDetected,
    showIncorrect,
    resetVisuals,
    showCorrect,
    showIncorrectAnimation,
    cleanup,
  };
}
