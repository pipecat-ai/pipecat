import { GAME_CONFIG } from "@/constants/gameConstants";
import { clearTimer } from "@/utils/timerUtils";
import { useCallback, useEffect, useRef, useState } from "react";

export function useGameTimer(onTimeUp: () => void) {
  const [timeLeft, setTimeLeft] = useState(GAME_CONFIG.GAME_DURATION);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const hasCalledTimeUpRef = useRef(false);

  // Start the game timer with initial duration
  const startTimer = useCallback(() => {
    // Reset time left and timeUp flag
    setTimeLeft(GAME_CONFIG.GAME_DURATION);
    hasCalledTimeUpRef.current = false;

    // Clear any existing timer
    timerRef.current = clearTimer(timerRef.current);

    // Start a new timer
    timerRef.current = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev <= 1 && !hasCalledTimeUpRef.current) {
          // Time's up - clear the interval and call the callback
          timerRef.current = clearTimer(timerRef.current);
          hasCalledTimeUpRef.current = true;
          onTimeUp();
          return 0;
        }
        return prev - 1;
      });
    }, GAME_CONFIG.TIMER_INTERVAL);
  }, [onTimeUp]);

  // Stop the timer
  const stopTimer = useCallback(() => {
    timerRef.current = clearTimer(timerRef.current);
    hasCalledTimeUpRef.current = false;
  }, []);

  // Reset the timer to initial value without starting it
  const resetTimer = useCallback(() => {
    setTimeLeft(GAME_CONFIG.GAME_DURATION);
    hasCalledTimeUpRef.current = false;
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      timerRef.current = clearTimer(timerRef.current);
    };
  }, []);

  return {
    timeLeft,
    startTimer,
    stopTimer,
    resetTimer,
  };
}
