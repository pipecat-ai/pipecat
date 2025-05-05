/**
 * Safely clears any type of timer
 * @param timer The timer to clear
 * @returns null to reassign to the timer reference
 */
export function clearTimer(timer: NodeJS.Timeout | null): null {
  if (timer) {
    clearTimeout(timer);
  }
  return null;
}

/**
 * Creates a countdown timer that calls the callback every second
 * @returns A function to stop the timer
 */
export function createCountdownTimer(
  durationSeconds: number,
  onTick: (secondsLeft: number) => void,
  onComplete: () => void
): () => void {
  let secondsLeft = durationSeconds;

  const timer = setInterval(() => {
    secondsLeft--;
    onTick(secondsLeft);

    if (secondsLeft <= 0) {
      clearInterval(timer);
      onComplete();
    }
  }, 1000);

  return () => {
    clearInterval(timer);
  };
}
