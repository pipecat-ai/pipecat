import { TRANSCRIPT_PATTERNS } from '@/constants/gameConstants';

/**
 * Checks if a transcript contains a correct guess for the target word
 */
export function detectWordGuess(transcript: string, targetWord: string) {
  const currentWordLower = targetWord.toLowerCase().trim();

  // Primary detection: Look for explicit guesses
  const guessPattern = TRANSCRIPT_PATTERNS.GUESS_PATTERN;
  const guessMatch = transcript.match(guessPattern);

  if (guessMatch) {
    // Extract the guessed word from whichever group matched (group 1 or 2)
    let guessedWord = (guessMatch[1] || guessMatch[2] || '')
      .toLowerCase()
      .trim();

    // Remove articles ("a", "an", "the") from the beginning of the guessed word
    guessedWord = guessedWord.replace(/^(a|an|the)\s+/i, '');

    return {
      isCorrect: guessedWord === currentWordLower,
      isExplicitGuess: true,
      guessedWord,
    };
  }

  // Secondary detection: Check if word appears in transcript
  const containsWord = transcript.toLowerCase().includes(currentWordLower);

  return {
    isCorrect: containsWord,
    isExplicitGuess: false,
    guessedWord: containsWord ? targetWord : null,
  };
}
