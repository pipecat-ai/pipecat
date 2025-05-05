import { GAME_CONFIG, GAME_STATES, GameState } from "@/constants/gameConstants";
import { getRandomCatchPhraseWords } from "@/data/wordWranglerWords";
import { useCallback, useState } from "react";

export function useGameState() {
  // Game state
  const [gameState, setGameState] = useState<GameState>(GAME_STATES.IDLE);
  const [timeLeft, setTimeLeft] = useState(GAME_CONFIG.GAME_DURATION);
  const [score, setScore] = useState(0);
  const [words, setWords] = useState<string[]>([]);
  const [currentWordIndex, setCurrentWordIndex] = useState(0);
  const [skipsRemaining, setSkipsRemaining] = useState(GAME_CONFIG.MAX_SKIPS);
  const [bestScore, _setBestScore] = useState(0);

  // Initialize or reset game state
  const initializeGame = useCallback(() => {
    const freshWords = getRandomCatchPhraseWords(GAME_CONFIG.WORD_POOL_SIZE);
    setWords(freshWords);
    setGameState(GAME_STATES.ACTIVE);
    setTimeLeft(GAME_CONFIG.GAME_DURATION);
    setScore(0);
    setCurrentWordIndex(0);
    setSkipsRemaining(GAME_CONFIG.MAX_SKIPS);

    // Get best score from local storage
    const storedScore = localStorage.getItem("bestScore");
    if (storedScore) {
      _setBestScore(Number(storedScore) || 0);
    }
    return freshWords;
  }, []);

  // End game
  const finishGame = useCallback(() => {
    setGameState(GAME_STATES.FINISHED);
  }, []);

  // Handle scoring
  const incrementScore = useCallback(() => {
    setScore((prev) => prev + 1);
  }, []);

  // Handle best score
  const setBestScore = useCallback((newBestScore: number) => {
    _setBestScore(newBestScore);
    localStorage.setItem("bestScore", newBestScore.toString());
  }, []);

  // Handle word navigation
  const moveToNextWord = useCallback(() => {
    setCurrentWordIndex((prev) => {
      if (prev >= words.length - 1) {
        // If we're at the end of the word list, get new words
        setWords(getRandomCatchPhraseWords(GAME_CONFIG.WORD_POOL_SIZE));
        return 0;
      }
      return prev + 1;
    });
  }, [words]);

  // Handle skipping
  const useSkip = useCallback(() => {
    if (skipsRemaining <= 0) return false;
    setSkipsRemaining((prev) => prev - 1);
    return true;
  }, [skipsRemaining]);

  // Update timer
  const decrementTimer = useCallback(() => {
    return setTimeLeft((prev) => {
      if (prev <= 1) {
        return 0;
      }
      return prev - 1;
    });
  }, []);

  return {
    // State
    gameState,
    setGameState,
    timeLeft,
    score,
    bestScore,
    words,
    currentWord: words[currentWordIndex] || "",
    skipsRemaining,

    // Actions
    initializeGame,
    finishGame,
    incrementScore,
    setBestScore,
    moveToNextWord,
    useSkip,
    decrementTimer,
  };
}
