import { GAME_STATES, GAME_TEXT } from "@/constants/gameConstants";
import { useConnectionState } from "@/hooks/useConnectionState";
import { useGameState } from "@/hooks/useGameState";
import { useGameTimer } from "@/hooks/useGameTimer";
import { useVisualFeedback } from "@/hooks/useVisualFeedback";
import { useWordDetection } from "@/hooks/useWordDetection";
import { RTVIEvent } from "@pipecat-ai/client-js";
import { useRTVIClientEvent } from "@pipecat-ai/client-react";
import { IconCircleDashedCheck, IconDoorExit } from "@tabler/icons-react";
import { useCallback, useEffect, useRef } from "react";
import Logo from "../../assets/logo.png";
import { GameContent } from "./GameContent";
import { ScoreRow } from "./ScoreRow";

import JSConfetti from "js-confetti";

import Image from "next/image";
import styles from "./WordWrangler.module.css";

export const WordWrangler: React.FC<{
  onGameEnded: (score: number, bestScore: number) => void;
}> = ({ onGameEnded }) => {
  const botIntroCompletedRef = useRef(false);
  const currentScoreRef = useRef(0);
  const gameState = useGameState();
  const visualFeedback = useVisualFeedback();
  const { isConnected, client } = useConnectionState();

  // Update the ref whenever score changes
  useEffect(() => {
    currentScoreRef.current = gameState.score;
  }, [gameState.score]);

  // End the game
  const endGame = useCallback(async () => {
    const scoreAtCallTime = currentScoreRef.current;

    // Prevent multiple calls to endGame
    if (gameState.gameState === GAME_STATES.FINISHED) {
      console.log("endGame prevented - game already finished");
      return;
    }

    // Capture the current score before any state changes
    const finalScore = scoreAtCallTime;
    const currentBestScore = gameState.bestScore;

    // Update game state
    gameState.finishGame();
    visualFeedback.resetVisuals();

    // Update best score if needed
    if (currentBestScore < finalScore) {
      gameState.setBestScore(finalScore);
    }

    // Disconnect the bot
    if (client && isConnected) {
      try {
        await client.disconnectBot();
        await client.disconnect();
      } catch (error) {
        console.error("Error disconnecting bot:", error);
      }
    }

    // Call the callback with the captured scores
    onGameEnded(finalScore, Math.max(finalScore, currentBestScore));
  }, [gameState, visualFeedback, client, isConnected, onGameEnded]);

  const gameTimer = useGameTimer(endGame);

  const wordDetection = useWordDetection({
    gameState: gameState.gameState,
    currentWord: gameState.currentWord,
    onCorrectGuess: handleCorrectGuess,
    onIncorrectGuess: handleIncorrectGuess,
  });

  // Initialize on component mount
  useEffect(() => {
    gameState.initializeGame();
  }, []);

  // Handle connection state changes
  useEffect(() => {
    if (isConnected) {
      if (!botIntroCompletedRef.current) {
        // Connection is active, but bot hasn't completed intro
        gameState.setGameState(GAME_STATES.WAITING_FOR_INTRO);
      }
    } else {
      // Connection lost or never established
      if (gameState.gameState === GAME_STATES.ACTIVE) {
        // If game was active, it's now finished
        endGame();
      } else if (gameState.gameState !== GAME_STATES.FINISHED) {
        // Reset to idle state if not already finished
        gameState.setGameState(GAME_STATES.IDLE);
      }

      // Reset intro state when connection is lost
      botIntroCompletedRef.current = false;
    }
  }, [isConnected, gameState.gameState, endGame]);

  // Listen for the bot to stop speaking to detect intro completion
  useRTVIClientEvent(RTVIEvent.BotStoppedSpeaking, () => {
    if (
      gameState.gameState === GAME_STATES.WAITING_FOR_INTRO &&
      !botIntroCompletedRef.current
    ) {
      // First time the bot stops speaking, consider intro done and start the game
      botIntroCompletedRef.current = true;
      startGame();
    }
  });

  // Handle correct guess with animation
  function handleCorrectGuess() {
    visualFeedback.showCorrect(() => {
      gameState.incrementScore();
      gameState.moveToNextWord();
      wordDetection.resetLastProcessedMessage();
    });
    const jsConfetti = new JSConfetti();
    jsConfetti.addConfetti();
  }

  // Handle incorrect guess with animation
  function handleIncorrectGuess() {
    visualFeedback.showIncorrectAnimation();
  }

  // Start the game
  function startGame() {
    // Initialize game state
    gameState.initializeGame();
    wordDetection.resetLastProcessedMessage();

    // Start the timer - now it internally manages countdown and calls endGame when done
    gameTimer.startTimer();
  }

  // Handle manual marking as correct
  function handleManualCorrect() {
    if (gameState.gameState !== GAME_STATES.ACTIVE) return;

    gameState.incrementScore();

    const jsConfetti = new JSConfetti();
    jsConfetti.addConfetti();

    gameState.moveToNextWord();
    wordDetection.resetLastProcessedMessage();
  }

  // Handle skipping a word
  function handleSkip() {
    if (gameState.gameState !== GAME_STATES.ACTIVE) return;

    // Try to use a skip and proceed if successful
    if (gameState.useSkip()) {
      gameState.moveToNextWord();
      wordDetection.resetLastProcessedMessage();
    }
  }

  // Clean up on unmount
  useEffect(() => {
    return () => {
      gameTimer.stopTimer();
      visualFeedback.cleanup();
    };
  }, []);

  return (
    <div className="min-h-[100dvh] flex flex-col">
      <div className="flex-1 flex flex-col items-center justify-center h-screen">
        <div className="flex flex-1 flex-col lg:flex-row gap-6 lg:gap-12 items-center justify-center w-full lg:w-auto">
          <div className={styles.gameContainer}>
            <Image
              src={Logo}
              alt="Word Wrangler"
              className="logo size-[140px] absolute top-[-50px] lg:top-[-60px] left-[50%] -translate-x-1/2 lg:left-auto lg:-translate-x-0 lg:right-[-50px] z-10"
              priority
            />
            <div className={styles.gameContent}>
              <GameContent
                gameState={gameState.gameState}
                timeLeft={gameTimer.timeLeft}
                currentWord={gameState.currentWord}
                showAutoDetected={visualFeedback.showAutoDetected}
                showIncorrect={visualFeedback.showIncorrect}
                score={gameState.score}
                skipsRemaining={gameState.skipsRemaining}
                onSkip={handleSkip}
              />
            </div>
          </div>
          <ScoreRow score={gameState.score} bestScore={gameState.bestScore} />
        </div>
        <footer className="flex gap-2 py-4 lg:flex-row lg:gap-4 lg:py-6 w-full items-center justify-center">
          <button
            className="button outline w-full lg:w-auto"
            onClick={handleManualCorrect}
            disabled={gameState.gameState !== GAME_STATES.ACTIVE}
          >
            <IconCircleDashedCheck size={24} />
            {GAME_TEXT.correct}
          </button>
          <button
            className="button outline w-full lg:w-auto"
            onClick={endGame}
            disabled={
              gameState.gameState == GAME_STATES.CONNECTING ||
              gameState.gameState == GAME_STATES.WAITING_FOR_INTRO
            }
          >
            <IconDoorExit size={24} />
            End Game
          </button>
        </footer>
      </div>
    </div>
  );
};
