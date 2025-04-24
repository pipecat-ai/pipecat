import { GAME_STATES, GAME_TEXT, GameState } from "@/constants/gameConstants";
import { IconArrowForwardUp, IconClockPause } from "@tabler/icons-react";
import React from "react";
import { GameWord } from "./GameWord";
import { Timer } from "./Timer";
import styles from "./WordWrangler.module.css";

interface GameContentProps {
  gameState: GameState;
  currentWord: string;
  showAutoDetected: boolean;
  timeLeft: number;
  showIncorrect: boolean;
  score: number;
  skipsRemaining: number;
  // onCorrect: () => void;
  onSkip: () => void;
}

export const GameContent: React.FC<GameContentProps> = ({
  gameState,
  currentWord,
  showAutoDetected,
  showIncorrect,
  timeLeft,
  score,
  skipsRemaining,
  //onCorrect,
  onSkip,
}) => {
  // Idle or Connecting State
  if (gameState === GAME_STATES.IDLE || gameState === GAME_STATES.CONNECTING) {
    return (
      <div className={styles.simpleLoadingPlaceholder}>
        {GAME_TEXT.startingGame}
      </div>
    );
  }

  // Waiting for Intro State
  if (gameState === GAME_STATES.WAITING_FOR_INTRO) {
    return (
      <div className="animate-pulse flex flex-1 flex-col gap-3 items-center justify-center">
        <span className="size-18 flex items-center justify-center rounded-full bg-slate-900/50 text-white">
          <IconClockPause size={42} className="opacity-50" />
        </span>
        <span className="text-white text-2xl font-bold">
          {GAME_TEXT.waitingForIntro}
        </span>
      </div>
    );
  }

  // Finished State
  if (gameState === GAME_STATES.FINISHED) {
    return (
      <div className={styles.gameReadyArea}>
        <div className={styles.gameResults}>
          <h2>{GAME_TEXT.gameOver}</h2>
          <p>
            {GAME_TEXT.finalScore}: <strong>{score}</strong>
          </p>
        </div>
        <div className={styles.statusNote}>{GAME_TEXT.clickToStart}</div>
      </div>
    );
  }

  // Active Game State
  return (
    <div className={styles.gameArea}>
      <GameWord
        word={currentWord}
        showAutoDetected={showAutoDetected}
        showIncorrect={showIncorrect}
      />
      <div className="flex flex-col lg:flex-row gap-2 w-full">
        <Timer timeLeft={timeLeft} gameState={gameState} />
        <button
          className="button"
          onClick={onSkip}
          disabled={skipsRemaining <= 0}
        >
          <IconArrowForwardUp size={24} className="opacity-50" />
          {skipsRemaining > 0
            ? GAME_TEXT.skipsRemaining(skipsRemaining)
            : GAME_TEXT.noSkips}
        </button>
      </div>
    </div>
  );
};
