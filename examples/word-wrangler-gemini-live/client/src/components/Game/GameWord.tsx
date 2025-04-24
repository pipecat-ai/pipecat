import { GAME_TEXT } from "@/constants/gameConstants";
import React from "react";
import styles from "./WordWrangler.module.css";

interface GameWordProps {
  word: string;
  showAutoDetected: boolean;
  showIncorrect: boolean;
}

export const GameWord: React.FC<GameWordProps> = ({
  word,
  showAutoDetected,
  showIncorrect,
}) => {
  return (
    <div
      className={`${styles.currentWord} ${
        showAutoDetected ? styles.correctWordDetected : ""
      } ${showIncorrect ? styles.incorrectWordDetected : ""}`}
    >
      <span className={styles.helpText}>{GAME_TEXT.describeWord}</span>
      <span className={styles.word}>{word}</span>

      {showAutoDetected && <CorrectOverlay />}
      {showIncorrect && <IncorrectOverlay />}
    </div>
  );
};

const CorrectOverlay: React.FC = () => (
  <div className={styles.autoDetectedOverlay}>
    <div className={styles.checkmarkContainer}>
      <svg
        className={styles.checkmarkSvg}
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 52 52"
      >
        <circle
          className={styles.checkmarkCircle}
          cx="26"
          cy="26"
          r="25"
          fill="none"
        />
        <path
          className={styles.checkmarkCheck}
          fill="none"
          d="M14.1 27.2l7.1 7.2 16.7-16.8"
        />
      </svg>
    </div>
  </div>
);

const IncorrectOverlay: React.FC = () => (
  <div className={styles.incorrectOverlay}>
    <div className={styles.xmarkContainer}>
      <svg
        className={styles.xmarkSvg}
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 52 52"
      >
        <circle
          className={styles.xmarkCircle}
          cx="26"
          cy="26"
          r="25"
          fill="none"
        />
        <path
          className={styles.xmarkX}
          fill="none"
          d="M16 16 L36 36 M36 16 L16 36"
        />
      </svg>
    </div>
  </div>
);
