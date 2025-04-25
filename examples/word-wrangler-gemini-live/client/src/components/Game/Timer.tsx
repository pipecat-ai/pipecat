import { GAME_CONFIG, GAME_STATES } from "@/constants/gameConstants";
import { formatTime } from "@/utils/formatTime";
import { IconStopwatch } from "@tabler/icons-react";
import styles from "./WordWrangler.module.css";

interface TimerProps {
  timeLeft: number;
  gameState: string;
}

export function Timer({ timeLeft, gameState }: TimerProps) {
  const lowTimer =
    gameState === GAME_STATES.ACTIVE &&
    timeLeft <= GAME_CONFIG.LOW_TIME_WARNING;

  return (
    <div className={`${styles.timer} ${lowTimer ? styles.lowTime : ""}`}>
      <div className={styles.timerBadge}>
        <IconStopwatch size={24} />
        <span>{formatTime(timeLeft)}</span>
      </div>
      <div className={styles.timerBar}>
        <div
          className={styles.timerBarFill}
          style={{ width: `${(timeLeft / GAME_CONFIG.GAME_DURATION) * 100}%` }}
        />
      </div>
    </div>
  );
}
