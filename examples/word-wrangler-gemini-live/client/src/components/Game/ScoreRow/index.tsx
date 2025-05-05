import { IconLaurelWreathFilled, IconStarFilled } from "@tabler/icons-react";
import styles from "./ScoreRow.module.css";
interface ScoreRowProps {
  score: number;
  bestScore: number;
}

export function ScoreRow({ score, bestScore = 0 }: ScoreRowProps) {
  return (
    <div className="flex flex-col w-full lg:w-auto justify-between gap-3 lg:gap-5">
      <div className="flex flex-1 w-full lg:w-auto flex-row items-center gap-3 lg:gap-5 text-white bg-black/20 rounded-2xl lg:rounded-3xl px-4 py-3 lg:px-6 lg:py-4">
        <IconStarFilled
          size={42}
          className="text-amber-300 size-8 lg:size-10"
        />
        <div className="flex flex-col gap-1">
          <span className="text-xs lg:text-sm uppercase font-extrabold tracking-wider">
            Current score
          </span>
          <span className="text-xl lg:text-2xl font-extrabold leading-none">
            {score}
          </span>
        </div>
      </div>
      <div className={styles.divider} />
      <div className="flex flex-row items-center gap-5 text-white rounded-3xl px-6">
        <IconLaurelWreathFilled
          size={42}
          className="text-amber-300 size-8 lg:size-10"
        />
        <div className="flex flex-col gap-1">
          <span className="text-xs lg:text-sm uppercase font-extrabold tracking-wider">
            Best score
          </span>
          <span className="text-xl lg:text-2xl font-extrabold leading-none">
            {bestScore}
          </span>
        </div>
      </div>
    </div>
  );
}

export default ScoreRow;
