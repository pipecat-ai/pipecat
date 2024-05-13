"use client";

import { cn } from "@/app/utils";
import { motion } from "framer-motion";

export const TypewriterEffect = ({
  words,
  className,
}: {
  words: string[];
  className?: string;
  cursorClassName?: string;
}) => {
  const renderWords = () => {
    return (
      <div>
        {words.map((word, idx) => {
          return (
            <div key={`word-${idx}`} className="inline-block">
              {word.split("").map((char, index) => (
                <span key={`char-${index}`}>{char}</span>
              ))}
              &nbsp;
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className={cn("flex", className)}>
      {words.length < 1 ? (
        <span>...</span>
      ) : (
        <motion.div
          className="overflow-hidden"
          initial={{
            width: "0%",
          }}
          whileInView={{
            width: "fit-content",
          }}
          transition={{
            duration: 0.5,
            ease: "linear",
            delay: 1,
          }}
        >
          <div
            style={{
              whiteSpace: "nowrap",
            }}
          >
            {renderWords()}
          </div>
        </motion.div>
      )}
    </div>
  );
};
