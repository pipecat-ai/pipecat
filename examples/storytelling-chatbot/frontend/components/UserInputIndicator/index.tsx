import React, { useState, useEffect } from "react";

import { useAppMessage } from "@daily-co/daily-react";
import { DailyEventObjectAppMessage } from "@daily-co/daily-js";
import styles from "./UserInputIndicator.module.css";
import { IconMicrophone } from "@tabler/icons-react";
import { TypewriterEffect } from "../ui/typewriter";
import AudioIndicator from "../AudioIndicator";

interface Props {
  active: boolean;
}

export default function UserInputIndicator({ active }: Props) {
  const [transcription, setTranscription] = useState<string[]>([]);

  useAppMessage({
    onAppMessage: (e: DailyEventObjectAppMessage<any>) => {
      if (e.fromId && e.fromId === "transcription") {
        if (e.data.user_id === "" && e.data.is_final) {
          setTranscription((t) => [...t, ...e.data.text.split(" ")]);
        }
      }
    },
  });

  useEffect(() => {
    if (active) return;
    const t = setTimeout(() => setTranscription([]), 4000);
    return () => clearTimeout(t);
  }, [active]);

  return (
    <div className={`${styles.panel} ${active ? styles.active : ""}`}>
      <div className="relative z-20 flex flex-col">
        <div
          className={`${styles.micIcon} ${active ? styles.micIconActive : ""}`}
        >
          <IconMicrophone size={42} />
          {active && <AudioIndicator />}
        </div>
        <footer className={styles.transcript}>
          <TypewriterEffect words={transcription} />
        </footer>
      </div>
    </div>
  );
}
