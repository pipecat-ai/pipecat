import React, { useCallback, useRef } from "react";

import {
  useAudioLevel,
  useAudioTrack,
  useLocalSessionId,
  //useAppMessage,
} from "@daily-co/daily-react";
//import { DailyEventObjectAppMessage } from "@daily-co/daily-js";
import { Mic, MicOff } from "lucide-react";
//import { TypewriterEffect } from "../ui/typewriter";
import styles from "./styles.module.css";

const AudioIndicatorBubble: React.FC = () => {
  const localSessionId = useLocalSessionId();
  const audioTrack = useAudioTrack(localSessionId);
  const volRef = useRef<HTMLDivElement>(null);

  useAudioLevel(
    audioTrack?.persistentTrack,
    useCallback((volume) => {
      // this volume number will be between 0 and 1
      // give it a minimum scale of 0.15 to not completely disappear 👻
      if (volRef.current) {
        const v = volume * 1.75;
        volRef.current.style.transform = `scale(${Math.max(0.1, v)})`;
      }
    }, [])
  );

  // Your audio track's audio volume visualized in a small circle,
  // whose size changes depending on the volume level
  return <div ref={volRef} className={styles.volume} />;
};

interface Props {
  active: boolean;
}

export default function UserMicBubble({ active }: Props) {
  /*
  const [transcription, setTranscription] = useState<string[]>([]);
  useAppMessage({
    onAppMessage: (e: DailyEventObjectAppMessage<any>) => {
      if (e.fromId && e.fromId === "transcription") {
        if (e.data.user_id === "" && e.data.is_final) {
          //setTranscription((t) => [...t, ...e.data.text.split(" ")]);
        }
      }
    },
  });

  useEffect(() => {
    if (active) return;
    const t = setTimeout(() => setTranscription([]), 4000);
    return () => clearTimeout(t);
  }, [active]);*/

  return (
    <div className={`${styles.bubbleContainer} ${active ? styles.active : ""}`}>
      <div
        className={`${styles.micIcon} ${active ? styles.micIconActive : ""}`}
      >
        {active ? <Mic size={42} /> : <MicOff size={42} />}
        {active && <AudioIndicatorBubble />}
      </div>
      <footer className={styles.transcript}></footer>
    </div>
  );
}
