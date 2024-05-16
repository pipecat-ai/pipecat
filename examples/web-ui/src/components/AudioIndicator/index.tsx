import {
  useAudioLevel,
  useAudioTrack,
  useLocalSessionId,
} from "@daily-co/daily-react";
import { useCallback, useRef } from "react";

import styles from "./styles.module.css";

export const AudioIndicatorBar: React.FC = () => {
  const localSessionId = useLocalSessionId();
  const audioTrack = useAudioTrack(localSessionId);

  const volRef = useRef<HTMLDivElement>(null);

  useAudioLevel(
    audioTrack?.persistentTrack,
    useCallback((volume) => {
      if (volRef.current)
        volRef.current.style.width = Math.max(2, volume * 100) + "%";
    }, [])
  );

  return (
    <div className={styles.bar}>
      <div ref={volRef} />
    </div>
  );
};

export default AudioIndicatorBar;
