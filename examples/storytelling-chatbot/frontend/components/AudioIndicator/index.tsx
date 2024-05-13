import {
  useAudioLevel,
  useAudioTrack,
  useLocalSessionId,
} from "@daily-co/daily-react";
import { useCallback, useRef } from "react";

export const AudioIndicator: React.FC = () => {
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
  return (
    <>
      <div className="vol bg-teal-700" ref={volRef} />
      <style jsx>{`
        .vol {
          position: absolute;
          overflow: hidden;
          inset: 0px;
          z-index: 0;
          border-radius: 999px;
          transition: all 0.1s ease;
          transform: scale(0);
        }
      `}</style>
    </>
  );
};

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
    <div className="flex-1 bg-gray-200 h-[8px] rounded-full overflow-hidden">
      <div
        className="bg-green-500 h-[8px] w-[0] rounded-full transition-all duration-100 ease"
        ref={volRef}
      />
    </div>
  );
};

export default AudioIndicator;
