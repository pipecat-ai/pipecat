import {
  useDaily,
  useLocalSessionId,
  useMediaTrack,
} from "@daily-co/daily-react";
import { useCallback } from "react";
import { IconMicrophone, IconMicrophoneOff } from "@tabler/icons-react";

export const MicToggle: React.FC = () => {
  const daily = useDaily();
  const localSessionId = useLocalSessionId();
  const audioTrack = useMediaTrack(localSessionId, "audio");
  const isMicMuted =
    audioTrack.state === "blocked" || audioTrack.state === "off";

  const handleClick = useCallback(() => {
    if (!daily) return;
    daily.setLocalAudio(isMicMuted);
  }, [daily, isMicMuted]);

  const text = isMicMuted ? (
    <IconMicrophone size={21} />
  ) : (
    <IconMicrophoneOff size={21} />
  );

  return (
    <button className="MicToggle UIButton" onClick={handleClick}>
      {text}
    </button>
  );
};

export default MicToggle;
