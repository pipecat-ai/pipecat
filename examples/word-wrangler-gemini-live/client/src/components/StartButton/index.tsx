import { BUTTON_TEXT } from "@/constants/gameConstants";
import { useConnectionState } from "@/hooks/useConnectionState";
import { IconArrowRight } from "@tabler/icons-react";

interface StartGameButtonProps {
  onGameStarted?: () => void;
  onGameEnded?: () => void;
  isGameEnded?: boolean;
}

export function StartGameButton({
  onGameStarted,
  onGameEnded,
  isGameEnded,
}: StartGameButtonProps) {
  const { isConnecting, isDisconnecting, toggleConnection } =
    useConnectionState(onGameStarted, onGameEnded);

  // Show spinner during connection process
  const showSpinner = isConnecting;
  const btnText = isGameEnded ? BUTTON_TEXT.RESTART : BUTTON_TEXT.START;

  return (
    <div className="flex justify-center">
      <button
        className="styled-button"
        onClick={toggleConnection}
        disabled={isConnecting || isDisconnecting}
      >
        <>
          <span className="styled-button-text">
            {isConnecting ? BUTTON_TEXT.CONNECTING : btnText}
          </span>
          <span className="styled-button-icon">
            {showSpinner ? (
              <span className="spinner"></span>
            ) : (
              <IconArrowRight size={16} strokeWidth={3} />
            )}
          </span>
        </>
      </button>
    </div>
  );
}
