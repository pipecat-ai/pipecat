import React, { useEffect, useRef, useState } from "react";
import { LogOut, Settings } from "lucide-react";
import { DailyAudio, useAppMessage, useDaily } from "@daily-co/daily-react";

import DeviceSelect from "../DeviceSelect";
import Agent from "./agent";
import { Button } from "../button";
import UserMicBubble from "../UserMicBubble";

import styles from "./styles.module.css";

interface SessionProps {
  onLeave: () => void;
  openMic?: boolean;
}

export const Session: React.FC<SessionProps> = ({
  onLeave,
  openMic = false,
}) => {
  const daily = useDaily();
  const [showDevices, setShowDevices] = useState(false);
  const modalRef = useRef<HTMLDialogElement>(null);
  const [talkState, setTalkState] = useState<"user" | "assistant" | "open">(
    openMic ? "open" : "assistant"
  );

  useAppMessage({
    onAppMessage: (e) => {
      if (!daily || !e.data?.cue) return;

      // Determine the UI state from the cue sent by the bot
      if (e.data?.cue === "user_turn") {
        // Delay enabling local mic input to avoid feedback from LLM
        setTimeout(() => daily.setLocalAudio(true), 500);
        setTalkState("user");
      } else {
        daily.setLocalAudio(false);
        setTalkState("assistant");
      }
    },
  });

  useEffect(() => {
    const current = modalRef.current;
    // Backdrop doesn't currently work with dialog open, so we use setModal instead
    if (current && showDevices) {
      current.inert = true;
      current.showModal();
      current.inert = false;
    }
    return () => current?.close();
  }, [showDevices]);

  return (
    <>
      <dialog ref={modalRef}>
        <h2>Configure devices</h2>
        <DeviceSelect />
        <Button onClick={() => setShowDevices(false)}>Close</Button>
      </dialog>

      <div className={styles.agentContainer}>
        <Agent />
        <UserMicBubble openMic={openMic} active={talkState !== "assistant"} />
        <DailyAudio />
      </div>

      <footer className={styles.footer}>
        <div className={styles.controls}>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setShowDevices(true)}
          >
            <Settings />
          </Button>
          <Button onClick={() => onLeave()}>
            <LogOut size={16} />
            End
          </Button>
        </div>
      </footer>
    </>
  );
};

export default Session;
