import React, { useEffect, useRef, useState } from "react";
import { LogOut, Settings } from "lucide-react";
import { DailyAudio } from "@daily-co/daily-react";

import DeviceSelect from "../DeviceSelect";
import Agent from "./agent";
import { Button } from "../button";
import styles from "./styles.module.css";
import UserMicBubble from "../UserMicBubble";

export const Session: React.FC = () => {
  const [showDevices, setShowDevices] = useState(false);
  const [canTalk, setCanTalk] = useState(false);
  const modalRef = useRef<HTMLDialogElement>(null);

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
        <UserMicBubble active={canTalk} />
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
          <Button>
            <LogOut size={16} />
            End
          </Button>
        </div>
      </footer>
    </>
  );
};

export default Session;
