import React from "react";
import Status from "./status";
import styles from "./styles.module.css";

import { useParticipantIds } from "@daily-co/daily-react";

export const Agent: React.FC = () => {
  const participantIds = useParticipantIds({ filter: "remote" });

  const status = participantIds.length > 0 ? "connected" : "connecting";
  return (
    <div className={styles.agent}>
      <div className={styles.agentWindow}></div>
      <footer className={styles.agentFooter}>
        <Status>User status</Status>
        <Status variant={status}>Agent status</Status>
      </footer>
    </div>
  );
};

export default Agent;
