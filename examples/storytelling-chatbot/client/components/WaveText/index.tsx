import React from "react";
import styles from "./WaveText.module.css";

interface Props {
  active: boolean;
}

export default function WaveText({ active }: Props) {
  return (
    <div className={`${styles.waveText} ${active ? styles.active : ""}`}>
      <span>W</span>
      <span>h</span>
      <span>a</span>
      <span>t</span>
      <span>&nbsp;&nbsp;</span>
      <span>n</span>
      <span>e</span>
      <span>x</span>
      <span>t</span>
      <span>?</span>
    </div>
  );
}
