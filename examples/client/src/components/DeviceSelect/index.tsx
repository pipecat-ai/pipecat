"use client";

import { useEffect } from "react";
import { DailyMeetingState } from "@daily-co/daily-js";
import { useDaily, useDevices } from "@daily-co/daily-react";
import { Mic, Speaker } from "lucide-react";
import { AudioIndicatorBar } from "../AudioIndicator";

import styles from "./styles.module.css";
import { Alert } from "../alert";

export function DeviceSelect() {
  const daily = useDaily();
  const {
    currentMic,
    hasMicError,
    micState,
    microphones,
    setMicrophone,
    currentSpeaker,
    speakers,
    setSpeaker,
  } = useDevices();

  const handleMicrophoneChange = (value: string) => {
    setMicrophone(value);
  };

  const handleSpeakerChange = (value: string) => {
    setSpeaker(value);
  };

  useEffect(() => {
    if (microphones.length > 0 || !daily || daily.isDestroyed()) return;
    const meetingState = daily.meetingState();
    const meetingStatesBeforeJoin: DailyMeetingState[] = [
      "new",
      "loading",
      "loaded",
    ];
    if (meetingStatesBeforeJoin.includes(meetingState)) {
      daily.startCamera({ startVideoOff: true, startAudioOff: false });
    }
  }, [daily, microphones]);

  return (
    <div className={styles.deviceSelect}>
      {hasMicError && (
        <Alert intent="danger" title="Device error">
          {micState === "blocked" ? (
            <>
              Please check your browser and system permissions. Make sure that
              this app is allowed to access your microphone and refresh the
              page.
            </>
          ) : micState === "in-use" ? (
            <>
              Your microphone is being used by another app. Please close any
              other apps using your microphone and restart this app.
            </>
          ) : micState === "not-found" ? (
            <>
              No microphone seems to be connected. Please connect a microphone.
            </>
          ) : micState === "not-supported" ? (
            <>
              This app is not supported on your device. Please update your
              software or use a different device.
            </>
          ) : (
            <>
              There seems to be an issue accessing your microphone. Try
              restarting the app or consult a system administrator.
            </>
          )}
        </Alert>
      )}

      <section className={styles.field}>
        <label className={styles.label}>Microphone:</label>
        <div className={styles.selectContainer}>
          <Mic size={24} />
          <select
            onChange={(e) => handleMicrophoneChange(e.target.value)}
            defaultValue={currentMic?.device.deviceId}
            className={styles.deviceSelectField}
          >
            {microphones.length === 0 ? (
              <option value="">Loading devices...</option>
            ) : (
              microphones.map((m) => (
                <option key={m.device.deviceId} value={m.device.deviceId}>
                  {m.device.label}
                </option>
              ))
            )}
          </select>
        </div>
        <AudioIndicatorBar />
      </section>

      <section className={styles.field}>
        <label className={styles.label}>Speakers:</label>
        <div className={styles.selectContainer}>
          <Speaker size={24} />
          <select
            onChange={(e) => handleSpeakerChange(e.target.value)}
            defaultValue={currentSpeaker?.device.deviceId}
            className={styles.deviceSelectField}
          >
            {speakers.length === 0 ? (
              <option value="">Loading devices...</option>
            ) : (
              speakers.map((m) => (
                <option key={m.device.deviceId} value={m.device.deviceId}>
                  {m.device.label}
                </option>
              ))
            )}
          </select>
        </div>
      </section>
    </div>
  );
}

export default DeviceSelect;
