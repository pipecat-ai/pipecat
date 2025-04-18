"use client";

import { useEffect } from "react";
import { DailyMeetingState } from "@daily-co/daily-js";
import { useDaily, useDevices } from "@daily-co/daily-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { IconMicrophone, IconDeviceSpeaker } from "@tabler/icons-react";
import { AudioIndicatorBar } from "../AudioIndicator";

interface Props {}

export default function DevicePicker({}: Props) {
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
    <div className="flex flex-col gap-5">
      <section>
        <label className="uppercase text-sm tracking-wider text-gray-500">
          Microphone:
        </label>
        <div className="flex flex-row gap-4 items-center mt-2">
          <IconMicrophone size={24} />
          <div className="flex flex-col flex-1 gap-3">
            <Select onValueChange={handleMicrophoneChange}>
              <SelectTrigger className="">
                <SelectValue
                  placeholder={
                    hasMicError ? "error" : currentMic?.device?.label
                  }
                />
              </SelectTrigger>
              <SelectContent>
                {hasMicError && (
                  <option value="error" disabled>
                    No microphone access.
                  </option>
                )}

                {microphones.map((m) => (
                  <SelectItem key={m.device.deviceId} value={m.device.deviceId}>
                    {m.device.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <AudioIndicatorBar />
          </div>
        </div>
      </section>

      <section>
        <label className="uppercase text-sm tracking-wider text-gray-500">
          Speakers:
        </label>
        <div className="flex flex-row gap-4 items-center mt-2">
          <IconDeviceSpeaker size={24} />
          <Select onValueChange={handleSpeakerChange}>
            <SelectTrigger className="">
              <SelectValue placeholder={currentSpeaker?.device?.label} />
            </SelectTrigger>
            <SelectContent>
              {speakers.map((m) => (
                <SelectItem key={m.device.deviceId} value={m.device.deviceId}>
                  {m.device.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </section>
      {hasMicError && (
        <div className="error">
          {micState === "blocked" ? (
            <p className="text-red-500">
              Please check your browser and system permissions. Make sure that
              this app is allowed to access your microphone.
            </p>
          ) : micState === "in-use" ? (
            <p className="text-red-500">
              Your microphone is being used by another app. Please close any
              other apps using your microphone and restart this app.
            </p>
          ) : micState === "not-found" ? (
            <p className="text-red-500">
              No microphone seems to be connected. Please connect a microphone.
            </p>
          ) : micState === "not-supported" ? (
            <p className="text-red-500">
              This app is not supported on your device. Please update your
              software or use a different device.
            </p>
          ) : (
            <p className="text-red-500">
              There seems to be an issue accessing your microphone. Try
              restarting the app or consult a system administrator.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
