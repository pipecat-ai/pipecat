"use client";

import React from "react";
import { DailyProvider, useCallObject } from "@daily-co/daily-react";

import App from "../components/App";

export default function Home() {
  const callObject = useCallObject({});

  return (
    <DailyProvider callObject={callObject}>
      <App />
    </DailyProvider>
  );
}
