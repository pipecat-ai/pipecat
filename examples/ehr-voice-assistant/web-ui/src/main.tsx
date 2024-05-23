import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.tsx";
import Header from "./components/header.tsx";
import { DailyProvider } from "@daily-co/daily-react";

import "./css/global.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Header />
    <main>
      <DailyProvider>
        <App />
      </DailyProvider>
    </main>
  </React.StrictMode>
);
