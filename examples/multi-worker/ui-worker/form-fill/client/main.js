/**
 * Form fill — vanilla JS client.
 *
 * Same base wiring as pointing/deixis (PipecatClient
 * + managed snapshot streaming + bot audio sink). Three command
 * handlers: ``scroll_to``, ``set_input_value``, and ``click``.
 *
 * ``set_input_value`` writes a string into an ``<input>`` /
 * ``<textarea>``. Crucially it dispatches ``input`` and ``change``
 * events so React-controlled or other frameworks pick up the change
 * naturally; the React standard handler does the same. ``click``
 * presses the submit button (and works for any clickable element).
 *
 * The submit button intercepts the form's submit event so the demo
 * stays on-page after the worker submits. Real apps would let it
 * through.
 */

import {
  PipecatClient,
  RTVIEvent,
  findElementByRef,
} from "@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";

const BOT_URL = "http://localhost:7860/api/offer";

const connectButton = document.getElementById("connect");
const status = document.getElementById("status");
const botAudio = document.getElementById("bot-audio");
const form = document.getElementById("application-form");
const formStatus = document.getElementById("form-status");

let client;
let unsubscribes = [];

function setStatus(text, autoHideMs = 0) {
  status.textContent = text;
  status.dataset.show = text ? "1" : "0";
  if (text && autoHideMs > 0) {
    setTimeout(() => {
      if (status.textContent === text) status.dataset.show = "0";
    }, autoHideMs);
  }
}

function resolveTarget(payload) {
  if (payload?.ref) {
    const el = findElementByRef(payload.ref);
    if (el) return el;
  }
  if (payload?.target_id) {
    return document.getElementById(payload.target_id);
  }
  return null;
}

function handleScrollTo(payload) {
  const el = resolveTarget(payload);
  if (!el) return;
  const behavior =
    payload?.behavior === "instant" || payload?.behavior === "smooth"
      ? payload.behavior
      : "smooth";
  el.scrollIntoView({ behavior, block: "center", inline: "nearest" });
}

/**
 * Write ``payload.value`` into the targeted input/textarea.
 *
 * Skips ``disabled`` / ``readonly`` / ``type="hidden"`` targets so
 * the worker can't bypass UI affordances. Dispatches ``input`` and
 * ``change`` events so framework-controlled inputs (React, Vue, etc.)
 * notice the change. Briefly flashes the field so the user sees
 * what the worker wrote.
 */
function handleSetInputValue(payload) {
  const el = resolveTarget(payload);
  if (!el) return;
  if (!(el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement))
    return;
  if (el.disabled || el.readOnly) return;
  if (el.type === "hidden") return;

  const value = String(payload?.value ?? "");
  const replace = payload?.replace !== false;
  el.value = replace ? value : (el.value || "") + value;

  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));

  // Visual confirmation: a brief background flash so the user
  // notices the write happened.
  el.classList.remove("fill-flash");
  void el.offsetWidth;
  el.classList.add("fill-flash");
  setTimeout(() => el.classList.remove("fill-flash"), 1200);
}

/**
 * Click the targeted element. Skips ``disabled`` targets so the
 * worker can't bypass disabled affordances; the standard React
 * handler does the same.
 */
function handleClick(payload) {
  const el = resolveTarget(payload);
  if (!el) return;
  if ("disabled" in el && el.disabled) return;
  el.click();
}

// Don't actually submit the form on the demo; the worker says "I
// submitted it" and we show a status message instead.
form.addEventListener("submit", (e) => {
  e.preventDefault();
  formStatus.textContent = "Submitted (demo only — no network call).";
  formStatus.style.color = "#16a34a";
});

function onUICommand(command, handler) {
  const listener = (data) => {
    if (data.command !== command) return;
    handler(data.payload);
  };
  client.on(RTVIEvent.UICommand, listener);
  return () => client.off(RTVIEvent.UICommand, listener);
}

async function connect() {
  connectButton.disabled = true;
  setStatus("Connecting…");

  client = new PipecatClient({
    transport: new SmallWebRTCTransport(),
    enableMic: true,
    enableCam: false,
  });

  client.on(RTVIEvent.BotConnected, () => setStatus("Bot connected", 1500));
  client.on(RTVIEvent.Disconnected, () => {
    setStatus("Disconnected", 2000);
    connectButton.dataset.state = "";
    connectButton.textContent = "Connect";
    connectButton.disabled = false;
    teardownUI();
  });

  client.on(RTVIEvent.TrackStarted, (track, participant) => {
    if (track.kind !== "audio") return;
    if (participant?.local) return;
    botAudio.srcObject = new MediaStream([track]);
  });

  unsubscribes = [
    onUICommand("scroll_to", handleScrollTo),
    onUICommand("set_input_value", handleSetInputValue),
    onUICommand("click", handleClick),
  ];

  try {
    await client.connect({ webrtcUrl: BOT_URL });
    client.startUISnapshotStream();
    connectButton.dataset.state = "connected";
    connectButton.textContent = "Disconnect";
    connectButton.disabled = false;
    setStatus("Connected — the assistant will guide you through the form.", 5000);
  } catch (err) {
    console.error("Connect failed:", err);
    setStatus(`Connect failed: ${err.message ?? err}`, 4000);
    teardownUI();
    connectButton.disabled = false;
  }
}

async function disconnect() {
  connectButton.disabled = true;
  setStatus("Disconnecting…");
  try {
    await client?.disconnect();
  } finally {
    teardownUI();
    connectButton.dataset.state = "";
    connectButton.textContent = "Connect";
    connectButton.disabled = false;
  }
}

function teardownUI() {
  client?.stopUISnapshotStream();
  unsubscribes.forEach((unsubscribe) => unsubscribe());
  unsubscribes = [];
  if (botAudio.srcObject) botAudio.srcObject = null;
  client = undefined;
}

connectButton.addEventListener("click", () => {
  if (connectButton.dataset.state === "connected") {
    disconnect();
  } else {
    connect();
  }
});
