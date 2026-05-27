/**
 * Shopping list — vanilla JS client.
 *
 * The "every input acts, may speak" pattern:
 *
 * - A standard voice pipeline runs on the bot. Its LLM converses (you
 *   hear it), and its user aggregator forwards each transcript to a
 *   separate UIWorker.
 * - The UIWorker reads the snapshot of THIS page and drives the list
 *   silently via three custom commands — ``add_item``, ``set_checked``,
 *   ``remove_item`` — plus the standard ``highlight`` (used to show
 *   "what's left").
 *
 * Each item is a checkbox with its text as the accessible name, so the
 * snapshot exposes every item's label and checked state to the worker.
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
const list = document.getElementById("list");
const listEmpty = document.getElementById("list-empty");
const addForm = document.getElementById("add-form");
const addInput = document.getElementById("add-input");

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

function refreshEmptyState() {
  listEmpty.hidden = list.children.length > 0;
}

// Render one item. The checkbox's accessible name is the item text, so
// the snapshot walker exposes each item as a checkbox with its label and
// checked state — that's what the UIWorker reads to resolve "the milk",
// "the last one", "the checked ones", etc.
function addItem(text, checked = false) {
  const trimmed = String(text ?? "").trim();
  if (!trimmed) return;

  const li = document.createElement("li");
  li.className = "item";

  const label = document.createElement("label");
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = !!checked;
  cb.setAttribute("aria-label", trimmed);
  cb.addEventListener("change", () => {
    li.classList.toggle("checked", cb.checked);
  });

  const span = document.createElement("span");
  span.className = "item-text";
  span.textContent = trimmed;

  label.appendChild(cb);
  label.appendChild(span);
  li.appendChild(label);
  li.classList.toggle("checked", !!checked);

  list.appendChild(li);
  li.classList.add("item-flash");
  setTimeout(() => li.classList.remove("item-flash"), 1000);
  refreshEmptyState();
}

function checkboxFor(ref) {
  const el = ref ? findElementByRef(ref) : null;
  if (!el) return null;
  if (el instanceof HTMLInputElement) return el;
  return el.querySelector?.("input[type=checkbox]") ?? null;
}

// ─────────────────────────────────────────────
// Command handlers
// ─────────────────────────────────────────────

function handleAddItem(payload) {
  addItem(payload?.text);
}

function handleSetChecked(payload) {
  const cb = checkboxFor(payload?.ref);
  if (!cb) return;
  cb.checked = payload?.checked !== false;
  cb.dispatchEvent(new Event("change", { bubbles: true }));
}

function handleRemoveItem(payload) {
  const cb = checkboxFor(payload?.ref);
  const li = cb?.closest(".item");
  if (!li) return;
  li.remove();
  refreshEmptyState();
}

// Standard ``highlight`` command: pulse the item row(s). The server may
// send one ref per command; we flash the enclosing item.
function handleHighlight(payload) {
  const cb = checkboxFor(payload?.ref);
  const row = cb?.closest(".item");
  if (!row) return;
  const duration = payload?.duration_ms ?? 1500;
  row.style.setProperty("--highlight-duration", `${duration}ms`);
  row.classList.remove("ui-highlight");
  void row.offsetWidth; // restart the animation if already running
  row.classList.add("ui-highlight");
  setTimeout(() => {
    row.classList.remove("ui-highlight");
    row.style.removeProperty("--highlight-duration");
  }, duration);
}

function onUICommand(command, handler) {
  const listener = (data) => {
    if (data.command !== command) return;
    handler(data.payload);
  };
  client.on(RTVIEvent.UICommand, listener);
  return () => client.off(RTVIEvent.UICommand, listener);
}

// Manual add (works without voice, and exercises the same snapshot path).
addForm.addEventListener("submit", (e) => {
  e.preventDefault();
  addItem(addInput.value);
  addInput.value = "";
});

// ─────────────────────────────────────────────
// Connection lifecycle
// ─────────────────────────────────────────────

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
    onUICommand("add_item", handleAddItem),
    onUICommand("set_checked", handleSetChecked),
    onUICommand("remove_item", handleRemoveItem),
    onUICommand("highlight", handleHighlight),
  ];

  try {
    await client.connect({ webrtcUrl: BOT_URL });
    client.startUISnapshotStream();
    connectButton.dataset.state = "connected";
    connectButton.textContent = "Disconnect";
    connectButton.disabled = false;
    setStatus("Connected. Try 'add milk and eggs'.", 5000);
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

refreshEmptyState();
