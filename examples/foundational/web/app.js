const micBtn = document.getElementById("mic-btn");
const visualizer = document.getElementById("visualizer");
const chatWindow = document.getElementById("chat-window");

// Draw basic waveform circle that pulses on audiolevel events
const ctx = visualizer.getContext("2d");
let level = 0;
function draw() {
  const radius = (visualizer.width / 2) * (0.4 + 0.6 * level);
  ctx.clearRect(0, 0, visualizer.width, visualizer.height);
  ctx.beginPath();
  ctx.arc(visualizer.width / 2, visualizer.height / 2, radius, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(88,166,255,0.7)";
  ctx.fill();
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

// No external client needed; we leverage prebuilt /client/ via hidden iframe.

// We'll update `level` directly from VAD callbacks below.

// Live captions over WebSocket
let ws;
function openWS() {
  const proto = location.protocol === "https:" ? "wss://" : "ws://";
  ws = new WebSocket(`${proto}${location.hostname}:9876`);
  ws.onmessage = (ev) => {
    const div = document.createElement("div");
    div.textContent = ev.data;
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  };
}
openWS();

// BroadcastChannel wiring to control hidden /client/ iframe
const controlChan = new BroadcastChannel("pipecat_control");
const vadChan = new BroadcastChannel("pipecat_vad");

micBtn.addEventListener("click", () => {
  controlChan.postMessage({ action: "toggleMic" });
});

vadChan.onmessage = (e) => {
  level = Math.min(1, e.data.level || 0);
};