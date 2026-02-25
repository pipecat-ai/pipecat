const isDark = document.documentElement.classList.contains("dark");
const iconColor = isDark ? "white" : "black";
const buttonIcon = `data:image/svg+xml,${encodeURIComponent(`<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M3.3088 5.05615C3.64682 4.92779 4.02833 5.02411 4.26653 5.29797L7.36884 8.86461H16.6312L19.7335 5.29797C19.9717 5.02411 20.3532 4.92779 20.6912 5.05615C21.0292 5.18452 21.253 5.51072 21.253 5.87504V13.75H24V15.5H19.5181V8.19909L17.6762 10.3167C17.5115 10.506 17.2738 10.6146 17.0241 10.6146H6.9759C6.72616 10.6146 6.48854 10.506 6.32383 10.3167L4.48193 8.19909V15.5H0V13.75H2.74699V5.87504C2.74699 5.51072 2.97078 5.18452 3.3088 5.05615Z" fill="${iconColor}"/><path d="M19.5181 17.25H24V19H19.5181V17.25Z" fill="${iconColor}"/><path d="M0 17.25H4.48193V19H0V17.25Z" fill="${iconColor}"/><path d="M9.25301 14.3333C9.25301 14.9777 8.73517 15.5 8.09639 15.5C7.4576 15.5 6.93976 14.9777 6.93976 14.3333C6.93976 13.689 7.4576 13.1667 8.09639 13.1667C8.73517 13.1667 9.25301 13.689 9.25301 14.3333Z" fill="${iconColor}"/><path d="M17.0602 14.3333C17.0602 14.9777 16.5424 15.5 15.9036 15.5C15.2648 15.5 14.747 14.9777 14.747 14.3333C14.747 13.689 15.2648 13.1667 15.9036 13.1667C16.5424 13.1667 17.0602 13.689 17.0602 14.3333Z" fill="${iconColor}"/></svg>`)}`;

const script = document.createElement("script");
script.src = "https://widget.kapa.ai/kapa-widget.bundle.js";
script.async = true;
script.setAttribute("data-website-id", "b1959adf-0d0a-425d-8419-5b7ff6edf937");
script.setAttribute("data-project-name", "Pipecat");
script.setAttribute("data-project-color", "#4f46e5");
script.setAttribute("data-project-logo", "https://docs.pipecat.ai/favicon.svg");
script.setAttribute(
  "data-modal-disclaimer",
  "This is a custom LLM trained on Pipecat's knowledge base to provide you with efficient access to Pipecat's capabilities and APIs. However, it may not always understand the entire context of your query to produce an accurate answer. When in doubt, please refer to our documentation or reach out on Discord.",
);
script.setAttribute("data-button-image", buttonIcon);
script.setAttribute("data-mcp-enabled", "true");
script.setAttribute("data-mcp-server-url", "https://daily-docs.mcp.kapa.ai");

if (isDark) {
  // Button
  script.setAttribute("data-button-bg-color", "#1f2937");
  script.setAttribute("data-button-text-color", "#f3f4f6");
  script.setAttribute("data-button-border", "1px solid #374151");
  script.setAttribute("data-button-box-shadow", "0 0 10px rgba(0, 0, 0, 0.3)");
  script.setAttribute("data-button-text-shadow", "none");
  script.setAttribute("data-button-hover-bg-color", "#374151");
  script.setAttribute("data-button-hover-text-color", "#f3f4f6");
} else {
  // Button
  script.setAttribute("data-button-bg-color", "#ffffff");
  script.setAttribute("data-button-text-color", "#000000");
  script.setAttribute("data-button-border", "1px solid #e5e7eb");
  script.setAttribute("data-button-box-shadow", "0 0 10px rgba(0, 0, 0, 0.1)");
  script.setAttribute("data-button-text-shadow", "none");
  script.setAttribute("data-button-hover-bg-color", "#f3f4f6");
  script.setAttribute("data-button-hover-text-color", "#000000");
}

document.head.appendChild(script);
