/**
 * Minimal JavaScript for the LLM Ops Platform web UI.
 * Handles SSE streaming and theme switching.
 */

// Theme toggle
function toggleTheme() {
  const html = document.documentElement;
  const current = html.getAttribute("data-theme");
  const next = current === "light" ? "dark" : "light";
  html.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
}

// Restore theme from localStorage
(function () {
  const saved = localStorage.getItem("theme");
  if (saved) {
    document.documentElement.setAttribute("data-theme", saved);
  }
})();

// Auto-scroll chat container to bottom
function scrollToBottom(selector) {
  const el = document.querySelector(selector);
  if (el) {
    el.scrollTop = el.scrollHeight;
  }
}

// htmx: auto-scroll after content swap in chat
document.addEventListener("htmx:afterSwap", function (event) {
  if (event.detail.target.id === "chat-messages") {
    scrollToBottom("#chat-messages");
  }
});

// htmx: handle SSE message events for streaming
document.addEventListener("htmx:sseMessage", function (event) {
  if (event.detail.target && event.detail.target.id === "chat-messages") {
    scrollToBottom("#chat-messages");
  }
});

// Auto-dismiss toast notifications
document.addEventListener("htmx:afterSwap", function (event) {
  if (event.detail.target.id === "toast-container") {
    const alerts = event.detail.target.querySelectorAll(".alert");
    alerts.forEach(function (alert) {
      setTimeout(function () {
        alert.remove();
      }, 5000);
    });
  }
});
