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

// Chat: toggle send/cancel buttons and disable input during generation
function showCancelBtn() {
  document.getElementById("send-btn").classList.add("hidden");
  document.getElementById("cancel-btn").classList.remove("hidden");
  var input = document.querySelector("#chat-form input[name='message']");
  if (input) {
    input.disabled = true;
  }
}

function showSendBtn() {
  document.getElementById("send-btn").classList.remove("hidden");
  document.getElementById("cancel-btn").classList.add("hidden");
  var input = document.querySelector("#chat-form input[name='message']");
  if (input) {
    input.disabled = false;
    input.focus();
  }
}

var _chatEventSource = null;

function cancelChat() {
  // Close SSE connection if active
  if (_chatEventSource) {
    _chatEventSource.close();
    _chatEventSource = null;
  }
  htmx.trigger(document.getElementById("chat-form"), "htmx:abort");
  showSendBtn();
}

// SSE streaming chat handler
function sendChatStreaming(form) {
  var messageInput = form.querySelector("input[name='message']");
  var message = messageInput ? messageInput.value.trim() : "";
  if (!message) return;

  var convId = document.getElementById("conversation-id");
  var conversationId = convId ? convId.value : "";

  var url =
    "/web/chat/stream?message=" +
    encodeURIComponent(message) +
    "&conversation_id=" +
    encodeURIComponent(conversationId);

  showCancelBtn();

  // Clear placeholder text
  var chatMessages = document.getElementById("chat-messages");
  var placeholder = chatMessages.querySelector(".text-center.text-base-content\\/50");
  if (placeholder) placeholder.remove();

  _chatEventSource = new EventSource(url);

  _chatEventSource.addEventListener("user-message", function (e) {
    var html = JSON.parse(e.data);
    chatMessages.insertAdjacentHTML("beforeend", html);
    scrollToBottom("#chat-messages");
  });

  _chatEventSource.addEventListener("agent-step", function (e) {
    var html = JSON.parse(e.data);
    chatMessages.insertAdjacentHTML("beforeend", html);
    scrollToBottom("#chat-messages");
  });

  _chatEventSource.addEventListener("agent-answer", function (e) {
    var html = JSON.parse(e.data);
    // Remove step indicators (they were temporary progress display)
    chatMessages.insertAdjacentHTML("beforeend", html);
    scrollToBottom("#chat-messages");
  });

  // Token-level streaming events
  _chatEventSource.addEventListener("answer-start", function (e) {
    var html = JSON.parse(e.data);
    chatMessages.insertAdjacentHTML("beforeend", html);
    scrollToBottom("#chat-messages");
  });

  _chatEventSource.addEventListener("answer-chunk", function (e) {
    var text = JSON.parse(e.data);
    var target = document.getElementById("streaming-answer-content");
    if (target) {
      target.textContent += text;
    }
    scrollToBottom("#chat-messages");
  });

  _chatEventSource.addEventListener("answer-end", function (e) {
    var html = JSON.parse(e.data);
    var answer = document.getElementById("streaming-answer");
    if (answer) {
      answer.insertAdjacentHTML("beforeend", html);
      answer.removeAttribute("id");
    } else {
      chatMessages.insertAdjacentHTML("beforeend", html);
    }
    var content = document.getElementById("streaming-answer-content");
    if (content) content.removeAttribute("id");
    scrollToBottom("#chat-messages");
  });

  _chatEventSource.addEventListener("conversation-id", function (e) {
    var data = JSON.parse(JSON.parse(e.data));
    if (data.id) {
      _lastConversationId = data.id;
      if (convId) convId.value = data.id;
    }
  });

  _chatEventSource.addEventListener("error-event", function (e) {
    var data = JSON.parse(JSON.parse(e.data));
    var msg = data.error || "An error occurred";
    chatMessages.insertAdjacentHTML(
      "beforeend",
      '<div class="alert alert-error mt-2"><span>' + msg + "</span></div>"
    );
    scrollToBottom("#chat-messages");
  });

  _chatEventSource.addEventListener("done", function () {
    _chatEventSource.close();
    _chatEventSource = null;
    showSendBtn();
    messageInput.value = "";
    // Refresh conversation sidebar
    var sidebar = document.getElementById("conversation-list");
    if (sidebar && typeof htmx !== "undefined") {
      htmx.ajax("GET", "/web/chat/conversations", {
        target: "#conversation-list",
        swap: "innerHTML",
      });
    }
  });

  _chatEventSource.onerror = function () {
    _chatEventSource.close();
    _chatEventSource = null;
    showSendBtn();
  };
}

// Preserve conversation_id across form resets
var _lastConversationId = "";
function restoreConversationId() {
  var el = document.getElementById("conversation-id");
  if (el && _lastConversationId) {
    el.value = _lastConversationId;
  }
}

// Update conversation_id when a new one arrives in response
document.addEventListener("htmx:afterSwap", function (event) {
  if (event.detail.target.id === "chat-messages") {
    var el = document.getElementById("conversation-id");
    if (el && el.value) {
      _lastConversationId = el.value;
    }
  }
});

// Expand / collapse truncated table cells
function toggleExpand(el) {
  if (el.classList.contains("truncate")) {
    el.classList.remove("truncate", "max-w-xs");
    el.classList.add("whitespace-pre-wrap", "max-w-2xl");
  } else {
    el.classList.add("truncate", "max-w-xs");
    el.classList.remove("whitespace-pre-wrap", "max-w-2xl");
  }
}

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
