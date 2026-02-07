const chat = document.getElementById("chat");
const form = document.getElementById("chat-form");
const input = document.getElementById("message-input");
const statusEl = document.getElementById("status");

let sessionId = null;

function addBubble(text, role) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  bubble.textContent = text;
  chat.appendChild(bubble);
  chat.scrollTop = chat.scrollHeight;
}

async function sendMessage(message) {
  statusEl.textContent = "Status: sending...";
  const payload = { message };
  if (sessionId) {
    payload.session_id = sessionId;
  }

  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    statusEl.textContent = "Status: error";
    addBubble("Something went wrong. Please try again.", "bot");
    return;
  }

  const data = await response.json();
  sessionId = data.session_id;
  addBubble(data.response, "bot");
  statusEl.textContent = data.escalated
    ? `Status: escalated (${data.escalation_reason})`
    : "Status: resolved";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) {
    return;
  }
  addBubble(message, "user");
  input.value = "";
  await sendMessage(message);
});
