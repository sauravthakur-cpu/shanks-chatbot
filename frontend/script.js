const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");

const SOURCE_LABEL = {
  knowledge_base: { text: "Knowledge base", cls: "badge-kb" },
  web: { text: "Web", cls: "badge-web" },
  none: { text: "No source found", cls: "badge-web" },
};

function addMessage(role, text, sourceKey) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;

  if (role === "bot" && sourceKey && SOURCE_LABEL[sourceKey]) {
    const badge = document.createElement("span");
    badge.className = `badge source-badge ${SOURCE_LABEL[sourceKey].cls}`;
    badge.textContent = SOURCE_LABEL[sourceKey].text;
    el.appendChild(badge);
    el.appendChild(document.createElement("br"));
  }

  const textNode = document.createElement("span");
  textNode.textContent = text;
  el.appendChild(textNode);

  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

async function sendMessage(message) {
  addMessage("user", message);
  const pending = addMessage("bot pending", "Shanks is thinking...");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    if (!res.ok) {
      throw new Error(`Server returned ${res.status}`);
    }

    const data = await res.json();
    pending.remove();
    addMessage("bot", data.reply, data.source);
  } catch (err) {
    pending.remove();
    addMessage(
      "bot",
      "Something went wrong reaching the server. Make sure the backend " +
        "(uvicorn backend.main:app --reload) is running.",
      null
    );
    console.error(err);
  }
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const value = inputEl.value.trim();
  if (!value) return;
  inputEl.value = "";
  sendMessage(value);
});

addMessage(
  "bot",
  "Hey, I'm Shanks! Ask me anything about Amizone -- attendance, exams, fees, " +
    "timetables, or general college stuff.",
  null
);
