let useModel;
let kb = [];
let kbEmbeddings = [];

// ✅ Load KB + Model + Build Embeddings
async function init() {
  addMessage("🤖 Loading knowledge base and brain...", "bot");
  kb = await fetch("kb.json").then((res) => res.json());

  useModel = await use.load();
  addMessage("✅ Brain loaded. Indexing experience...", "bot");

  for (const item of kb) {
    const text = `${item.title} ${item.area} ${item.tags.join(" ")} ${item.summary} ${item.details}`;
    const embedding = await useModel.embed(text);
    kbEmbeddings.push({
      item,
      embedding: embedding.arraySync()[0],
    });
  }

  addMessage("✅ Ready! Ask me about Shruti's experience.", "bot");
}

// ✅ Chat UI Helpers
function addMessage(text, sender) {
  const messages = document.getElementById("messages");
  const div = document.createElement("div");
  div.className = sender;
  div.innerHTML = text;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

// ✅ Cosine Similarity
function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (normA * normB);
}

// ✅ Semantic Search
async function search(query, topK = 3) {
  const queryEmbedding = (await useModel.embed(query)).arraySync()[0];

  const scores = kbEmbeddings.map((entry) => {
    const sim = cosineSimilarity(queryEmbedding, entry.embedding);
    return { item: entry.item, score: sim };
  });

  return scores
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map((r) => r.item);
}

// ✅ Chat Handler with Realistic Delay
async function handleUserQuery() {
  const input = document.getElementById("user-input");
  const query = input.value.trim();
  if (!query) return;
  addMessage(query, "user");
  input.value = "";

  // ✅ Fake thinking delay
  await new Promise((resolve) => setTimeout(resolve, 700));
  addMessage("🤔 Let me think...", "bot");
  await new Promise((resolve) => setTimeout(resolve, 900));

  const results = await search(query);

  if (results.length === 0) {
    addMessage("I couldn't find anything relevant.", "bot");
    return;
  }

  let response = "✅ Here's what matches:<br><br>";
  results.forEach((r) => {
    response += `<b>${r.title}</b><br>• ${r.summary}<br>
    <button onclick='showDetails(\`${JSON.stringify(r.details)}\`)'>Show Details</button><br><br>`;
  });

  addMessage(response, "bot");
}

// ✅ Details Popup
function showDetails(details) {
  alert(details);
}

// ✅ Send Button & Enter Key
document.getElementById("send-btn").addEventListener("click", handleUserQuery);
document.getElementById("user-input").addEventListener("keypress", (e) => {
  if (e.key === "Enter") handleUserQuery();
});

// ✅ Start App
init();
