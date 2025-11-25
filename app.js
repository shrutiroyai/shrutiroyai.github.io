const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const statusPill = document.getElementById("status-pill");
const chips = document.querySelectorAll(".prompt-chip");
const form = document.getElementById("chat-form");

let kb = [];
let ready = false;
let fallbackIndex = null;
let useModel = null;
let kbEmbeddings = [];

// -------- UI helpers --------
function setStatus(text, variant) {
  statusPill.textContent = text;
  statusPill.classList.remove("ok", "warn");
  if (variant) statusPill.classList.add(variant);
}

function setInputDisabled(disabled) {
  inputEl.disabled = disabled;
  sendBtn.disabled = disabled;
}

function appendMessage(node, sender = "bot") {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${sender}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.appendChild(node);
  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrapper;
}

function addBotMessage(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return appendMessage(div, "bot");
}

function addUserMessage(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return appendMessage(div, "user");
}

function addSystemMessage(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return appendMessage(div, "system");
}

function addThinking() {
  const dots = document.createElement("div");
  dots.className = "typing";
  dots.innerHTML = "<span></span><span></span><span></span>";
  return appendMessage(dots, "bot");
}

// -------- Fallback keyword index (if model fails) --------
const STOP = new Set(
  "a,an,the,and,or,of,in,on,for,to,with,without,by,at,from,as,that,this,is,are,was,were,be,been,has,have,had,do,does,did,not,if,but,then,so,than,it,its,into,over,per,via,about,your,my,our,their,them,they,you,we,i".split(
    ","
  )
);

function normalizeText(s = "") {
  return s
    .toLowerCase()
    .replace(/\bllms\b/g, "llm")
    .replace(/\bagents\b/g, "agent")
    .replace(/\brags\b/g, "rag");
}

function tokenize(text) {
  return normalizeText(text)
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w && w.length > 1 && !STOP.has(w));
}

function buildFallbackIndex(docs) {
  const vocab = new Map();
  const tokensPerDoc = docs.map((d) => {
    const text = `${d.title || ""} ${d.area || ""} ${(d.tags || []).join(" ")} ${
      d.summary || ""
    } ${d.details || ""}`;
    const toks = tokenize(text);
    toks.forEach((t) => {
      if (!vocab.has(t)) vocab.set(t, vocab.size);
    });
    return toks;
  });
  const df = new Float64Array(vocab.size);
  tokensPerDoc.forEach((toks) => {
    const seen = new Set();
    toks.forEach((t) => {
      const id = vocab.get(t);
      if (!seen.has(id)) {
        df[id]++;
        seen.add(id);
      }
    });
  });
  const idf = new Float64Array(vocab.size);
  const N = docs.length;
  for (let i = 0; i < idf.length; i++) idf[i] = Math.log((N + 1) / (df[i] + 1)) + 1;

  const docVecs = tokensPerDoc.map((toks, idx) => {
    const vec = new Float64Array(vocab.size);
    const tf = new Map();
    toks.forEach((t) => tf.set(t, (tf.get(t) || 0) + 1));
    let norm = 0;
    tf.forEach((count, tok) => {
      const j = vocab.get(tok);
      const w = (count / Math.sqrt(toks.length)) * idf[j];
      vec[j] = w;
      norm += w * w;
    });
    const inv = 1 / Math.max(Math.sqrt(norm), 1e-9);
    for (let j = 0; j < vec.length; j++) vec[j] *= inv;
    return { vec, item: docs[idx] };
  });

  return { vocab, idf, docs: docVecs };
}

function embedFallbackQuery(query) {
  if (!fallbackIndex) return null;
  const toks = tokenize(query);
  const vec = new Float64Array(fallbackIndex.vocab.size);
  const tf = new Map();
  toks.forEach((t) => tf.set(t, (tf.get(t) || 0) + 1));
  let norm = 0;
  tf.forEach((count, tok) => {
    const j = fallbackIndex.vocab.get(tok);
    if (j === undefined) return;
    const w = (count / Math.sqrt(toks.length)) * fallbackIndex.idf[j];
    vec[j] = w;
    norm += w * w;
  });
  const inv = 1 / Math.max(Math.sqrt(norm), 1e-9);
  for (let j = 0; j < vec.length; j++) vec[j] *= inv;
  return vec;
}

// -------- Semantic search helpers --------
function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1e-9);
}

function searchWithFallback(query, topK = 3) {
  if (!fallbackIndex) return [];
  const qv = embedFallbackQuery(query);
  if (!qv) return [];
  const scored = fallbackIndex.docs.map((d) => ({
    item: d.item,
    score: cosineSimilarity(qv, d.vec),
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK).map((s) => s.item);
}

async function buildEmbeddingsWithModel(model, docs) {
  const texts = docs.map((item) =>
    normalizeText(
      `${item.title || ""} ${item.area || ""} ${(item.tags || []).join(" ")} ${(item.tags || []).join(" ")} ${
        item.summary || ""
      } ${item.details || ""}`
    )
  );
  const tensor = await model.embed(texts);
  const arr = tensor.arraySync();
  tensor.dispose();
  return arr.map((vec, idx) => ({ item: docs[idx], embedding: vec }));
}

async function searchWithModel(query, topK = 3) {
  const q = normalizeText(query);
  const tensor = await useModel.embed([q]);
  const qv = tensor.arraySync()[0];
  tensor.dispose();
  const scored = kbEmbeddings.map((entry) => ({
    item: entry.item,
    score: cosineSimilarity(qv, entry.embedding),
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK).map((s) => s.item);
}

async function search(query, topK = 3) {
  if (useModel && kbEmbeddings.length) {
    try {
      return await searchWithModel(query, topK);
    } catch (err) {
      console.warn("Model search failed, falling back", err);
    }
  }
  return searchWithFallback(query, topK);
}

// -------- Render results --------
function buildResultsNode(results) {
  const wrapper = document.createElement("div");
  wrapper.className = "result-list";

  results.forEach((item) => {
    const card = document.createElement("div");
    card.className = "result-card";

    const head = document.createElement("div");
    head.className = "result-head";
    const title = document.createElement("div");
    title.className = "result-title";
    title.textContent = item.title || "Experience";
    head.appendChild(title);

    const metaText = [item.company, item.area].filter(Boolean).join(" • ");
    if (metaText) {
      const meta = document.createElement("div");
      meta.className = "result-meta";
      meta.textContent = metaText;
      head.appendChild(meta);
    }

    const summary = document.createElement("div");
    summary.className = "result-summary";
    summary.textContent = item.summary || "";

    const detail = document.createElement("div");
    detail.className = "result-detail";
    detail.textContent = item.details || "";

    const tagsRow = document.createElement("div");
    tagsRow.className = "tag-row";
    (item.tags || []).slice(0, 6).forEach((tag) => {
      const pill = document.createElement("span");
      pill.className = "tag-pill";
      pill.textContent = tag;
      tagsRow.appendChild(pill);
    });

    card.append(head, summary, detail, tagsRow);
    wrapper.appendChild(card);
  });

  return wrapper;
}

// -------- Chat handling --------
async function handleUserQuery(event) {
  event?.preventDefault();
  const query = inputEl.value.trim();
  if (!query) return;

  addUserMessage(query);
  inputEl.value = "";

  if (!ready) {
    addSystemMessage("Still loading the knowledge base. One sec...");
    return;
  }

  const thinking = addThinking();
  let results = [];
  try {
    results = await search(query, 1);
  } catch (err) {
    console.error("Search failed", err);
  } finally {
    thinking.remove();
  }

  if (!results.length) {
    addBotMessage("I could not find anything relevant. Try a broader topic or another phrasing.");
    return;
  }

  const node = buildResultsNode(results);
  appendMessage(node, "bot");
}

// -------- Init --------
async function init() {
  setInputDisabled(true);
  setStatus("Loading experience…");

  try {
    kb = await fetch("kb.json").then((res) => res.json());
  } catch (err) {
    console.error("Failed to load kb.json", err);
    addSystemMessage("Could not load experience data. Please refresh the page.");
    setStatus("Load failed", "warn");
    return;
  }

  // Build fallback index immediately so we can answer even if the model fails.
  fallbackIndex = buildFallbackIndex(kb);
  ready = true;
  setInputDisabled(false);
  addBotMessage("Hi, I'm Shruti 👋");
  addBotMessage("Ask about ML systems, personalization, causal inference, or experimentation. I'll surface the closest projects with full details.");
  setStatus("Loading model…");

  // Fire-and-forget model load; fallback search stays available.
  loadModel();
}

async function loadModel() {
  try {
    if (typeof tf === "undefined" || typeof use === "undefined") {
      throw new Error("TensorFlow.js not available");
    }
    await tf.ready();
    useModel = await use.load();
    kbEmbeddings = await buildEmbeddingsWithModel(useModel, kb);
    setStatus("Model ready", "ok");
  } catch (err) {
    console.warn("Model load failed, staying on fallback", err);
    useModel = null;
    setStatus("Using fallback search", "warn");
    addSystemMessage("Could not load the model; keeping fast keyword search instead.");
  }
}

// -------- Wiring --------
form.addEventListener("submit", handleUserQuery);
chips.forEach((chip) =>
  chip.addEventListener("click", () => {
    inputEl.value = chip.dataset.question || chip.textContent;
    inputEl.focus();
  })
);

init();
