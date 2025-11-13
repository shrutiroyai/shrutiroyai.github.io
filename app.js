
// Simple semantic-ish retrieval over kb.json with nice answers.
// NOTE: This is written so you can later plug in real embeddings if you want,
// but currently uses a lightweight keyword-based similarity so it runs everywhere with no extra downloads.

const messagesBox = document.getElementById("messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("user-input");

let KB = [];

// -------- Chat helpers --------
function addMessage(who, html, extraClass = "") {
  let div = document.createElement("div");
  div.className = "msg " + who + (extraClass ? " " + extraClass : "");
  div.innerHTML = `<div class="bubble">${html}</div>`;
  messagesBox.appendChild(div);
  messagesBox.scrollTop = messagesBox.scrollHeight;
  return div;
}

function typewriter(text, element, speed = 28, done = null) {
  let i = 0;
  function type() {
    if (i < text.length) {
      element.innerHTML = text.substring(0, i+1) + "<span class='cursor'>|</span>";
      i++;
      setTimeout(type, speed);
    } else {
      element.innerHTML = text;
      if (done) done();
    }
  }
  type();
}

function fadeReveal(text) {
  let div = addMessage("bot", text, "fade-reveal");
  return div;
}

// -------- Lightweight "semantic" similarity --------

// Basic tokenizer
const STOP = new Set("a,an,the,and,or,of,in,on,for,to,with,without,by,at,from,as,that,this,is,are,was,were,be,been,has,have,had,do,does,did,not,if,but,then,so,than,it,its,into,over,per,via,about,your,my,our,their,them,they,you,we,i".split(","));

function tokens(s){
  return (s || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g," ")
    .split(/\s+/)
    .filter(w => w && !STOP.has(w) && w.length>1);
}

// Simple synonym expansion to catch recruiter language
const SYNONYMS = {
  "llm": ["llm","language","gpt","foundation","large","rag","agent","mcp"],
  "causal": ["causal","uplift","counterfactual","bsts","experiment","ab","a/b","experiment","experimentation"],
  "production": ["production","prod","deploy","deployment","pipeline","mle","mlops"],
  "recommendation": ["reco","recommendation","personalization","ranking"],
  "pricing": ["pricing","promotion","discount","price"],
  "marketing": ["marketing","channel","campaign","crm"],
  "routing": ["route","routing","logistics","path","shuttle"]
};

function expandTokens(toks){
  let expanded = [];
  for(const t of toks){
    expanded.push(t);
    for(const [key, group] of Object.entries(SYNONYMS)){
      if(group.includes(t)){
        expanded.push(key);
      }
    }
  }
  return expanded;
}

// Build a simple bag-of-words index with IDF
let KB_INDEX = null;
function buildIndex(kb){
  const docs = kb.map((d,idx)=>({...d,__id:idx}));
  const vocab = new Map();
  const docTokens = [];
  docs.forEach(d=>{
    const text = (d.title + " " + (d.area||"") + " " + (d.tags||[]).join(" ") + " " + (d.summary||"") + " " + (d.details||""));
    const toks = expandTokens(tokens(text));
    docTokens.push(toks);
    toks.forEach(t=>{ if(!vocab.has(t)) vocab.set(t, vocab.size); });
  });
  const N = docs.length;
  const V = vocab.size;
  const df = new Float64Array(V);
  docTokens.forEach(toks=>{
    const seen = new Set();
    toks.forEach(t=>{
      const id = vocab.get(t);
      if(!seen.has(id)){ df[id]++; seen.add(id); }
    });
  });
  const idf = new Float64Array(V);
  for(let i=0;i<V;i++){
    idf[i] = Math.log((N+1)/(df[i]+1)) + 1;
  }
  const docVecs = docs.map((d,i)=>{
    const toks = docTokens[i];
    const tf = new Map();
    toks.forEach(t=> tf.set(t, (tf.get(t)||0)+1));
    const vec = new Float64Array(V);
    let norm = 0;
    tf.forEach((count, tok)=>{
      const j = vocab.get(tok);
      const w = (count/Math.sqrt(toks.length)) * idf[j];
      vec[j] = w;
      norm += w*w;
    });
    const inv = 1/Math.max(Math.sqrt(norm),1e-9);
    for(let j=0;j<V;j++) vec[j]*=inv;
    return {vec, ...d};
  });
  KB_INDEX = {vocab, idf, docs: docVecs};
}

function embedQuery(q){
  if(!KB_INDEX) return null;
  const toks = expandTokens(tokens(q));
  const V = KB_INDEX.vocab.size;
  const tf = new Map();
  toks.forEach(t=> tf.set(t, (tf.get(t)||0)+1));
  const vec = new Float64Array(V);
  let norm=0;
  tf.forEach((count, tok)=>{
    const j = KB_INDEX.vocab.get(tok);
    if(j === undefined) return;
    const w = (count/Math.sqrt(toks.length)) * KB_INDEX.idf[j];
    vec[j] = w;
    norm += w*w;
  });
  const inv = 1/Math.max(Math.sqrt(norm),1e-9);
  for(let j=0;j<V;j++) vec[j]*=inv;
  return vec;
}

function cosine(a,b){
  let s=0;
  for(let i=0;i<a.length;i++){ s += a[i]*b[i]; }
  return s;
}

function retrieveSemantic(query, k=3){
  if(!KB_INDEX) return [];
  const qv = embedQuery(query);
  if(!qv) return [];
  const scored = KB_INDEX.docs.map(d=>({score: cosine(qv, d.vec), doc:d}));
  scored.sort((a,b)=> b.score - a.score);
  return scored.slice(0,k);
}

// -------- Answer synthesis --------
function synthesizeAnswer(question, hits){
  if(!hits || hits.length === 0){
    return "I couldn’t quite match that to anything in my experience yet. You can try rephrasing, or ask about areas like pricing, experimentation, LLMs, causal inference, or recommendations.";
  }
  let intro = `Here’s the experience most relevant to “<strong>${question}</strong>”:`;
  let bullets = hits.map(h=>{
    const d = h.doc;
    return `<li><strong>${d.title}</strong> — ${d.summary}</li>`;
  }).join("");
  let closing = "If you’d like more detail, you can ask follow‑ups like “tell me more about that project” or “what were the business results?”.";
  return `${intro}<ul>${bullets}</ul>${closing}`;
}

// -------- Auto intro on load --------
window.addEventListener("load", () => {
  // Load KB
  fetch("kb.json")
    .then(r => r.json())
    .then(kb => {
      KB = kb;
      buildIndex(KB);
      addMessage("system", "Loaded Shruti’s experience. Ask about LLMs, causal inference, experimentation, pricing, recommendations, routing, or general profile.", "system");
    })
    .catch(err => {
      console.error("Failed to load kb.json", err);
      addMessage("system", "I couldn’t load my knowledge base. The chatbot might not answer accurately until this is fixed.", "system");
    });

  // Greeting + intro with typewriter + fade
  let msg1 = addMessage("bot", "");
  typewriter("Hi, I am Shruti 👋", msg1.querySelector(".bubble"), 40, () => {
    let first = "I love turning data and algorithms into real business impact.";
    let msg2 = addMessage("bot", "");
    typewriter(first, msg2.querySelector(".bubble"), 22, () => {
      let more = " With 10+ years of experience deploying ML systems and a Master’s in Data Science from USF, I’ve worked across experimentation, personalization, and production ML. Outside work, I enjoy cooking, reading, and travelling.";
      fadeReveal(more);
    });
  });
});

// -------- User interaction --------
form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if(!text) return;
  addMessage("user", text);
  input.value = "";

  // Retrieve relevant experience
  const hits = retrieveSemantic(text, 3);
  const ansHtml = synthesizeAnswer(text, hits);
  addMessage("bot", ansHtml);
});
