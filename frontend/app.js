// Temporal RAG demo - vanilla JS
// To connect to the real backend, replace callBackend() with a fetch to your endpoint.

const CONFIG_LABELS = {
  "two-stage": "Classifier + Adaptive Reranker + Generator",
  "plain": "Cosine Retrieval + Generator (no temporal awareness)",
  "naive": "Full Age Penalty Applied to All Queries (Grofsky baseline)"
};

const CONFIG_EXPLAIN = {
  "two-stage": {
    how: "Classifies the query into Static, Dynamic-Current, or Dynamic-Historical. Then applies the age penalty only if the query is Dynamic-Current, scaled by how confident the classifier is. Static and Historical queries are protected, their document order is unchanged.",
    why: "Freshness helps when the answer changes over time, but hurts when it does not. This system knows the difference."
  },
  "plain": {
    how: "Retrieves documents purely by semantic similarity (cosine score). No classifier. No age penalty. The newest and oldest documents are treated identically. Returns the top 5 most topically similar chunks.",
    why: "This is what a standard RAG system does. It works well for static questions but returns outdated answers for current events."
  },
  "naive": {
    how: "Applies the full age penalty to every query regardless of type, identical to Grofsky (2025). Newer documents always rank higher than older ones, even when the question is about a permanent fact.",
    why: "This improves current-event accuracy but damages accuracy on static questions by wrongly demoting old-but-correct documents. Our Two-Stage system fixes this regression."
  }
};

const CLASS_META = {
  "STATIC": {
    label: "STATIC",
    css: "badge-static",
    status: "Age penalty suppressed, document dates ignored"
  },
  "DYNAMIC_CURRENT": {
    label: "DYNAMIC-CURRENT",
    css: "badge-dynamic-current",
    status: "Full adaptive penalty applied"
  },
  "DYNAMIC_HISTORICAL": {
    label: "DYNAMIC-HISTORICAL",
    css: "badge-dynamic-historical",
    status: "Penalty suppressed, historical match sought"
  }
};

// ---------- DOM refs ----------
const $ = (id) => document.getElementById(id);
const input = $("question-input");
const submitBtn = $("submit-btn");
const badge = $("classification-badge");
const confidenceEl = $("confidence");
const penaltyStatusEl = $("penalty-status");
const answerEl = $("answer-text");
const sourceLineEl = $("source-line");
const beforeBody = $("before-body");
const afterBody = $("after-body");
const logToggle = $("log-toggle");
const logBody = $("log-body");
const logJson = $("log-json");
const copyBtn = $("copy-btn");
const configDesc = $("config-desc");
const configExplain = $("config-explain");

// ---------- Example buttons ----------
document.querySelectorAll(".ex-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    input.value = btn.dataset.q;
    input.focus();
  });
});

function updateConfigExplain(value) {
  const e = CONFIG_EXPLAIN[value];
  if (!e || !configExplain) return;
  configExplain.innerHTML =
    `<p><strong>HOW IT WORKS:</strong> ${e.how}</p>` +
    `<p><strong>WHY IT MATTERS:</strong> ${e.why}</p>`;
}

// ---------- Config radios ----------
document.querySelectorAll('input[name="config"]').forEach(r => {
  r.addEventListener("change", () => {
    configDesc.textContent = CONFIG_LABELS[r.value];
    updateConfigExplain(r.value);
  });
});

// ---------- Log toggle ----------
logToggle.addEventListener("click", () => {
  const hidden = logBody.classList.toggle("hidden");
  logToggle.textContent = (hidden ? "▶" : "▼") + " FULL DIAGNOSTIC LOG";
});

// ---------- Copy JSON ----------
copyBtn.addEventListener("click", () => {
  navigator.clipboard.writeText(logJson.textContent).then(() => {
    const t = copyBtn.textContent;
    copyBtn.textContent = "COPIED";
    setTimeout(() => copyBtn.textContent = t, 1200);
  });
});

// ---------- Submit ----------
submitBtn.addEventListener("click", onSubmit);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") onSubmit();
});

async function onSubmit() {
  const q = input.value.trim();
  if (!q) return;
  setLoading(true);
  try {
    const data = await callBackend(q, selectedConfig());
    render(data);
  } catch (err) {
    console.error(err);
    answerEl.classList.add("placeholder");
    answerEl.textContent = "Error contacting backend. See console.";
  } finally {
    setLoading(false);
  }
}

function selectedConfig() {
  const r = document.querySelector('input[name="config"]:checked');
  return r ? r.value : "two-stage";
}

// ---------- Backend call ----------
// Replace this with a real fetch, e.g.:
//   const res = await fetch("/api/query", { method: "POST", headers: {"Content-Type":"application/json"},
//     body: JSON.stringify({ question, config }) });
//   return await res.json();
async function callBackend(question, config) {
  const res = await fetch("http://localhost:8000/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, config })
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Server error");
  }
  return await res.json();
}

// ---------- Rendering ----------
function render(d) {
  // classification
  const meta = CLASS_META[d.query_type] || CLASS_META.STATIC;
  badge.className = "badge " + meta.css;
  badge.textContent = meta.label;
  confidenceEl.textContent = "Confidence: P = " + Number(d.p_dynamic).toFixed(2);
  penaltyStatusEl.textContent = d.penalty_status || meta.status;

  // answer
  answerEl.classList.remove("placeholder");
  answerEl.textContent = d.answer || "-";
  sourceLineEl.textContent = `Source: ${d.source_title} | Date: ${d.source_date}`;

  // tables
  const beforeIndex = new Map();
  d.top5_before_rerank.forEach(r => beforeIndex.set(keyOf(r), r.rank));

  renderTableRows(beforeBody, d.top5_before_rerank, () => "");
  renderTableRows(afterBody, d.top5_after_rerank, (row) => {
    const prev = beforeIndex.get(keyOf(row));
    if (prev == null) return "";
    if (prev > row.rank) return ' <span class="arrow">↑</span>';
    if (prev < row.rank) return ' <span class="arrow">↓</span>';
    return "";
  });

  // log
  logJson.textContent = JSON.stringify(d.full_log || d, null, 2);
}

function keyOf(row) { return row.title + "|" + row.date; }

function renderTableRows(tbody, rows, arrowFn) {
  tbody.innerHTML = "";
  rows.forEach(row => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="rank">${row.rank}${arrowFn(row)}</td>
      <td class="title">${truncate(row.title, 30)}</td>
      <td class="date">${row.date}</td>
      <td class="score">${Number(row.score).toFixed(3)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function truncate(s, n) {
  if (!s) return "";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function emptyTable(tbody) {
  tbody.innerHTML = "";
  for (let i = 0; i < 5; i++) {
    const tr = document.createElement("tr");
    tr.className = "empty-row";
    tr.innerHTML = `<td class="rank">${i+1}</td><td></td><td></td><td></td>`;
    tbody.appendChild(tr);
  }
}

// ---------- Loading state ----------
function setLoading(on) {
  if (on) {
    submitBtn.classList.add("processing");
    submitBtn.textContent = "PROCESSING...";
    answerEl.classList.add("placeholder");
    answerEl.textContent = "Retrieving and reasoning...";
    sourceLineEl.textContent = "";
    emptyTable(beforeBody);
    emptyTable(afterBody);
    badge.className = "badge badge-neutral";
    badge.textContent = "PROCESSING";
    confidenceEl.textContent = "Confidence: P = -";
    penaltyStatusEl.textContent = "-";
  } else {
    submitBtn.classList.remove("processing");
    submitBtn.textContent = "SUBMIT QUESTION";
  }
}

// ---------- Init: load sample data ----------
(function init() {
  configDesc.textContent = CONFIG_LABELS["two-stage"];
  emptyTable(beforeBody);
  emptyTable(afterBody);
  if (window.SAMPLE_RESPONSE) {
    input.value = window.SAMPLE_RESPONSE.question;
    render(window.SAMPLE_RESPONSE);
  }
})();
