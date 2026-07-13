# Temporal RAG, Demo Interface

A single-page, dependency-free web UI for demonstrating a Temporal
Retrieval-Augmented Generation system to a research guide and in
presentations.

## Running

Just open `index.html` in any modern browser (double-click). No build
step, no server required. The page auto-loads a sample response so the
interface is populated on first open.

Files:

- `index.html`, page structure
- `styles.css`, neo-brutalist styling
- `app.js`, all interactivity and rendering
- `sample_data.js`, hardcoded example response (DYNAMIC_CURRENT)
- `README.md`, this file

## Connecting to your Python pipeline

Open `app.js` and replace the `callBackend(question, config)` function.
The current implementation returns a locally-mocked response after a
short delay. Replace it with a real `fetch` call to your API, e.g.:

```js
async function callBackend(question, config) {
  const res = await fetch("http://localhost:8000/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, config })
  });
  return await res.json();
}
```

The response JSON must match the schema described in the project spec
(fields: `question`, `query_type`, `p_dynamic`, `penalty_status`,
`answer`, `source_title`, `source_date`, `top5_before_rerank`,
`top5_after_rerank`, `full_log`). `query_type` must be one of
`STATIC`, `DYNAMIC_CURRENT`, `DYNAMIC_HISTORICAL`.

## Output panels

1. **Query Classification**, the classifier's predicted query type, its
   confidence `P`, and the resulting penalty policy.
2. **Answer**, the generated answer plus the title and date of the
   top-ranked source document used to produce it.
3. **Before Rerank**, the top-5 candidate documents by pure cosine
   similarity, with rank, title, date, and score.
4. **After Rerank**, the same 5 documents (or overlapping set) after
   the adaptive age penalty is applied. Score is the final score.
5. **Full Diagnostic Log**, collapsible JSON block with all intermediate
   values (cosine scores, freshness scores, final scores, `P`, query
   type, timestamps, reranker parameters). A `COPY` button copies the
   JSON to the clipboard.

## Rank-change arrows

Each row in the *After Rerank* table is matched to its *Before Rerank*
counterpart by the composite key `title | date`. If the rank decreased
(document moved up) an `↑` is shown next to the rank number; if it
increased (moved down) a `↓` is shown; unchanged rows show nothing.
Documents that appear only after rerank show no arrow.

## Configuration selector

The three radio options (`Two-Stage`, `Plain Baseline`, `Naive Penalty`)
change only the descriptive line displayed below them. The selected
config value is passed to `callBackend()` as the second argument; use it
server-side to switch pipelines.
