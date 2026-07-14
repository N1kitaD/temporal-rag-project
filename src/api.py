"""
FastAPI backend for Temporal RAG System.
Connects the Python pipeline to the frontend interface.

Run with: python src/api.py
Then open frontend/index.html in your browser.
The server runs at http://localhost:8000
"""

import os
import sys
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))

from pipeline import (
    load_pipeline,
    run_pipeline,
    run_plain_pipeline,
    run_naive_pipeline,
)

# ── App setup ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Temporal RAG API",
    description="Freshness-Aware Retrieval-Augmented Generation System",
    version="1.0.0",
)

# CORS — allows the frontend (opened as a local file) to call this server
# Without this, browsers block requests from file:// to localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load pipeline once at startup ─────────────────────────────────────────

print("Loading pipeline components at startup...")
index, chunks, embed_model, groq_client = load_pipeline()
print("API ready.\n")


# ── Request / Response models ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    config: str = "two-stage"


# ── Helper: penalty status text ───────────────────────────────────────────

def get_penalty_status(query_type: str, config: str) -> str:
    if config == "plain":
        return "No penalty — plain cosine retrieval"
    if config == "naive":
        return "Full age penalty applied unconditionally (Grofsky baseline)"
    if query_type == "STATIC":
        return "Age penalty suppressed — document dates ignored"
    if query_type == "DYNAMIC_CURRENT":
        return "Full adaptive penalty applied"
    if query_type == "DYNAMIC_HISTORICAL":
        return "Penalty suppressed — historical match sought"
    return "-"


# ── Helper: format top-5 for frontend ────────────────────────────────────

def format_top5_before(docs):
    rows = []
    for i, doc in enumerate(docs[:5], 1):
        rows.append({
            "rank":  i,
            "title": doc.get("title", "")[:50],
            "date":  doc.get("date",  "")[:10],
            "score": round(doc.get("cosine_score", 0), 3),
        })
    return rows


def format_top5_after(docs):
    rows = []
    for i, doc in enumerate(docs[:5], 1):
        rows.append({
            "rank":  i,
            "title": doc.get("title", "")[:50],
            "date":  doc.get("date",  "")[:10],
            "score": round(doc.get("final_score",
                           doc.get("cosine_score", 0)), 3),
        })
    return rows


# ── Main query endpoint ───────────────────────────────────────────────────

@app.post("/query")
async def query(request: QueryRequest):
    question = request.question.strip()
    config   = request.config.strip().lower()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if config not in ("two-stage", "plain", "naive"):
        config = "two-stage"

    print(f"\n[{time.strftime('%H:%M:%S')}] {question} | config={config}")

    try:
        if config == "plain":
            answer, retrieved = run_plain_pipeline(
                question, index, chunks, embed_model, groq_client
            )
            query_type  = "UNKNOWN"
            p_dynamic   = 0.0
            before_docs = retrieved[:5]
            after_docs  = retrieved[:5]
            log = {"config": "plain", "question": question, "answer": answer}

        elif config == "naive":
            answer, reranked = run_naive_pipeline(
                question, index, chunks, embed_model, groq_client
            )
            query_type  = "DYNAMIC_CURRENT"
            p_dynamic   = 1.0
            before_docs = sorted(reranked[:5],
                                 key=lambda x: x.get("cosine_score", 0),
                                 reverse=True)
            after_docs  = reranked[:5]
            log = {"config": "naive", "question": question,
                   "p_dynamic": 1.0, "answer": answer}

        else:
            answer, log = run_pipeline(
                question, index, chunks, embed_model, groq_client,
                verbose=True
            )
            query_type = log["query_type"]
            p_dynamic  = log["p_dynamic"]

            before_docs = [
                {
                    "title":        d["title"],
                    "date":         d["date"],
                    "cosine_score": d["cosine_score"],
                    "final_score":  d["cosine_score"],
                }
                for d in log["top3_before_rerank"]
            ]
            after_docs = [
                {
                    "title":        d["title"],
                    "date":         d["date"],
                    "cosine_score": d["cosine_score"],
                    "final_score":  d["final_score"],
                }
                for d in log["top5_after_rerank"]
            ]

        top1         = after_docs[0] if after_docs else {}
        source_title = top1.get("title", "-")
        source_date  = top1.get("date",  "-")[:10]

        response = {
            "question":           question,
            "query_type":         query_type,
            "p_dynamic":          round(p_dynamic, 4),
            "penalty_status":     get_penalty_status(query_type, config),
            "answer":             answer,
            "source_title":       source_title,
            "source_date":        source_date,
            "top5_before_rerank": format_top5_before(before_docs),
            "top5_after_rerank":  format_top5_after(after_docs),
            "full_log":           log,
        }

        print(f"  type={query_type} p={p_dynamic:.2f} | {answer[:60]}")
        return response

    except Exception as e:
        print(f"  [ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Health check ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "chunks_loaded": len(chunks),
        "model":         "all-MiniLM-L6-v2",
    }


# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("TEMPORAL RAG API SERVER")
    print("  Server: http://localhost:8000")
    print("  Health: http://localhost:8000/health")
    print("  Docs:   http://localhost:8000/docs")
    print("="*55 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")