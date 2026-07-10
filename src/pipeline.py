import os
import json
import pickle
import time
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()

# import the four components we already built and tested individually
from retriever   import retrieve, load_index
from classifier  import classify_query, get_confidence, get_client
from reranker    import adaptive_rerank
from generator   import generate_answer, format_documents

INDEX_PATH = "data/corpus/faiss_index.pkl"


def load_pipeline():
    """
    Load all heavyweight components once at startup.
    Returns everything the pipeline function needs.
    Calling this once and reusing avoids reloading the model
    and index on every question — which would be very slow.
    """
    print("Loading pipeline components...")

    # load FAISS index + chunks + model name from disk
    index, chunks, model_name = load_index(INDEX_PATH)

    # load the same embedding model used to build the index
    print(f"Loading embedding model: {model_name}")
    embed_model = SentenceTransformer(model_name)

    # create one shared Groq client for all LLM calls
    groq_client = get_client()

    print("Pipeline ready.\n")
    return index, chunks, embed_model, groq_client


def run_pipeline(question, index, chunks, embed_model, groq_client,
                 top_k_retrieve=20, top_k_generate=5, verbose=True):
    """
    Full Temporal RAG pipeline: question in → answer out.

    Steps:
        1. Retrieve top-20 documents by cosine similarity (no time awareness)
        2. Classify query: STATIC / DYNAMIC_CURRENT / DYNAMIC_HISTORICAL
        3. Get classifier confidence score P
        4. Rerank top-20 using adaptive age penalty (scaled by P)
        5. Generate answer from top-5 reranked documents

    Returns:
        answer (str), log (dict with all intermediate values)
    """
    log = {
        "question": question,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── Step 1: Retrieve ──────────────────────────────────────────────
    retrieved = retrieve(question, index, chunks, embed_model,
                         top_k=top_k_retrieve)
    log["retrieved_count"] = len(retrieved)
    log["top3_before_rerank"] = [
        {
            "title": d["title"],
            "date":  d["date"][:10],
            "cosine_score": round(d["cosine_score"], 4),
            "topic_type": d["topic_type"],
        }
        for d in retrieved[:3]
    ]

    # ── Step 2 & 3: Classify + Confidence ────────────────────────────
    query_type = classify_query(question, groq_client)
    p_dynamic  = get_confidence(question, groq_client)

    log["query_type"]   = query_type
    log["p_dynamic"]    = round(p_dynamic, 4)

    # ── Step 4: Rerank ────────────────────────────────────────────────
    reranked = adaptive_rerank(retrieved, p_dynamic_current=p_dynamic)

    log["top5_after_rerank"] = [
        {
            "title":         d["title"],
            "date":          d["date"][:10],
            "cosine_score":  d["cosine_score"],
            "freshness_score": d["freshness_score"],
            "final_score":   d["final_score"],
            "topic_type":    d["topic_type"],
        }
        for d in reranked[:5]
    ]

    # ── Step 5: Generate ──────────────────────────────────────────────
    answer = generate_answer(question, reranked[:top_k_generate],
                             query_type, groq_client)
    log["answer"] = answer

    # ── Verbose output ────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*65}")
        print(f"QUESTION:   {question}")
        print(f"TYPE:       {query_type}  (P={p_dynamic:.2f})")
        print(f"\nTOP 3 BEFORE RERANK (pure cosine):")
        for i, d in enumerate(log["top3_before_rerank"], 1):
            print(f"  [{i}] {d['date']} | score={d['cosine_score']:.3f} | "
                  f"{d['title'][:45]}")
        print(f"\nTOP 5 AFTER RERANK (adaptive penalty applied):")
        for i, d in enumerate(log["top5_after_rerank"], 1):
            print(f"  [{i}] {d['date']} | cosine={d['cosine_score']:.3f} | "
                  f"fresh={d['freshness_score']:.3f} | "
                  f"final={d['final_score']:.3f} | "
                  f"{d['title'][:35]}")
        print(f"\nANSWER:     {answer}")
        print(f"{'='*65}")

    return answer, log


def run_plain_pipeline(question, index, chunks, embed_model, groq_client,
                       top_k_retrieve=20, top_k_generate=5):
    """
    Plain baseline: no classifier, no reranker.
    Just retrieve by cosine and generate from top-5.
    This is Configuration 1 in your four-way benchmark comparison.
    """
    retrieved = retrieve(question, index, chunks, embed_model,
                         top_k=top_k_retrieve)
    answer = generate_answer(question, retrieved[:top_k_generate],
                             "UNKNOWN", groq_client)
    return answer, retrieved


def run_naive_pipeline(question, index, chunks, embed_model, groq_client,
                       top_k_retrieve=20, top_k_generate=5):
    """
    Naive penalty baseline: apply full age penalty to every query (P=1.0).
    No classifier — treats all queries as DYNAMIC_CURRENT.
    This is Configuration 2 in your four-way benchmark comparison.
    Equivalent to Grofsky (2025)'s unconditional recency prior.
    """
    from reranker import naive_rerank
    retrieved = retrieve(question, index, chunks, embed_model,
                         top_k=top_k_retrieve)
    reranked  = naive_rerank(retrieved)
    answer    = generate_answer(question, reranked[:top_k_generate],
                                "DYNAMIC_CURRENT", groq_client)
    return answer, reranked


def test_pipeline():
    """
    Run 5 test questions through the full pipeline.
    These span all three query types so you can see the full
    classifier-reranker-generator chain working end to end.
    """
    index, chunks, embed_model, groq_client = load_pipeline()

    test_questions = [
        # (question, expected_type)
        ("What is the boiling point of water?",            "STATIC"),
        ("When did India gain independence?",               "STATIC"),
        ("Who is the current Chief Minister of Assam?",    "DYNAMIC_CURRENT"),
        ("Who is the Prime Minister of India?",            "DYNAMIC_CURRENT"),
        ("Who was the Chief Minister of Assam in 2016?",   "DYNAMIC_HISTORICAL"),
    ]

    logs = []

    for question, expected_type in test_questions:
        answer, log = run_pipeline(
            question, index, chunks, embed_model, groq_client,
            verbose=True
        )
        log["expected_type"] = expected_type
        log["type_correct"]  = (log["query_type"] == expected_type)
        logs.append(log)
        time.sleep(1)  # small pause between questions to stay within rate limits

    # summary
    print(f"\n{'='*65}")
    print("PIPELINE TEST SUMMARY")
    print(f"{'='*65}")
    correct_types = sum(1 for l in logs if l["type_correct"])
    print(f"Classifier accuracy: {correct_types}/{len(logs)}")
    for log in logs:
        status = "✓" if log["type_correct"] else "✗"
        print(f"  {status} [{log['query_type']:<20}] {log['question'][:50]}")
        print(f"      Answer: {log['answer'][:80]}")

    # save logs for inspection
    with open("results/pipeline_test_logs.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"\nFull logs saved to results/pipeline_test_logs.json")


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    test_pipeline()