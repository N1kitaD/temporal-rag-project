"""
Four-Way Benchmark Evaluation — Phase 10
Auto-resumes from checkpoint if interrupted.
Delete results/evaluation_results.csv to start fresh.

Run with: python benchmark/run_evaluation.py
"""

import os
import sys
import json
import time
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import (
    load_pipeline,
    run_pipeline,
    run_plain_pipeline,
    run_naive_pipeline,
)

BENCHMARK_PATH = "data/benchmark/benchmark.json"
RESULTS_PATH   = "results/evaluation_results.csv"
TABLE_PATH     = "results/main_table.json"
TAXONOMY_PATH  = "results/failure_taxonomy.json"

# ── Answer checking ───────────────────────────────────────────────────────
def check_answer(system_answer, correct_answer):
    if not system_answer or not correct_answer:
        return False
    if "not found" in system_answer.lower():
        return False
    return correct_answer.lower() in system_answer.lower()


# ── Oracle pipeline ───────────────────────────────────────────────────────
def run_oracle_pipeline(question, true_type,
                        index, chunks, embed_model, groq_client):
    from retriever import retrieve
    from reranker  import adaptive_rerank
    from generator import generate_answer

    retrieved = retrieve(question, index, chunks, embed_model, top_k=20)
    p_dynamic = 1.0 if true_type == "DYNAMIC_CURRENT" else 0.0
    reranked  = adaptive_rerank(retrieved, p_dynamic_current=p_dynamic)
    answer    = generate_answer(question, reranked[:5], true_type, groq_client)
    return answer, reranked


# ── Groq call with retry ──────────────────────────────────────────────────
def safe_run(fn, *args, retries=3, **kwargs):
    """
    Wrap any pipeline call with automatic retry on rate limit errors.
    Waits 60s on first hit, 120s on second, 180s on third.
    """
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower() or "limit" in err.lower():
                wait = 60 * (attempt + 1)
                print(f"  [rate limit] waiting {wait}s before retry {attempt+1}/{retries}...")
                time.sleep(wait)
            else:
                print(f"  [error] {e}")
                raise
    raise RuntimeError(f"Failed after {retries} retries")


# ── Main evaluation loop ──────────────────────────────────────────────────
def run_evaluation():
    os.makedirs("results", exist_ok=True)

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    print(f"Loaded {len(benchmark)} benchmark questions")

    index, chunks, embed_model, groq_client = load_pipeline()

    # --- RESUME LOGIC ---
    results = []
    start_from = 1
    
    if os.path.exists(RESULTS_PATH):
        try:
            df_existing = pd.read_csv(RESULTS_PATH)
            # Filter out any rows that logged an ERROR during the crash
            df_clean = df_existing[df_existing["plain_answer"] != "ERROR"]
            if len(df_clean) > 0:
                results = df_clean.to_dict(orient="records")
                start_from = len(results) + 1
                print(f"--> Found clean checkpoint! Resuming automatically from question {start_from}...")
        except Exception as e:
            print(f"--> Could not read checkpoint ({e}), starting from scratch.")
            results = []
            start_from = 1
    # -------------------

    total = len(benchmark)

    print(f"\nRunning four configurations on {total} questions...")
    print("This takes approximately 30-40 minutes due to API rate limits.\n")

    for i, q in enumerate(benchmark, 1):
        # SKIP ALREADY PROCESSED QUESTIONS
        if i < start_from:
            continue
            
        qid      = q["question_id"]
        question  = q["question"]
        correct   = q["correct_answer"]
        true_type = q["true_type"]
        group     = q["group"]

        print(f"[{i:03d}/{total}] {question[:58]}")

        row = {
            "question_id":    qid,
            "question":       question,
            "correct_answer": correct,
            "true_type":      true_type,
            "group":          group,
        }

        # ── Config 1: Plain ───────────────────────────────────────────
        try:
            plain_ans, plain_docs = safe_run(
                run_plain_pipeline,
                question, index, chunks, embed_model, groq_client
            )
            row["plain_answer"]    = plain_ans
            row["plain_correct"]   = check_answer(plain_ans, correct)
            row["plain_top1_date"] = plain_docs[0]["date"][:10] if plain_docs else ""
        except Exception as e:
            print(f"  [FAIL plain] {e}")
            row["plain_answer"]    = "ERROR"
            row["plain_correct"]   = False
            row["plain_top1_date"] = ""
        time.sleep(8.5)

        # ── Config 2: Naive Penalty ───────────────────────────────────
        try:
            naive_ans, naive_docs = safe_run(
                run_naive_pipeline,
                question, index, chunks, embed_model, groq_client
            )
            row["naive_answer"]    = naive_ans
            row["naive_correct"]   = check_answer(naive_ans, correct)
            row["naive_top1_date"] = naive_docs[0]["date"][:10] if naive_docs else ""
        except Exception as e:
            print(f"  [FAIL naive] {e}")
            row["naive_answer"]    = "ERROR"
            row["naive_correct"]   = False
            row["naive_top1_date"] = ""
        time.sleep(8.5)

        # ── Config 3: Two-Stage ───────────────────────────────────────
        try:
            ts_ans, ts_log = safe_run(
                run_pipeline,
                question, index, chunks, embed_model, groq_client,
                verbose=False
            )
            row["ts_answer"]          = ts_ans
            row["ts_correct"]         = check_answer(ts_ans, correct)
            row["ts_classifier_type"] = ts_log["query_type"]
            row["ts_p_dynamic"]       = ts_log["p_dynamic"]
            row["ts_top1_date"]       = (ts_log["top5_after_rerank"][0]["date"][:10]
                                         if ts_log["top5_after_rerank"] else "")
            row["classifier_correct"] = (ts_log["query_type"] == true_type)
        except Exception as e:
            print(f"  [FAIL two-stage] {e}")
            row["ts_answer"]          = "ERROR"
            row["ts_correct"]         = False
            row["ts_classifier_type"] = "ERROR"
            row["ts_p_dynamic"]       = 0.0
            row["ts_top1_date"]       = ""
            row["classifier_correct"] = False
        time.sleep(8.5)

        # ── Config 4: Oracle ──────────────────────────────────────────
        try:
            oracle_ans, oracle_docs = safe_run(
                run_oracle_pipeline,
                question, true_type, index, chunks, embed_model, groq_client
            )
            row["oracle_answer"]    = oracle_ans
            row["oracle_correct"]   = check_answer(oracle_ans, correct)
            row["oracle_top1_date"] = oracle_docs[0]["date"][:10] if oracle_docs else ""
        except Exception as e:
            print(f"  [FAIL oracle] {e}")
            row["oracle_answer"]    = "ERROR"
            row["oracle_correct"]   = False
            row["oracle_top1_date"] = ""
        time.sleep(8.5)

        results.append(row)

        # Print row summary
        p = "✓" if row["plain_correct"]  else "✗"
        n = "✓" if row["naive_correct"]  else "✗"
        t = "✓" if row["ts_correct"]     else "✗"
        o = "✓" if row["oracle_correct"] else "✗"
        c = "✓" if row.get("classifier_correct") else "✗"
        
        print(f"       Plain={p} Naive={n} TwoStage={t} Oracle={o} Clf={c} | {true_type}")

        # Save checkpoint every 5 questions
        if len(results) % 5 == 0:
            df_temp = pd.DataFrame(results)
            df_temp.to_csv(RESULTS_PATH, index=False)
            done = len(results)
            print(f"  [checkpoint saved: {done}/{total} done]")

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nAll results saved to {RESULTS_PATH}")
    
    build_main_table(df)
    build_failure_taxonomy(df)


# ── Results tables ────────────────────────────────────────────────────────
def build_main_table(df):
    configs = {
        "Plain":         "plain_correct",
        "Naive Penalty": "naive_correct",
        "Two-Stage":     "ts_correct",
        "Oracle":        "oracle_correct",
    }

    type_rows = {}
    for label, subset in [
        ("Static (G1+G5)",          df[df["true_type"] == "STATIC"]),
        ("Dynamic-Current (G2+G4)", df[df["true_type"] == "DYNAMIC_CURRENT"]),
        ("Dynamic-Historical (G3)", df[df["true_type"] == "DYNAMIC_HISTORICAL"]),
        ("Overall",                 df),
    ]:
        n = len(subset)
        type_rows[label] = {"N": n}
        for cfg, col in configs.items():
            acc = subset[col].mean() * 100 if n > 0 else 0
            type_rows[label][cfg] = f"{acc:.1f}%"

    group_rows = {}
    group_names = {
        1: "G1 Static",
        2: "G2 Dynamic-Current",
        3: "G3 Dynamic-Historical",
        4: "G4 Undated-Dynamic",
        5: "G5 No-date Control",
    }
    for g, name in group_names.items():
        sub = df[df["group"] == g]
        n   = len(sub)
        group_rows[name] = {"N": n}
        for cfg, col in configs.items():
            acc = sub[col].mean() * 100 if n > 0 else 0
            group_rows[name][cfg] = f"{acc:.1f}%"

    clf_acc = df["classifier_correct"].mean() * 100
    per_type_clf = {}
    for t in ["STATIC", "DYNAMIC_CURRENT", "DYNAMIC_HISTORICAL"]:
        sub = df[df["true_type"] == t]
        per_type_clf[t] = (f"{sub['classifier_correct'].mean()*100:.1f}%"
                           if len(sub) > 0 else "N/A")

    table = {
        "by_type":            type_rows,
        "by_group":           group_rows,
        "classifier_overall": f"{clf_acc:.1f}%",
        "classifier_by_type": per_type_clf,
    }

    with open(TABLE_PATH, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*72}")
    print("MAIN RESULTS TABLE")
    print(f"{'='*72}")
    print(f"{'Category':<32} {'N':>4} {'Plain':>7} {'Naive':>7} "
          f"{'Two-Stage':>10} {'Oracle':>7}")
    print("-" * 72)
    for label, row in type_rows.items():
        print(f"{label:<32} {row['N']:>4} {row['Plain']:>7} "
              f"{row['Naive Penalty']:>7} {row['Two-Stage']:>10} {row['Oracle']:>7}")

    print(f"\nClassifier accuracy: {clf_acc:.1f}%")
    for t, acc in per_type_clf.items():
        print(f"  {t}: {acc}")
    print(f"\nSaved to {TABLE_PATH}")


def build_failure_taxonomy(df):
    wrong    = df[df["ts_correct"] == False]
    taxonomy = []

    for _, row in wrong.iterrows():
        if not row.get("classifier_correct", True):
            ftype  = "A"
            reason = (f"Classifier predicted {row['ts_classifier_type']} "
                      f"but true type is {row['true_type']}")
        elif row["plain_correct"] and not row["ts_correct"]:
            ftype  = "C"
            reason = "Plain correct but Two-Stage wrong — reranker regression"
        elif not row["oracle_correct"] and not row["ts_correct"]:
            ftype  = "E"
            reason = "Oracle also failed — corpus gap or generator failure"
        elif row["oracle_correct"] and not row["ts_correct"]:
            ftype  = "B"
            reason = "Oracle succeeded but Two-Stage failed — penalty or retriever error"
        else:
            ftype  = "D"
            reason = "Generator failure"

        taxonomy.append({
            "question_id":     row["question_id"],
            "question":        row["question"],
            "true_type":       row["true_type"],
            "group":           row["group"],
            "failure_type":    ftype,
            "reason":          reason,
            "plain_correct":   row["plain_correct"],
            "oracle_correct":  row["oracle_correct"],
            "classifier_pred": row.get("ts_classifier_type", ""),
            "ts_answer":       row["ts_answer"],
            "correct_answer":  row["correct_answer"],
        })

    type_counts = {}
    for t in taxonomy:
        ft = t["failure_type"]
        type_counts[ft] = type_counts.get(ft, 0) + 1

    type_labels = {
        "A": "Classifier error",
        "B": "Retriever/penalty miss",
        "C": "Reranker regression",
        "D": "Generator failure",
        "E": "Corpus gap",
    }

    with open(TAXONOMY_PATH, "w", encoding="utf-8") as f:
        json.dump({"failures": taxonomy, "counts": type_counts},
                  f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print("FAILURE TAXONOMY (Two-Stage system)")
    print(f"{'='*50}")
    total_wrong = len(taxonomy)
    for ft, count in sorted(type_counts.items()):
        pct = count / total_wrong * 100 if total_wrong > 0 else 0
        print(f"  Type {ft} ({type_labels.get(ft,'?')}): {count} ({pct:.0f}%)")
    
    print(f"  Total failures: {total_wrong}/{len(df)}")
    print(f"Saved to {TAXONOMY_PATH}")


if __name__ == "__main__":
    run_evaluation()