"""
Four-Way Benchmark Evaluation — Phase 10
Runs all 110 benchmark questions through four pipeline configurations
and produces the comparison tables for the paper.

Configurations:
  1. Plain         — pure cosine, no classifier, no penalty
  2. Naive Penalty — full penalty on every query (Grofsky baseline)
  3. Two-Stage     — your full system (classifier + adaptive penalty)
  4. Oracle        — uses ground truth labels instead of classifier

Run with: python benchmark/run_evaluation.py
Output:   results/evaluation_results.csv
          results/main_table.json
          results/failure_taxonomy.json
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
from reranker import adaptive_rerank, naive_rerank

BENCHMARK_PATH = "data/benchmark/benchmark.json"
RESULTS_PATH   = "results/evaluation_results.csv"
TABLE_PATH     = "results/main_table.json"
TAXONOMY_PATH  = "results/failure_taxonomy.json"


# ── Answer checking ───────────────────────────────────────────────────────

def check_answer(system_answer, correct_answer):
    """
    Check if the correct answer is present in the system's answer.
    Case-insensitive substring match — simple and consistent.
    """
    if not system_answer or not correct_answer:
        return False
    if "not found" in system_answer.lower():
        return False
    return correct_answer.lower() in system_answer.lower()


# ── Oracle pipeline ───────────────────────────────────────────────────────

def run_oracle_pipeline(question, true_type,
                        index, chunks, embed_model, groq_client):
    """
    Oracle: uses ground truth label instead of classifier prediction.
    P=1.0 for DYNAMIC_CURRENT, P=0.0 for everything else.
    Represents the upper bound — what Two-Stage would achieve with
    a perfect classifier.
    """
    from retriever import retrieve
    from reranker  import adaptive_rerank
    from generator import generate_answer

    retrieved = retrieve(question, index, chunks, embed_model, top_k=20)

    if true_type == "DYNAMIC_CURRENT":
        p_dynamic = 1.0
    else:
        p_dynamic = 0.0

    reranked = adaptive_rerank(retrieved, p_dynamic_current=p_dynamic)
    answer   = generate_answer(question, reranked[:5], true_type, groq_client)
    return answer, reranked


# ── Main evaluation loop ──────────────────────────────────────────────────

def run_evaluation():
    os.makedirs("results", exist_ok=True)

    # Load benchmark
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    print(f"Loaded {len(benchmark)} benchmark questions")

    # Load pipeline
    index, chunks, embed_model, groq_client = load_pipeline()

    results = []
    total   = len(benchmark)

    print(f"\nRunning four configurations on {total} questions...")
    print("This takes approximately 30-40 minutes due to API rate limits.\n")

    for i, q in enumerate(benchmark, 1):
        qid      = q["question_id"]
        question = q["question"]
        correct  = q["correct_answer"]
        true_type = q["true_type"]
        group    = q["group"]

        print(f"[{i:03d}/{total}] {question[:55]}")

        row = {
            "question_id":    qid,
            "question":       question,
            "correct_answer": correct,
            "true_type":      true_type,
            "group":          group,
        }

        # ── Config 1: Plain ───────────────────────────────────────────
        try:
            plain_ans, plain_docs = run_plain_pipeline(
                question, index, chunks, embed_model, groq_client
            )
            row["plain_answer"]  = plain_ans
            row["plain_correct"] = check_answer(plain_ans, correct)
            row["plain_top1_date"] = plain_docs[0]["date"][:10] if plain_docs else ""
        except Exception as e:
            print(f"  [ERROR plain] {e}")
            row["plain_answer"]  = "ERROR"
            row["plain_correct"] = False
            row["plain_top1_date"] = ""
        time.sleep(1.5)

        # ── Config 2: Naive Penalty ───────────────────────────────────
        try:
            naive_ans, naive_docs = run_naive_pipeline(
                question, index, chunks, embed_model, groq_client
            )
            row["naive_answer"]  = naive_ans
            row["naive_correct"] = check_answer(naive_ans, correct)
            row["naive_top1_date"] = naive_docs[0]["date"][:10] if naive_docs else ""
        except Exception as e:
            print(f"  [ERROR naive] {e}")
            row["naive_answer"]  = "ERROR"
            row["naive_correct"] = False
            row["naive_top1_date"] = ""
        time.sleep(1.5)

        # ── Config 3: Two-Stage ───────────────────────────────────────
        try:
            ts_ans, ts_log = run_pipeline(
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
            print(f"  [ERROR two-stage] {e}")
            row["ts_answer"]          = "ERROR"
            row["ts_correct"]         = False
            row["ts_classifier_type"] = "ERROR"
            row["ts_p_dynamic"]       = 0.0
            row["ts_top1_date"]       = ""
            row["classifier_correct"] = False
        time.sleep(1.5)

        # ── Config 4: Oracle ──────────────────────────────────────────
        try:
            oracle_ans, oracle_docs = run_oracle_pipeline(
                question, true_type, index, chunks, embed_model, groq_client
            )
            row["oracle_answer"]  = oracle_ans
            row["oracle_correct"] = check_answer(oracle_ans, correct)
            row["oracle_top1_date"] = oracle_docs[0]["date"][:10] if oracle_docs else ""
        except Exception as e:
            print(f"  [ERROR oracle] {e}")
            row["oracle_answer"]  = "ERROR"
            row["oracle_correct"] = False
            row["oracle_top1_date"] = ""
        time.sleep(1.5)

        results.append(row)

        # Print row summary
        p = "✓" if row["plain_correct"]  else "✗"
        n = "✓" if row["naive_correct"]  else "✗"
        t = "✓" if row["ts_correct"]     else "✗"
        o = "✓" if row["oracle_correct"] else "✗"
        c = "✓" if row.get("classifier_correct") else "✗"
        print(f"       Plain={p} Naive={n} TwoStage={t} Oracle={o} "
              f"Classifier={c} | type={true_type}")

        # Save incrementally every 10 questions
        if i % 10 == 0:
            df_temp = pd.DataFrame(results)
            df_temp.to_csv(RESULTS_PATH, index=False)
            print(f"  [saved checkpoint at {i} questions]")

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nFull results saved to {RESULTS_PATH}")

    # ── Build main comparison table ───────────────────────────────────
    build_main_table(df)
    build_failure_taxonomy(df)


def build_main_table(df):
    """
    Build the four-way accuracy table broken down by query type and group.
    This is Table 1 in your paper.
    """
    configs = {
        "Plain":        "plain_correct",
        "Naive Penalty":"naive_correct",
        "Two-Stage":    "ts_correct",
        "Oracle":       "oracle_correct",
    }

    # By true type
    type_rows = {}
    for label, col in [
        ("Static (Groups 1+5)",    df[df["true_type"] == "STATIC"]),
        ("Dynamic-Current (G2+4)", df[df["true_type"] == "DYNAMIC_CURRENT"]),
        ("Dynamic-Historical (G3)",df[df["true_type"] == "DYNAMIC_HISTORICAL"]),
        ("Overall",                df),
    ]:
        n = len(col)
        type_rows[label] = {"N": n}
        for cfg_name, cfg_col in configs.items():
            acc = col[cfg_col].mean() * 100 if n > 0 else 0
            type_rows[label][cfg_name] = f"{acc:.1f}%"

    # By group
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
        for cfg_name, cfg_col in configs.items():
            acc = sub[cfg_col].mean() * 100 if n > 0 else 0
            group_rows[name][cfg_name] = f"{acc:.1f}%"

    # Classifier accuracy
    clf_acc = df["classifier_correct"].mean() * 100
    per_type_clf = {}
    for t in ["STATIC", "DYNAMIC_CURRENT", "DYNAMIC_HISTORICAL"]:
        sub = df[df["true_type"] == t]
        per_type_clf[t] = f"{sub['classifier_correct'].mean()*100:.1f}%" if len(sub) > 0 else "N/A"

    table = {
        "by_type":           type_rows,
        "by_group":          group_rows,
        "classifier_overall": f"{clf_acc:.1f}%",
        "classifier_by_type": per_type_clf,
    }

    with open(TABLE_PATH, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)

    # Print to terminal
    print(f"\n{'='*70}")
    print("MAIN RESULTS TABLE")
    print(f"{'='*70}")
    print(f"{'Category':<30} {'N':>4} {'Plain':>8} {'Naive':>8} {'Two-Stage':>10} {'Oracle':>8}")
    print("-"*70)
    for label, row in type_rows.items():
        print(f"{label:<30} {row['N']:>4} {row['Plain']:>8} "
              f"{row['Naive Penalty']:>8} {row['Two-Stage']:>10} {row['Oracle']:>8}")
    print(f"\nClassifier accuracy: {clf_acc:.1f}%")
    for t, acc in per_type_clf.items():
        print(f"  {t}: {acc}")
    print(f"\nSaved to {TABLE_PATH}")


def build_failure_taxonomy(df):
    """
    Categorize Two-Stage system failures into five types.
    This becomes the failure taxonomy table in the paper.
    """
    wrong = df[df["ts_correct"] == False]

    taxonomy = []
    for _, row in wrong.iterrows():
        # Classify failure type automatically where possible
        if not row.get("classifier_correct", True):
            ftype = "A"
            reason = f"Classifier predicted {row['ts_classifier_type']} but true type is {row['true_type']}"
        elif row["plain_correct"] and not row["ts_correct"]:
            ftype = "C"
            reason = "Plain got it right but Two-Stage didn't — reranker demoted correct doc"
        elif not row["oracle_correct"] and not row["ts_correct"]:
            ftype = "E"
            reason = "Oracle also failed — corpus gap or generator failure"
        elif row["oracle_correct"] and not row["ts_correct"]:
            ftype = "B"
            reason = "Oracle succeeded but Two-Stage failed — classifier or penalty error"
        else:
            ftype = "D"
            reason = "Generator failure — correct doc likely retrieved but wrong answer written"

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

    # Count by type
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
        json.dump({"failures": taxonomy, "counts": type_counts}, f,
                  ensure_ascii=False, indent=2)

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