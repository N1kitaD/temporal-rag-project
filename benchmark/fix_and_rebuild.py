"""
Fix stale ground truth for q035 (Chief Minister of Tamil Nadu)
and rebuild result tables from existing evaluation_results.csv.

No API calls needed — works entirely from saved CSV data.

Run with: python benchmark/fix_and_rebuild.py
"""

import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

RESULTS_PATH  = "results/evaluation_results.csv"
TABLE_PATH    = "results/main_table.json"
TAXONOMY_PATH = "results/failure_taxonomy.json"
BENCHMARK_PATH = "data/benchmark/benchmark.json"

# ── Answer checking (same logic as run_evaluation.py) ────────────────────

def check_answer(system_answer, correct_answer):
    if not system_answer or not correct_answer:
        return False
    if "not found" in str(system_answer).lower():
        return False
    return correct_answer.lower() in str(system_answer).lower()


# ── Fix benchmark.json ────────────────────────────────────────────────────

def fix_benchmark():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    fixed = 0
    for q in benchmark:
        if q["question_id"] == "q035":
            print(f"Fixing q035: '{q['correct_answer']}' → 'C. Joseph Vijay'")
            q["correct_answer"]   = "C. Joseph Vijay"
            q["old_wrong_answer"] = "M. K. Stalin"
            q["notes"] = ("M. K. Stalin was CM until May 2026. "
                          "C. Joseph Vijay succeeded him after 2026 TN elections. "
                          "Benchmark ground truth updated post-evaluation.")
            fixed += 1

    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)

    print(f"benchmark.json updated ({fixed} question fixed)\n")


# ── Fix evaluation_results.csv ────────────────────────────────────────────

def fix_csv():
    df = pd.read_csv(RESULTS_PATH)

    print("Before fix — q035 correctness:")
    q35 = df[df["question_id"] == "q035"]
    for col in ["plain_correct", "naive_correct", "ts_correct", "oracle_correct"]:
        print(f"  {col}: {q35[col].values[0]}")

    # Update correct_answer column
    df.loc[df["question_id"] == "q035", "correct_answer"] = "C. Joseph Vijay"

    # Recheck all four configurations against new correct answer
    new_correct = "C. Joseph Vijay"
    for config_prefix in ["plain", "naive", "ts", "oracle"]:
        ans_col = f"{config_prefix}_answer"
        cor_col = f"{config_prefix}_correct"
        mask    = df["question_id"] == "q035"
        df.loc[mask, cor_col] = df.loc[mask, ans_col].apply(
            lambda x: check_answer(x, new_correct)
        )

    print("\nAfter fix — q035 correctness:")
    q35 = df[df["question_id"] == "q035"]
    for col in ["plain_correct", "naive_correct", "ts_correct", "oracle_correct"]:
        print(f"  {col}: {q35[col].values[0]}")

    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nevaluation_results.csv saved with corrected q035\n")
    return df


# ── Rebuild main table ────────────────────────────────────────────────────

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

    print(f"{'='*72}")
    print("UPDATED MAIN RESULTS TABLE")
    print(f"{'='*72}")
    print(f"{'Category':<32} {'N':>4} {'Plain':>7} {'Naive':>7} "
          f"{'Two-Stage':>10} {'Oracle':>7}")
    print("-"*72)
    for label, row in type_rows.items():
        print(f"{label:<32} {row['N']:>4} {row['Plain']:>7} "
              f"{row['Naive Penalty']:>7} {row['Two-Stage']:>10} {row['Oracle']:>7}")
    print(f"\nClassifier accuracy: {clf_acc:.1f}%")
    for t, acc in per_type_clf.items():
        print(f"  {t}: {acc}")
    print(f"\nSaved to {TABLE_PATH}")


# ── Rebuild failure taxonomy ──────────────────────────────────────────────

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
            "plain_correct":   bool(row["plain_correct"]),
            "oracle_correct":  bool(row["oracle_correct"]),
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
    print("UPDATED FAILURE TAXONOMY (Two-Stage)")
    print(f"{'='*50}")
    total_wrong = len(taxonomy)
    for ft, count in sorted(type_counts.items()):
        pct = count / total_wrong * 100 if total_wrong > 0 else 0
        print(f"  Type {ft} ({type_labels.get(ft,'?')}): {count} ({pct:.0f}%)")
    print(f"  Total failures: {total_wrong}/{len(df)}")
    print(f"Saved to {TAXONOMY_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("="*55)
    print("FIX AND REBUILD — q035 ground truth correction")
    print("="*55)
    print("No API calls — works from existing evaluation_results.csv\n")

    fix_benchmark()
    df = fix_csv()
    build_main_table(df)
    build_failure_taxonomy(df)

    print("\n" + "="*55)
    print("Done. All three output files updated:")
    print(f"  {BENCHMARK_PATH}")
    print(f"  {RESULTS_PATH}")
    print(f"  {TABLE_PATH}")
    print(f"  {TAXONOMY_PATH}")


if __name__ == "__main__":
    main()