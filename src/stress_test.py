"""
Stress-test script — run unscripted questions through the full pipeline
interactively. Type any question, press Enter, see the full output.
Type 'quit' to exit.

Run with: python src/stress_test.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pipeline import load_pipeline, run_pipeline

FAILURE_LOG_PATH = "results/stress_test_failures.json"


def main():
    os.makedirs("results", exist_ok=True)
    index, chunks, embed_model, groq_client = load_pipeline()

    failures = []
    question_count = 0

    print("\n" + "="*65)
    print("STRESS TEST — type any question and press Enter")
    print("Type 'quit' to exit and save failure log")
    print("After each answer, you will be asked to mark it:")
    print("  [y] correct   [n] wrong   [p] partial   [s] skip")
    print("="*65 + "\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        question_count += 1

        answer, log = run_pipeline(
            question, index, chunks, embed_model, groq_client,
            verbose=True
        )

        # ask for manual judgement
        verdict = ""
        while verdict not in ("y", "n", "p", "s"):
            verdict = input("\nCorrect? [y/n/p/s]: ").strip().lower()

        if verdict in ("n", "p"):
            # ask for failure type
            print("\nFailure type:")
            print("  A — Classifier wrong (wrong query type)")
            print("  B — Retriever miss (right docs not in top 20)")
            print("  C — Reranker error (right doc in top 20 but ranked too low)")
            print("  D — Generator error (right doc used but wrong answer written)")
            print("  E — Corpus gap (topic not in corpus at all)")
            ftype = ""
            while ftype not in ("a", "b", "c", "d", "e"):
                ftype = input("Type [A/B/C/D/E]: ").strip().lower()

            failures.append({
                "question": question,
                "verdict": verdict,
                "failure_type": ftype.upper(),
                "query_type_predicted": log["query_type"],
                "p_dynamic": log["p_dynamic"],
                "answer": answer,
                "top3_before": log["top3_before_rerank"],
                "top5_after": log["top5_after_rerank"],
            })
            print(f"  Logged as Type {ftype.upper()} failure.")

        elif verdict == "y":
            print("  Marked correct.")

        time.sleep(0.5)

    # save failure log
    with open(FAILURE_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*65}")
    print(f"STRESS TEST COMPLETE")
    print(f"Questions asked:  {question_count}")
    print(f"Failures logged:  {len(failures)}")
    if failures:
        types = {}
        for f in failures:
            t = f["failure_type"]
            types[t] = types.get(t, 0) + 1
        print(f"Failure breakdown:")
        for t, count in sorted(types.items()):
            labels = {
                "A": "Classifier wrong",
                "B": "Retriever miss",
                "C": "Reranker error",
                "D": "Generator error",
                "E": "Corpus gap"
            }
            print(f"  Type {t} ({labels.get(t,'?')}): {count}")
    print(f"Failure log saved to: {FAILURE_LOG_PATH}")
    print("="*65)


if __name__ == "__main__":
    main()