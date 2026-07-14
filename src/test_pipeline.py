import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from pipeline import load_pipeline, run_pipeline

index, chunks, embed_model, groq_client = load_pipeline()

questions = [
    "Who is the current Chief Justice of India?",
    "Who is the Chief Minister of Assam?",
    "Who is the Prime Minister of UK?",
    "Who is the President of the USA?",
]

for q in questions:
    answer, log = run_pipeline(
        q, index, chunks, embed_model, groq_client, verbose=False
    )
    print(f"Q: {q}")
    print(f"A: {answer}")
    print()