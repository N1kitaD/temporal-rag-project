import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── model selection ────────────────────────────────────────────────────────
# llama-3.3-70b-versatile: best free model on Groq for classification tasks
# if you hit rate limits, swap to "llama3-8b-8192" (faster, slightly less accurate)
MODEL = "llama-3.3-70b-versatile"

# ── classification prompt ──────────────────────────────────────────────────
# Few-shot prompt: gives the model examples of each class before asking it
# to classify a new question. This is not training — the model already knows
# language. The examples just show it the exact format and reasoning we want.

CLASSIFY_PROMPT = """You are a query classifier for a question-answering system.
Classify the question into exactly one of three categories.

STATIC: The correct answer does not change over time. Historical facts,
scientific constants, completed events, geography, mathematical truths.
Even if the question mentions a specific date, the answer is permanently fixed.
Examples:
- "What is the boiling point of water?" → STATIC
- "When did India gain independence?" → STATIC
- "What is the Pythagorean theorem?" → STATIC
- "Who wrote the Indian national anthem?" → STATIC
- "When did World War II end?" → STATIC

DYNAMIC_CURRENT: The correct answer changes over time and the user needs
the most recent information. Words like "current", "now", "latest", "today",
"who is", "what is the" (about a role or record) often appear — but not always.
Examples:
- "Who is the current Chief Minister of Assam?" → DYNAMIC_CURRENT
- "Who is the Prime Minister of India?" → DYNAMIC_CURRENT
- "What is the latest version of Python?" → DYNAMIC_CURRENT
- "Who heads ISRO right now?" → DYNAMIC_CURRENT
- "Who is handling the opposition in the Assam assembly?" → DYNAMIC_CURRENT

DYNAMIC_HISTORICAL: The answer is time-sensitive but refers to a specific
past moment. The user wants what was true at a particular historical point,
NOT the latest update. A year or time period is usually mentioned.
Examples:
- "Who was the Chief Minister of Assam in 2018?" → DYNAMIC_HISTORICAL
- "Who was the Prime Minister of India in 2005?" → DYNAMIC_HISTORICAL
- "What was India's GDP in 2015?" → DYNAMIC_HISTORICAL
- "Who was the CEO of Apple when the iPhone launched?" → DYNAMIC_HISTORICAL

IMPORTANT RULE: If the question asks "who is" or "what is" without specifying
a past time, classify as DYNAMIC_CURRENT — not STATIC. Roles and records
change even if no temporal word appears.

Reply with ONLY one word: STATIC, DYNAMIC_CURRENT, or DYNAMIC_HISTORICAL.
No explanation. No punctuation. Just the category word.

Question: {question}"""

# ── confidence prompt ──────────────────────────────────────────────────────
# Asks the model how confident it is that this is a DYNAMIC_CURRENT question.
# This confidence score (0 to 1) is used in Phase 5 to scale the age penalty.
# High confidence → strong penalty. Low confidence → weak penalty. Zero → no penalty.

CONFIDENCE_PROMPT = """Rate your confidence that the question below requires
the MOST RECENT information to answer correctly.

Score 1.0 if: the answer definitely changes over time and the user needs
the latest version (current leaders, latest versions, current records).

Score 0.0 if: the answer is a permanent historical fact or scientific constant,
OR if the question asks about a specific past time period.

Score 0.5 if: you are genuinely uncertain whether the answer changes over time.

Reply with ONLY a decimal number between 0.0 and 1.0. Nothing else.

Question: {question}"""


def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Check your .env file.")
    return Groq(api_key=api_key)


def classify_query(question, client=None):
    """
    Classify a question as STATIC, DYNAMIC_CURRENT, or DYNAMIC_HISTORICAL.
    Returns the label string.
    """
    if client is None:
        client = get_client()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": CLASSIFY_PROMPT.format(question=question)
        }],
        max_tokens=10,        # we only need one word back
        temperature=0,        # temperature=0 means deterministic output
    )                         # same question always gets same answer

    raw = response.choices[0].message.content.strip().upper()

    # validate — handle cases where model adds punctuation or extra words
    valid_labels = {"STATIC", "DYNAMIC_CURRENT", "DYNAMIC_HISTORICAL"}
    if raw in valid_labels:
        return raw

    # partial match fallback
    for label in valid_labels:
        if label in raw:
            return label

    # safe default: if completely unparseable, treat as static
    # (better to miss a dynamic question than to wrongly penalize a static one)
    print(f"  [WARN] Unparseable classifier response: '{raw}' — defaulting to STATIC")
    return "STATIC"


def get_confidence(question, client=None):
    """
    Get a 0-1 confidence score: how sure are we this is DYNAMIC_CURRENT?
    This drives the adaptive penalty strength in the reranker.
    """
    if client is None:
        client = get_client()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": CONFIDENCE_PROMPT.format(question=question)
        }],
        max_tokens=10,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    try:
        score = float(raw)
        # clamp to valid range just in case
        return max(0.0, min(1.0, score))
    except ValueError:
        # if model returns something unparseable, use 0.5 (uncertain)
        print(f"  [WARN] Unparseable confidence response: '{raw}' — defaulting to 0.5")
        return 0.5


def classify_with_confidence(question, client=None):
    """
    Run both calls together and return label + confidence.
    This is the main function the pipeline will call.
    """
    if client is None:
        client = get_client()

    label = classify_query(question, client)
    confidence = get_confidence(question, client)

    return label, confidence


def test_classifier():
    """
    Run the classifier on known questions and verify outputs.
    This is your checkpoint — check the printed results make sense.
    """
    client = get_client()

    test_cases = [
        # (question, expected_label)
        ("What is the boiling point of water?",              "STATIC"),
        ("When did India gain independence?",                 "STATIC"),
        ("What is the Pythagorean theorem?",                  "STATIC"),
        ("Who is the current Chief Minister of Assam?",       "DYNAMIC_CURRENT"),
        ("Who is the Prime Minister of India?",               "DYNAMIC_CURRENT"),
        ("Who heads ISRO right now?",                         "DYNAMIC_CURRENT"),
        ("Who is handling the opposition in Assam assembly?", "DYNAMIC_CURRENT"),
        ("Who was the Chief Minister of Assam in 2018?",      "DYNAMIC_HISTORICAL"),
        ("Who was the Prime Minister of India in 2005?",      "DYNAMIC_HISTORICAL"),
        ("What was India's GDP in 2015?",                     "DYNAMIC_HISTORICAL"),
    ]

    print("="*65)
    print("CLASSIFIER TEST")
    print("="*65)

    correct = 0
    for question, expected in test_cases:
        label, confidence = classify_with_confidence(question, client)
        status = "✓" if label == expected else "✗"
        if label == expected:
            correct += 1
        print(f"{status} [{label:<20}] conf={confidence:.2f} | {question[:50]}")

    accuracy = correct / len(test_cases) * 100
    print(f"\nAccuracy: {correct}/{len(test_cases)} = {accuracy:.0f}%")

    if accuracy >= 80:
        print("Classifier is working well. Proceed to next step.")
    else:
        print("Accuracy below 80%. Review the prompt examples and retry.")


if __name__ == "__main__":
    test_classifier()