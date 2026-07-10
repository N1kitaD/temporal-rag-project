import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

# ── generation prompt ──────────────────────────────────────────────────────
#
# Three things this prompt must enforce:
#
# 1. "ONLY the documents below" — stops the model answering from its own
#    training memory. Without this, even if your retriever returns a wrong
#    document, the model might still give the right answer from memory,
#    which would make your retrieval accuracy measurements meaningless.
#
# 2. Date-aware instruction per query type — the generator gets a second
#    chance to catch temporal errors even if the reranker missed one.
#    For DYNAMIC_CURRENT, it's told to prefer the most recent document.
#    For DYNAMIC_HISTORICAL, it's told to look for documents from the
#    relevant time period. For STATIC, dates are irrelevant.
#
# 3. "Not found" fallback — without this, LLMs hallucinate a plausible
#    answer rather than admitting the documents don't cover the question.
#    Hallucinated answers would corrupt your accuracy measurements silently.

GENERATION_PROMPT = """Answer the question using ONLY the information in the documents provided below.
Do not use any knowledge from your training data. Base your answer entirely on what the documents say.

Query type: {query_type}

Instructions based on query type:
- If STATIC: the date of documents does not matter. Focus purely on accuracy.
- If DYNAMIC_CURRENT: prefer information from the most recently dated document. If documents give conflicting information, trust the newer one.
- If DYNAMIC_HISTORICAL: look for documents from the time period mentioned in the question. Prefer documents whose date matches the period being asked about.

Documents (ordered by relevance score):
{documents}

Question: {question}

Answer in 1-2 sentences. Be specific and factual.
If the documents do not contain enough information to answer the question, respond with exactly: "Not found in provided documents."
Do not explain your reasoning. Just give the answer."""


def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Check your .env file.")
    return Groq(api_key=api_key)


def format_documents(docs, top_k=5):
    """
    Format top-k documents into a readable string for the prompt.
    Each document shows its date prominently so the model can
    apply the date-aware instruction correctly.
    """
    formatted = []
    for i, doc in enumerate(docs[:top_k], 1):
        date_str = doc.get("date", "Unknown date")[:10]
        title    = doc.get("title", "Untitled")
        text     = doc.get("text", "")[:500]
        score    = doc.get("final_score", doc.get("cosine_score", 0))

        formatted.append(
            f"[Document {i}] Date: {date_str} | Title: {title} | Score: {score:.3f}\n"
            f"{text}"
        )

    return "\n\n---\n\n".join(formatted)


def generate_answer(question, docs, query_type, client=None):
    """
    Generate a final answer from the top retrieved documents.

    Parameters:
        question   : the user's original question string
        docs       : list of reranked chunk dicts (with final_score and date)
        query_type : "STATIC", "DYNAMIC_CURRENT", or "DYNAMIC_HISTORICAL"
        client     : Groq client (created if not provided)

    Returns:
        answer string
    """
    if client is None:
        client = get_client()

    formatted_docs = format_documents(docs, top_k=5)

    prompt = GENERATION_PROMPT.format(
        query_type=query_type,
        documents=formatted_docs,
        question=question,
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.1,
    )

    return response.choices[0].message.content.strip()


def test_generator():
    """
    Test on three cases:
    1. Static — answer from document content
    2. Dynamic-Current — must use newer document, not older one
    3. Not-found — must return fallback, not hallucinate
    """
    client = get_client()

    print("="*65)
    print("GENERATOR TEST")
    print("="*65)

    # ── Test 1: Static ────────────────────────────────────────────────
    print("\nTest 1: Static question")
    print("Question: What is the Pythagorean theorem?")

    static_docs = [
        {
            "title": "Pythagorean theorem",
            "date": "2024-01-15T00:00:00Z",
            "text": "The Pythagorean theorem states that in a right-angled triangle, "
                    "the square of the hypotenuse equals the sum of squares of the "
                    "other two sides. It is written as a^2 + b^2 = c^2, where c is the "
                    "hypotenuse and a and b are the other two sides.",
            "final_score": 0.91,
        },
        {
            "title": "Geometry fundamentals",
            "date": "2019-06-01T00:00:00Z",
            "text": "Basic geometry covers angles, triangles, and circles. "
                    "The Pythagorean theorem is one of the most important results "
                    "in Euclidean geometry.",
            "final_score": 0.74,
        },
    ]

    answer = generate_answer(
        "What is the Pythagorean theorem?",
        static_docs, "STATIC", client
    )
    print(f"Answer: {answer}")
    print("Expected: mentions a^2 + b^2 = c^2 or right-angled triangle")

    # ── Test 2: Dynamic-Current ───────────────────────────────────────
    print("\nTest 2: Dynamic-Current question")
    print("Question: Who is the Chief Minister of Assam?")

    dynamic_docs = [
        {
            "title": "Chief Minister of Assam (2021)",
            "date": "2021-06-01T00:00:00Z",
            "text": "Himanta Biswa Sarma became the Chief Minister of Assam "
                    "on 10 May 2021, succeeding Sarbananda Sonowal.",
            "final_score": 0.82,
        },
        {
            "title": "Chief Minister of Assam (2018)",
            "date": "2018-03-01T00:00:00Z",
            "text": "Sarbananda Sonowal has been the Chief Minister of Assam "
                    "since 2016, leading the BJP government in the state.",
            "final_score": 0.61,
        },
    ]

    answer = generate_answer(
        "Who is the Chief Minister of Assam?",
        dynamic_docs, "DYNAMIC_CURRENT", client
    )
    print(f"Answer: {answer}")
    print("Expected: Himanta Biswa Sarma (not Sonowal)")

    # ── Test 3: Not-found ─────────────────────────────────────────────
    print("\nTest 3: Not-found case")
    print("Question: What is the population of the moon?")

    irrelevant_docs = [
        {
            "title": "Solar system overview",
            "date": "2023-01-01T00:00:00Z",
            "text": "The solar system consists of the Sun and eight planets. "
                    "The Moon is Earth's only natural satellite and orbits "
                    "at an average distance of 384,400 km.",
            "final_score": 0.55,
        },
    ]

    answer = generate_answer(
        "What is the population of the moon?",
        irrelevant_docs, "STATIC", client
    )
    print(f"Answer: {answer}")
    print("Expected: 'Not found in provided documents' — no hallucinated number")

    print("\n✓ Generator test complete.")
    print("Critical check: Test 2 must say Himanta, not Sonowal.")
    print("Critical check: Test 3 must NOT give a number.")


if __name__ == "__main__":
    test_generator()