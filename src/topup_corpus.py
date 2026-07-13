"""
Targeted corpus top-up script.
Fetches only the missing articles identified from stress test failures
and appends them to the existing articles_full.json.
Then rebuilds the FAISS index to include the new chunks.

Run with: python src/topup_corpus.py
"""

import requests
import json
import time
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "TemporalRAGProject/1.0 (research internship; uddipan@student.edu)"
}

ARTICLES_PATH = "data/corpus/articles_full.json"
CHUNKS_PATH   = "data/corpus/chunks.json"
INDEX_PATH    = "data/corpus/faiss_index.pkl"

# ── Missing articles identified from stress test failures ─────────────────
# Static topics — answers never change
MISSING_STATIC = [
    "Water",
    "Properties of water",
    "Calcutta",
    "Paris",
    "History of Delhi",
    "New Delhi",
    "Steve Jobs",
    "Speed of light",
    "World War I",
    "Chemical formula",
]

# Dynamic topics — answers change over time
MISSING_DYNAMIC = [
    "Chief Justice of India",
    "Sundar Pichai",
    "ICC Men's Cricket World Cup",
    "ICC Cricket World Cup",
    "India national cricket team records",
    "Sachin Tendulkar",
    "List of Chief Justices of India",
    "Google",
    "Apple Inc.",
    "Narendra Modi",
]


def fetch_article(title):
    """Fetch one article's text and last-edited date from Wikipedia."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|revisions",
        "explaintext": True,
        "rvprop": "timestamp",
        "format": "json",
    }
    try:
        response = requests.get(WIKI_API, params=params,
                                headers=HEADERS, timeout=15)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1":
                print(f"  [missing] {title}")
                return None
            text = page.get("extract", "")
            revisions = page.get("revisions", [])
            date = revisions[0]["timestamp"] if revisions else None
            if not text or not date or len(text) < 200:
                print(f"  [empty]   {title}")
                return None
            return {
                "title": page.get("title", title),
                "text": text,
                "date": date,
            }
    except Exception as e:
        print(f"  [error]   {title}: {e}")
        return None


def chunk_article(article, chunk_size=300, overlap=50):
    """Split article into overlapping word-window chunks."""
    words = article["text"].split()
    chunks = []
    step = chunk_size - overlap
    chunk_index = 0
    for start in range(0, len(words), step):
        chunk_words = words[start: start + chunk_size]
        if len(chunk_words) < 50:
            break
        chunks.append({
            "id": f"{article['id']}_chunk_{str(chunk_index).zfill(3)}",
            "parent_id": article["id"],
            "title": article["title"],
            "text": " ".join(chunk_words),
            "date": article["date"],
            "domain": article["domain"],
            "topic_type": article["topic_type"],
        })
        chunk_index += 1
    return chunks


def fetch_missing(titles, topic_type, existing_titles):
    """
    Fetch articles for a list of titles.
    Skips titles already present in the corpus.
    Returns list of new article dicts.
    """
    new_articles = []
    for title in titles:
        if title in existing_titles:
            print(f"  [skip]    {title} (already in corpus)")
            continue
        print(f"  fetching  {title}")
        result = fetch_article(title)
        if result:
            article_id = title.lower().replace(" ", "_")[:60]
            new_articles.append({
                "id": article_id,
                "title": result["title"],
                "text": result["text"],
                "date": result["date"],
                "domain": "wikipedia_entity",
                "topic_type": topic_type,
            })
            print(f"  [saved]   {result['title']} ({result['date'][:10]})")
        time.sleep(0.4)
    return new_articles


def rebuild_index(chunks, model_name="all-MiniLM-L6-v2"):
    """Rebuild the full FAISS index from all chunks."""
    print(f"\nRebuilding FAISS index over {len(chunks)} chunks...")
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    payload = {"index": index, "chunks": chunks, "model_name": model_name}
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)
    size_mb = os.path.getsize(INDEX_PATH) / (1024 * 1024)
    print(f"Index rebuilt: {index.ntotal} vectors | saved to {INDEX_PATH} ({size_mb:.1f} MB)")


def main():
    os.makedirs("data/corpus", exist_ok=True)

    # ── Load existing corpus ──────────────────────────────────────────
    print(f"Loading existing corpus from {ARTICLES_PATH}...")
    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        existing_articles = json.load(f)

    existing_titles = {a["title"] for a in existing_articles}
    print(f"Existing articles: {len(existing_articles)}")
    print(f"Existing unique titles: {len(existing_titles)}")

    # ── Fetch missing articles ────────────────────────────────────────
    print("\n=== Fetching missing STATIC articles ===")
    new_static = fetch_missing(MISSING_STATIC, "static", existing_titles)

    print("\n=== Fetching missing DYNAMIC articles ===")
    new_dynamic = fetch_missing(MISSING_DYNAMIC, "dynamic", existing_titles)

    new_articles = new_static + new_dynamic
    if not new_articles:
        print("\nNo new articles fetched — corpus already contains all targets.")
        return

    # ── Append to existing corpus ─────────────────────────────────────
    all_articles = existing_articles + new_articles
    with open(ARTICLES_PATH, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    print(f"\nCorpus updated:")
    print(f"  Before: {len(existing_articles)} articles")
    print(f"  Added:  {len(new_articles)} articles")
    print(f"  After:  {len(all_articles)} articles")

    # ── Rebuild chunks ────────────────────────────────────────────────
    print("\nLoading existing chunks...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        existing_chunks = json.load(f)

    new_chunks = []
    for article in new_articles:
        new_chunks.extend(chunk_article(article))

    all_chunks = existing_chunks + new_chunks
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Chunks updated:")
    print(f"  Before: {len(existing_chunks)} chunks")
    print(f"  Added:  {len(new_chunks)} chunks")
    print(f"  After:  {len(all_chunks)} chunks")

    # ── Rebuild FAISS index ───────────────────────────────────────────
    rebuild_index(all_chunks)

    print("\n=== Top-up complete ===")
    print(f"New articles added: {len(new_articles)}")
    print("  Static: " + ", ".join(a["title"] for a in new_static))
    print("  Dynamic: " + ", ".join(a["title"] for a in new_dynamic))
    print("\nRun stress_test.py again to verify the gaps are filled.")


if __name__ == "__main__":
    main()