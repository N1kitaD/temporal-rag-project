import json
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

CHUNKS_PATH = "data/corpus/chunks.json"
INDEX_PATH = "data/corpus/faiss_index.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def build_index(chunks, model_name=MODEL_NAME, batch_size=64):
    """
    Embed every chunk and store vectors in a FAISS index.
    Returns: (faiss_index, embedding_model, chunks)
    """
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # extract just the text from each chunk for embedding
    texts = [chunk["text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks in batches of {batch_size}...")
    print("(This takes 2-5 minutes on CPU — let it run)\n")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # normalize to unit length — required for cosine similarity via inner product
    # after normalization, dot product == cosine similarity
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    # IndexFlatIP = exact search using inner product (cosine similarity after normalization)
    # "Flat" means no compression — stores full vectors, exact results every time
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"\nFAISS index built:")
    print(f"  Vectors stored: {index.ntotal}")
    print(f"  Vector dimension: {dimension}")

    return index, model, embeddings


def save_index(index, chunks, model_name, path):
    """
    Save FAISS index + chunk metadata together in one pickle file.
    Chunk metadata (text, date, title) is needed to display results later.
    """
    payload = {
        "index": index,
        "chunks": chunks,
        "model_name": model_name,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Saved index to {path} ({size_mb:.1f} MB)")


def load_index(path):
    """
    Load a saved index back from disk.
    Returns: (faiss_index, chunks, model_name)
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    print(f"Loaded index from {path}")
    print(f"  Chunks in index: {payload['index'].ntotal}")
    return payload["index"], payload["chunks"], payload["model_name"]


def retrieve(query, index, chunks, model, top_k=20):
    """
    Search the FAISS index for the top_k chunks most similar to query.
    Returns a list of chunk dicts, each with a 'cosine_score' field added.
    """
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx].copy()
        chunk["cosine_score"] = float(score)
        results.append(chunk)

    return results


def test_retrieval(index, chunks, model):
    """
    Run 3 test queries and print results.
    This is your checkpoint — verify results make sense before proceeding.
    """
    test_queries = [
        ("What is the boiling point of water?", "static"),
        ("Who is the current Chief Minister of Assam?", "dynamic"),
        ("When did World War II end?", "static"),
    ]

    print("\n" + "="*60)
    print("RETRIEVAL TEST — top 3 results per query")
    print("="*60)

    for query, expected_type in test_queries:
        print(f"\nQuery: {query}")
        print(f"Expected type: {expected_type}")
        results = retrieve(query, index, chunks, model, top_k=5)
        for i, r in enumerate(results[:3], 1):
            print(f"  [{i}] score={r['cosine_score']:.3f} | "
                  f"type={r['topic_type']} | "
                  f"date={r['date'][:10]} | "
                  f"title='{r['title'][:45]}'")


def main():
    os.makedirs("data/corpus", exist_ok=True)

    chunks = load_chunks(CHUNKS_PATH)
    index, model, _ = build_index(chunks, model_name=MODEL_NAME)
    save_index(index, chunks, MODEL_NAME, INDEX_PATH)
    test_retrieval(index, chunks, model)

    print("\n✓ Retriever built and verified.")
    print("Next step: build the classifier (src/classifier.py)")


if __name__ == "__main__":
    main()