import json
import os

def chunk_article(article, chunk_size=300, overlap=50):
    """
    Split one article into overlapping word-window chunks.
    Each chunk inherits the parent article's date and topic_type.

    chunk_size: target words per chunk
    overlap: words repeated between consecutive chunks
    so that facts that straddle a boundary aren't cut off
    """
    words = article["text"].split()
    chunks = []
    chunk_index = 0

    # step size: how far we move forward each time
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]

        # skip very short trailing chunks — not useful for retrieval
        if len(chunk_words) < 50:
            break

        chunk_text = " ".join(chunk_words)

        chunks.append({
            "id": f"{article['id']}_chunk_{str(chunk_index).zfill(3)}",
            "parent_id": article["id"],
            "title": article["title"],
            "text": chunk_text,
            "date": article["date"],
            "domain": article["domain"],
            "topic_type": article["topic_type"],
        })
        chunk_index += 1

    return chunks


def chunk_corpus(input_path, output_path):
    """
    Load all articles, chunk every one, save all chunks to output file.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    print(f"Loaded {len(articles)} articles from {input_path}")

    all_chunks = []
    static_chunks = 0
    dynamic_chunks = 0

    for i, article in enumerate(articles):
        chunks = chunk_article(article)
        all_chunks.extend(chunks)

        if article["topic_type"] == "static":
            static_chunks += len(chunks)
        else:
            dynamic_chunks += len(chunks)

        if (i + 1) % 50 == 0:
            print(f"  chunked {i + 1}/{len(articles)} articles "
                  f"({len(all_chunks)} chunks so far)")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n=== Chunking complete ===")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"  From static articles:  {static_chunks}")
    print(f"  From dynamic articles: {dynamic_chunks}")
    print(f"Saved to: {output_path}")

    # sanity checks
    print("\n--- Sample chunks (first 2) ---")
    for chunk in all_chunks[:2]:
        word_count = len(chunk["text"].split())
        print(f"  [{chunk['id']}] '{chunk['title']}' "
              f"| date: {chunk['date'][:10]} "
              f"| type: {chunk['topic_type']} "
              f"| words: {word_count}")

    return all_chunks


def main():
    os.makedirs("data/corpus", exist_ok=True)

    # use articles_full.json if it exists, fall back to articles.json
    if os.path.exists("data/corpus/articles_full.json"):
        input_path = "data/corpus/articles_full.json"
    else:
        input_path = "data/corpus/articles.json"

    output_path = "data/corpus/chunks.json"

    print(f"Reading from: {input_path}")
    chunk_corpus(input_path, output_path)


if __name__ == "__main__":
    main()