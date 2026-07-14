"""
Large-scale corpus expansion — 400 additional Wikipedia articles
covering widespread topics across science, history, geography,
technology, politics, sports, culture, and economics.

Run with: python src/topup_corpus_400.py
Takes 20-40 minutes. Rebuilds FAISS index at the end.
"""

import requests
import json
import time
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

WIKI_API    = "https://en.wikipedia.org/w/api.php"
HEADERS     = {"User-Agent": "TemporalRAGProject/1.0 (research internship; nikitadas6901@gmail.com)"}
ARTICLES_PATH = "data/corpus/articles_full.json"
CHUNKS_PATH   = "data/corpus/chunks.json"
INDEX_PATH    = "data/corpus/faiss_index.pkl"

# ── 400 article titles ────────────────────────────────────────────────────

STATIC_TITLES = [

    # Science — Physics
    "Newton's laws of motion", "Theory of relativity", "Quantum mechanics",
    "Thermodynamics", "Electromagnetism", "Nuclear fission", "Nuclear fusion",
    "Radioactivity", "Optics", "Acoustics", "Fluid mechanics",
    "Semiconductor", "Superconductivity", "Particle physics",
    "Standard Model", "Big Bang", "Black hole", "Dark matter",
    "Hubble Space Telescope", "International Space Station",

    # Science — Chemistry
    "Periodic table", "Chemical bond", "Acid–base reaction",
    "Photosynthesis", "DNA", "RNA", "Protein", "Enzyme",
    "Cell (biology)", "Mitosis", "Evolution", "Natural selection",
    "Genetics", "CRISPR", "Antibiotic", "Vaccine", "Virus",
    "Bacteria", "Immune system", "Human brain",

    # Science — Mathematics
    "Calculus", "Linear algebra", "Statistics", "Probability",
    "Number theory", "Graph theory", "Topology",
    "Fibonacci sequence", "Prime number", "Euclidean geometry",
    "Trigonometry", "Complex number", "Differential equation",
    "Matrix (mathematics)", "Set theory",

    # Technology — Computing
    "Artificial intelligence", "Machine learning", "Deep learning",
    "Neural network", "Natural language processing",
    "Computer vision", "Reinforcement learning",
    "Large language model", "Transformer (machine learning model)",
    "Internet", "World Wide Web", "Cloud computing",
    "Blockchain", "Cryptocurrency", "Bitcoin",
    "Operating system", "Linux", "Algorithm",
    "Data structure", "Database", "SQL",
    "Python (programming language)", "JavaScript",
    "Cybersecurity", "Encryption",

    # Technology — Companies (historical facts)
    "History of Google", "History of Apple Inc.",
    "History of Microsoft", "History of Amazon (company)",
    "History of Meta Platforms", "History of Tesla, Inc.",
    "History of IBM", "Intel", "NVIDIA",
    "SpaceX", "NASA", "ISRO",

    # Geography — World
    "Amazon River", "Nile", "Sahara", "Himalayas",
    "Mount Everest", "Pacific Ocean", "Atlantic Ocean",
    "Arctic", "Antarctica", "Equator",
    "United States", "China", "Russia", "United Kingdom",
    "Germany", "France", "Japan", "Brazil", "Australia",
    "Canada", "South Africa", "Nigeria", "Egypt",
    "Saudi Arabia", "Iran", "Turkey", "Indonesia",
    "Mexico", "Argentina",

    # Geography — India
    "India", "States and union territories of India",
    "Assam", "Maharashtra", "Tamil Nadu", "Karnataka",
    "Uttar Pradesh", "West Bengal", "Gujarat", "Rajasthan",
    "Kerala", "Punjab, India", "Guwahati", "Mumbai",
    "Delhi", "Chennai", "Kolkata", "Bengaluru", "Hyderabad",
    "Brahmaputra", "Ganges", "Deccan Plateau", "Western Ghats",

    # History — World
    "World War I", "World War II", "Cold War",
    "French Revolution", "American Revolution",
    "Industrial Revolution", "Renaissance",
    "Roman Empire", "Ancient Greece", "Ancient Egypt",
    "Mongol Empire", "Ottoman Empire", "British Empire",
    "Colonialism", "Slavery", "Holocaust",
    "United Nations", "NATO", "European Union",
    "League of Nations", "Treaty of Versailles",

    # History — India
    "History of India", "Mughal Empire", "Maratha Empire",
    "British Raj", "Indian independence movement",
    "Partition of India", "Constitution of India",
    "Green Revolution in India", "Emergency (India)",
    "Battle of Plassey", "Sepoy Mutiny",
    "Non-cooperation movement", "Salt March",
    "Quit India Movement",

    # Economics
    "Gross domestic product", "Inflation", "Unemployment",
    "Stock market", "Reserve Bank of India",
    "Federal Reserve", "World Bank", "International Monetary Fund",
    "Globalization", "Free trade", "Capitalism", "Socialism",
    "Supply and demand", "Interest rate",
    "Foreign exchange market", "Cryptocurrency",
    "Economic inequality", "Poverty",
    "Human Development Index",

    # Environment
    "Climate change", "Global warming", "Greenhouse gas",
    "Carbon dioxide", "Ozone layer", "Deforestation",
    "Biodiversity", "Endangered species", "Coral reef",
    "Renewable energy", "Solar energy", "Wind power",
    "Electric vehicle", "Paris Agreement",
    "Kyoto Protocol", "Environmental movement",

    # Philosophy and Society
    "Democracy", "Human rights", "Feminism",
    "Philosophy", "Ethics", "Logic",
    "Plato", "Aristotle", "Immanuel Kant",
    "Karl Marx", "Sigmund Freud",
    "Religion", "Islam", "Christianity",
    "Hinduism", "Buddhism", "Sikhism",

    # Arts and Culture
    "Classical music", "Jazz", "Rock music",
    "Cinema of India", "Bollywood",
    "William Shakespeare", "Leo Tolstoy",
    "Renaissance art", "Impressionism",
    "Architecture", "Photography",

    # Medicine and Health
    "Cancer", "Diabetes", "Heart disease",
    "COVID-19 pandemic", "HIV/AIDS",
    "World Health Organization",
    "Mental health", "Nutrition",
    "Surgery", "Pharmacology",
]

DYNAMIC_TITLES = [

    # Current world leaders
    "President of the United States",
    "Prime Minister of the United Kingdom",
    "Chancellor of Germany",
    "President of France",
    "President of China",
    "President of Russia",
    "Prime Minister of Japan",
    "Prime Minister of Australia",
    "Prime Minister of Canada",
    "President of Brazil",
    "President of South Africa",
    "Secretary-General of the United Nations",
    "Pope Francis",

    # Indian leadership
    "Prime Minister of India",
    "President of India",
    "Chief Minister of Assam",
    "Chief Minister of Maharashtra",
    "Chief Minister of Tamil Nadu",
    "Chief Minister of Karnataka",
    "Chief Minister of West Bengal",
    "Chief Justice of India",
    "Governor of Assam",
    "Speaker of the Lok Sabha",

    # Tech CEOs and companies
    "Sundar Pichai",
    "Satya Nadella",
    "Tim Cook",
    "Mark Zuckerberg",
    "Elon Musk",
    "Jensen Huang",
    "Sam Altman",
    "Google",
    "Apple Inc.",
    "Microsoft",
    "Meta Platforms",
    "Amazon (company)",
    "Tesla, Inc.",
    "OpenAI",
    "Anthropic",
    "NVIDIA",

    # Sports — Cricket
    "Virat Kohli",
    "Rohit Sharma",
    "MS Dhoni",
    "Sachin Tendulkar",
    "Jasprit Bumrah",
    "ICC Men's Cricket World Cup",
    "Indian Premier League",
    "Board of Control for Cricket in India",
    "Test cricket",
    "One Day International",

    # Sports — Football/Soccer
    "FIFA World Cup",
    "UEFA Champions League",
    "Premier League",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Kylian Mbappé",
    "Real Madrid CF",
    "FC Barcelona",
    "Manchester City FC",

    # Sports — Other
    "Olympics",
    "2024 Summer Olympics",
    "NBA",
    "LeBron James",
    "Novak Djokovic",
    "Roger Federer",
    "Serena Williams",
    "Formula One",
    "Lewis Hamilton",
    "PV Sindhu",
    "Neeraj Chopra",

    # Space and exploration
    "Mars exploration",
    "James Webb Space Telescope",
    "Chandrayaan-3",
    "Artemis program",
    "SpaceX Starship",

    # Current affairs — organizations
    "G20",
    "BRICS",
    "Association of Southeast Asian Nations",
    "World Trade Organization",
    "International Criminal Court",

    # Science — recent
    "ChatGPT",
    "GPT-4",
    "Gemini (language model)",
    "AlphaFold",
    "mRNA vaccine",
    "James Webb Space Telescope discoveries",

    # Indian current affairs
    "Goods and Services Tax (India)",
    "Digital India",
    "Make in India",
    "Ayushman Bharat",
    "Unified Payments Interface",
    "Aadhaar",
]


def fetch_article(title):
    params = {
        "action":      "query",
        "titles":      title,
        "prop":        "extracts|revisions",
        "explaintext": True,
        "rvprop":      "timestamp",
        "format":      "json",
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for pid, page in pages.items():
            if pid == "-1":
                return None
            text = page.get("extract", "")
            revs = page.get("revisions", [])
            date = revs[0]["timestamp"] if revs else None
            if not text or not date or len(text) < 200:
                return None
            return {"title": page.get("title", title), "text": text, "date": date}
    except Exception as e:
        print(f"  [error] {title}: {e}")
        return None


def chunk_article(article, chunk_size=300, overlap=50):
    words  = article["text"].split()
    chunks = []
    step   = chunk_size - overlap
    for ci, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start: start + chunk_size]
        if len(chunk_words) < 50:
            break
        chunks.append({
            "id":         f"{article['id']}_chunk_{str(ci).zfill(3)}",
            "parent_id":  article["id"],
            "title":      article["title"],
            "text":       " ".join(chunk_words),
            "date":       article["date"],
            "domain":     article["domain"],
            "topic_type": article["topic_type"],
        })
    return chunks


def fetch_batch(titles, topic_type, existing_titles):
    new_articles = []
    for i, title in enumerate(titles, 1):
        if title in existing_titles:
            print(f"  [{i:03d}] skip (exists)   {title}")
            continue
        result = fetch_article(title)
        if result:
            aid = title.lower().replace(" ", "_")[:60]
            new_articles.append({
                "id":         aid,
                "title":      result["title"],
                "text":       result["text"],
                "date":       result["date"],
                "domain":     "wikipedia_entity",
                "topic_type": topic_type,
            })
            print(f"  [{i:03d}] saved          {result['title']} ({result['date'][:10]})")
        else:
            print(f"  [{i:03d}] not found      {title}")
        time.sleep(0.35)
    return new_articles


def rebuild_index(chunks, model_name="all-MiniLM-L6-v2"):
    print(f"\nRebuilding FAISS index over {len(chunks)} chunks...")
    model      = SentenceTransformer(model_name)
    texts      = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=64,
                              show_progress_bar=True,
                              convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    payload = {"index": index, "chunks": chunks, "model_name": model_name}
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)
    mb = os.path.getsize(INDEX_PATH) / (1024 * 1024)
    print(f"Index rebuilt: {index.ntotal} vectors | {INDEX_PATH} ({mb:.1f} MB)")


def main():
    os.makedirs("data/corpus", exist_ok=True)

    print(f"Loading existing corpus...")
    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing_titles = {a["title"] for a in existing}
    print(f"Existing: {len(existing)} articles\n")

    print("=== Fetching STATIC articles ===")
    new_static  = fetch_batch(STATIC_TITLES,  "static",  existing_titles)

    print("\n=== Fetching DYNAMIC articles ===")
    existing_titles.update(a["title"] for a in new_static)
    new_dynamic = fetch_batch(DYNAMIC_TITLES, "dynamic", existing_titles)

    new_all = new_static + new_dynamic
    if not new_all:
        print("\nNothing new to add.")
        return

    all_articles = existing + new_all
    with open(ARTICLES_PATH, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    print(f"\nCorpus: {len(existing)} → {len(all_articles)} articles (+{len(new_all)})")

    print("\nLoading existing chunks...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        existing_chunks = json.load(f)

    new_chunks = []
    for a in new_all:
        new_chunks.extend(chunk_article(a))

    all_chunks = existing_chunks + new_chunks
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Chunks: {len(existing_chunks)} → {len(all_chunks)} (+{len(new_chunks)})")

    rebuild_index(all_chunks)

    print("\n=== DONE ===")
    print(f"Articles added: {len(new_all)}")
    print(f"  Static:  {len(new_static)}")
    print(f"  Dynamic: {len(new_dynamic)}")
    print(f"Total corpus: {len(all_articles)} articles, {len(all_chunks)} chunks")


if __name__ == "__main__":
    main()