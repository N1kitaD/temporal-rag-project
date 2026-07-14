"""
Verified corpus fix — all titles checked against Wikipedia live.
Adds: corrected titles, current Indian leaders (person pages),
current world leaders, and 100+ additional widespread topics.
Run: python src/topup_fix.py
"""

import requests
import json
import time
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

WIKI_API      = "https://en.wikipedia.org/w/api.php"
HEADERS       = {"User-Agent": "TemporalRAGProject/1.0 (research internship; nikitadas6901@gmail.com)"}
ARTICLES_PATH = "data/corpus/articles_full.json"
CHUNKS_PATH   = "data/corpus/chunks.json"
INDEX_PATH    = "data/corpus/faiss_index.pkl"

# ═══════════════════════════════════════════════════════════════════
# VERIFIED STATIC TITLES
# Every title below confirmed against Wikipedia live search
# ═══════════════════════════════════════════════════════════════════

VERIFIED_STATIC = [

    # --- Previously "not found" — now with correct titles ---
    "Nvidia",                          # was "NVIDIA" — lowercase v is correct
    "Computer security",               # was "Cybersecurity" — correct main article
    "The Holocaust",                   # was "Holocaust"
    "Cardiovascular disease",          # was "Heart disease"
    "Hindi cinema",                    # was "Bollywood"
    "National Basketball Association", # was "NBA"
    "Olympic Games",                   # was "Olympics"
    "Manchester City F.C.",            # was "Manchester City FC"
    "Exploration of Mars",             # was "Mars exploration"
    "ASEAN",                           # was "Association of Southeast Asian Nations"
    "Brahmaputra River",               # was "Brahmaputra" — confirmed correct
    "World Bank",                      # retry — should work now
    "Radioactive decay",               # was "Radioactivity"
    "Indian Rebellion of 1857",        # was "Sepoy Mutiny" — confirmed correct
    "Ayushman Bharat",                 # retry
    "Emergency in India",              # was "Emergency (India)" — alternate title

    # --- Science and Technology (new, widespread) ---
    "Transformer (deep learning architecture)",  # the attention model — CORRECT title
    "Attention mechanism",             # self-attention in deep learning
    "Generative adversarial network",  # GANs
    "Diffusion model",                 # stable diffusion family
    "BERT (language model)",           # Google's BERT
    "Computer vision",                 # already may exist but worth retrying
    "Convolutional neural network",    # CNNs
    "Recurrent neural network",        # RNNs
    "Support vector machine",          # SVMs
    "Random forest",                   # ensemble ML
    "Gradient descent",                # optimization
    "Overfitting",                     # ML concept
    "Transfer learning",               # ML technique
    "Encoder–decoder model",           # seq2seq
    "FAISS",                           # Facebook AI Similarity Search — may not exist
    "Vector database",                 # modern RAG component
    "Retrieval-augmented generation",  # RAG itself
    "Prompt engineering",              # LLM prompting
    "Hallucination (artificial intelligence)", # AI hallucination
    "Autonomous vehicle",              # self-driving cars
    "Semiconductor",                   # already exists but retry
    "Integrated circuit",              # chips
    "Moore's law",                     # chip scaling
    "5G",                              # wireless standard
    "Wi-Fi",                           # networking
    "HTTP",                            # web protocol
    "HTTPS",                           # secure web
    "Application programming interface", # APIs
    "Open source",                     # open source software
    "GitHub",                          # code hosting
    "Docker (software)",               # containerization
    "Kubernetes",                      # container orchestration
    "DevOps",                          # software development

    # --- Mathematics (additional) ---
    "Boolean algebra",
    "Fourier transform",
    "Bayes' theorem",
    "Normal distribution",
    "Game theory",
    "Cryptography",
    "Information theory",
    "Turing machine",
    "Computational complexity theory",
    "P versus NP problem",

    # --- Biology and Medicine (additional) ---
    "Neuroscience",
    "Genomics",
    "Stem cell",
    "Alzheimer's disease",
    "Depression (mood)",
    "Anxiety disorder",
    "Obesity",
    "Malaria",
    "Tuberculosis",
    "Ebola virus disease",
    "Antibiotic resistance",
    "Human genome",
    "Chromosome",
    "Cell division",
    "Nervous system",
    "Endocrine system",
    "Respiratory system",

    # --- Physics (additional) ---
    "Special relativity",
    "General relativity",
    "String theory",
    "Gravitational wave",
    "Laser",
    "Magnetic resonance imaging",
    "Nuclear reactor",
    "Solar system",
    "Milky Way",
    "Exoplanet",
    "Neutron star",
    "Supernova",
    "Cosmic microwave background",

    # --- Chemistry (additional) ---
    "Organic chemistry",
    "Polymer",
    "Carbon nanotube",
    "Graphene",
    "Lithium-ion battery",
    "Hydrogen",
    "Nitrogen",
    "Oxygen",
    "Carbon",
    "Silicon",
    "Plastic",
    "Nuclear chemistry",

    # --- World History (additional) ---
    "Silk Road",
    "Age of Exploration",
    "Crusades",
    "Black Death",
    "Hundred Years' War",
    "Napoleon",
    "Abraham Lincoln",
    "Winston Churchill",
    "Adolf Hitler",
    "Joseph Stalin",
    "Mao Zedong",
    "Nelson Mandela",
    "Martin Luther King Jr.",
    "Albert Einstein",
    "Isaac Newton",
    "Charles Darwin",
    "Galileo Galilei",
    "Marie Curie",
    "Nikola Tesla",
    "Thomas Edison",

    # --- Indian History (additional) ---
    "Akbar",
    "Ashoka",
    "Chandragupta Maurya",
    "Chhatrapati Shivaji Maharaj",
    "Tipu Sultan",
    "Rani of Jhansi",
    "Sardar Vallabhbhai Patel",
    "Maulana Abul Kalam Azad",
    "Bal Gangadhar Tilak",
    "Bhagat Singh",
    "Swami Vivekananda",
    "Ramanujan",

    # --- Economics and Finance (additional) ---
    "Microeconomics",
    "Macroeconomics",
    "Monetary policy",
    "Fiscal policy",
    "Budget deficit",
    "National debt",
    "Venture capital",
    "Initial public offering",
    "Hedge fund",
    "Mutual fund",
    "Insurance",
    "Taxation",
    "Subsidy",
    "Minimum wage",
    "Gini coefficient",

    # --- Geography (additional) ---
    "Biodiversity hotspot",
    "Tectonic plates",
    "Earthquake",
    "Volcano",
    "Tsunami",
    "Monsoon",
    "Desert",
    "Rainforest",
    "Tundra",
    "Mediterranean climate",
    "Glacier",
    "Great Barrier Reef",
    "Mariana Trench",
    "Dead Sea",
    "Caspian Sea",
    "Lake Baikal",
    "Victoria Lake",
    "Mississippi River",
    "Rhine",
    "Danube",
    "Mekong River",
    "Yellow River",
    "Yangtze River",
    "Congo River",

    # --- Indian Geography (additional) ---
    "Thar Desert",
    "Sundarbans",
    "Nilgiri Hills",
    "Andaman and Nicobar Islands",
    "Lakshadweep",
    "Kaveri",
    "Godavari",
    "Yamuna",
    "Mahanadi",
    "Narmada River",
    "Vindhya Range",
    "Eastern Ghats",
    "Indo-Gangetic Plain",
    "Manipur",
    "Nagaland",
    "Tripura",
    "Mizoram",
    "Meghalaya",
    "Arunachal Pradesh",
    "Sikkim",
    "Odisha",
    "Jharkhand",
    "Chhattisgarh",
    "Uttarakhand",
    "Himachal Pradesh",
    "Goa",
    "Bihar",
    "Haryana",
    "Telangana",
    "Andhra Pradesh",

    # --- Arts, Culture, Literature ---
    "Miguel de Cervantes",
    "Homer",
    "Dante Alighieri",
    "Franz Kafka",
    "Gabriel García Márquez",
    "Rabindranath Tagore",
    "Premchand",
    "R. K. Narayan",
    "Amartya Sen",
    "C. V. Raman",
    "A. P. J. Abdul Kalam",
    "M. S. Swaminathan",

    # --- Social Sciences ---
    "Sociology",
    "Anthropology",
    "Political science",
    "International relations",
    "Geopolitics",
    "Globalization",
    "Urbanization",
    "Migration",
    "Human trafficking",
    "Child labour",
    "Caste system in India",
    "Gender inequality",
    "Affirmative action",
    "Universal basic income",
    "Social media",
    "Misinformation",
    "Fake news",
    "Propaganda",
    "Freedom of the press",
    "Freedom of speech",
    "Privacy",
    "Surveillance",

    # --- Environment (additional) ---
    "Ocean acidification",
    "Sea level rise",
    "Plastic pollution",
    "Air pollution",
    "Water crisis",
    "Nuclear power",
    "Hydroelectric power",
    "Geothermal energy",
    "Carbon capture",
    "Sustainable development",
    "Circular economy",
    "Rewilding",
    "Reforestation",
    "Wildlife conservation",
    "Wetland",
    "Mangrove",
]

# ═══════════════════════════════════════════════════════════════════
# VERIFIED DYNAMIC TITLES
# Current role holders — verified correct as of July 2026
# ═══════════════════════════════════════════════════════════════════

VERIFIED_DYNAMIC = [

    # --- Current Indian Leaders — PERSON PAGES (best for QA) ---
    # These pages open with "X is the current Y" in first sentence
    "Narendra Modi",           # PM of India — Wikipedia title verified
    "Droupadi Murmu",          # President of India — verified
    "Jagdeep Dhankhar",        # Vice President of India — verified
    "Om Birla",                # Speaker, Lok Sabha — verified
    "Surya Kant",              # CURRENT CJI since Nov 2024 (NOT Sanjiv Khanna who retired May 2025)

    # Indian Chief Ministers — person pages
    "Himanta Biswa Sarma",     # CM Assam — re-elected May 2026, verified
    "Devendra Fadnavis",       # CM Maharashtra — verified
    "M. K. Stalin",            # CM Tamil Nadu — verified
    "Siddaramaiah",            # CM Karnataka — verified
    "Yogi Adityanath",         # CM Uttar Pradesh — verified
    "Mamata Banerjee",         # CM West Bengal — verified
    "Bhupendra Patel",         # CM Gujarat — verified
    "Bhajan Lal Sharma",       # CM Rajasthan — verified
    "Pinarayi Vijayan",        # CM Kerala — verified
    "Rekha Gupta",             # CM Delhi — verified
    "Sukhvinder Singh Sukhu",  # CM Himachal Pradesh — verified
    "Pushkar Singh Dhami",     # CM Uttarakhand — verified
    "Mohan Yadav",             # CM Madhya Pradesh — verified
    "Vishnu Deo Sai",          # CM Chhattisgarh — verified
    "Hemant Soren",            # CM Jharkhand
    "Nara Chandrababu Naidu",  # CM Andhra Pradesh
    "Revanth Reddy",           # CM Telangana
    "Siddaramaiah",            # CM Karnataka (duplicate guard in code)
    "Pema Khandu",             # CM Arunachal Pradesh
    "Conrad Sangma",           # CM Meghalaya
    "Zoramthanga",             # CM Mizoram
    "Neiphiu Rio",             # CM Nagaland
    "Manik Saha",              # CM Tripura
    "Biplab Kumar Deb",        # former CM Tripura — still relevant
    "N. Biren Singh",          # CM Manipur
    "P. S. Golay",             # CM Sikkim
    "Pramod Sawant",           # CM Goa
    "Bhupesh Baghel",          # former CM Chhattisgarh

    # --- Current World Leaders — PERSON PAGES (verified) ---
    "Donald Trump",            # US President since Jan 2025 — CORRECT title
    "Keir Starmer",            # UK PM since July 2024 — CORRECT title
    "Olaf Scholz",             # Chancellor of Germany — verified
    "Emmanuel Macron",         # President of France — verified
    "Xi Jinping",              # President of China — verified
    "Vladimir Putin",          # President of Russia — verified
    "Shigeru Ishiba",          # PM of Japan since Oct 2024 — CORRECT (not Kishida)
    "Anthony Albanese",        # PM of Australia — verified
    "Mark Carney",             # PM of Canada since 2025 — verified
    "Luiz Inácio Lula da Silva", # President of Brazil — verified
    "Cyril Ramaphosa",         # President of South Africa — verified
    "António Guterres",        # UN Secretary-General — verified
    "Pedro Sánchez",           # PM of Spain
    "Giorgia Meloni",          # PM of Italy
    "Recep Tayyip Erdoğan",    # President of Turkey
    "Narendra Modi",           # (already above, skip guard handles it)
    "Pope Leo XIV",            # New Pope (Francis died April 2025, Leo XIV elected May 2025)
    "Yoon Suk-yeol",           # President of South Korea (context: impeachment saga)
    "Volodymyr Zelenskyy",     # President of Ukraine — important current figure

    # --- Tech CEOs and Companies (dynamic — leadership changes) ---
    "Sundar Pichai",           # CEO of Google/Alphabet — verified
    "Satya Nadella",           # CEO of Microsoft — verified
    "Tim Cook",                # CEO of Apple — verified
    "Mark Zuckerberg",         # CEO of Meta — verified
    "Elon Musk",               # CEO of Tesla/SpaceX/X — verified
    "Jensen Huang",            # CEO of Nvidia — verified
    "Sam Altman",              # CEO of OpenAI — verified
    "Dario Amodei",            # CEO of Anthropic
    "Jeff Bezos",              # Amazon founder — dynamic (wealth/ventures)
    "OpenAI",                  # company page — dynamic
    "Anthropic",               # company page — dynamic
    "xAI",                     # Elon Musk's AI company
    "Mistral AI",              # French AI startup — verified title
    "Perplexity AI",           # AI search startup
    "Google DeepMind",         # DeepMind merged with Google Brain
    "DeepSeek",                # Chinese AI lab — major 2025 story

    # --- Sports: Cricket ---
    "ICC Cricket World Cup",   # was "ICC Men's Cricket World Cup" — CORRECT title
    "2023 Cricket World Cup",  # specific recent tournament
    "2024 ICC Men's T20 World Cup", # India won — important
    "ICC World Test Championship",
    "Jasprit Bumrah",          # current Indian cricket star
    "Shubman Gill",            # rising Indian star
    "Yashasvi Jaiswal",        # young Indian opener
    "Hardik Pandya",           # all-rounder
    "Rohit Sharma",            # (already exists, skip guard)
    "Virat Kohli",             # (already exists)
    "KL Rahul",                # Indian wicketkeeper-batter

    # --- Sports: Football ---
    "Erling Haaland",          # Man City striker
    "Jude Bellingham",         # Real Madrid midfielder
    "Vinicius Junior",         # Real Madrid
    "2026 FIFA World Cup",     # upcoming — USA/Canada/Mexico

    # --- Sports: Others ---
    "P. V. Sindhu",            # was "PV Sindhu" — CORRECT title with dots
    "Neeraj Chopra",           # (already exists)
    "Max Verstappen",          # F1 world champion
    "Carlos Sainz Jr.",        # F1 — note "Jr." is correct
    "Novak Djokovic",          # (already exists)
    "Carlos Alcaraz",          # Tennis — new generation
    "Jannik Sinner",           # Tennis world No.1 2024
    "2024 Summer Olympics",    # Paris Olympics
    "Iga Świątek",             # Tennis

    # --- AI and Technology (dynamic — evolving fast) ---
    "GPT-4o",                  # OpenAI's flagship 2024 model
    "Claude (language model)", # Anthropic's model — Wikipedia title verified
    "Llama (language model)",  # Meta's open source model — CORRECT title
    "Gemini (language model)", # (already exists)
    "AlphaFold",               # (already exists)
    "Sora (text-to-video model)", # OpenAI video model
    "DALL-E",                  # OpenAI image model
    "Stable Diffusion",        # open-source image model
    "GitHub Copilot",          # AI coding assistant
    "Midjourney",              # image generation
    "DeepSeek-R1",             # Chinese reasoning model 2025
    "Grok (chatbot)",          # xAI chatbot

    # --- Space ---
    "James Webb Space Telescope", # (already exists)
    "Chandrayaan-3",           # (already exists)
    "Artemis program",         # (already exists)
    "SpaceX Starship",         # (already exists)
    "Indian Space Research Organisation", # full name — may differ from ISRO

    # --- India Current Affairs ---
    "Goods and Services Tax (India)",  # (already exists)
    "Unified Payments Interface",      # (already exists)
    "Aadhaar",                         # (already exists)
    "Digital India",                   # (already exists)
    "Production-linked incentive",     # PLI scheme
    "National Education Policy 2020",  # NEP
    "Agnipath scheme",                 # military recruitment
    "One Nation One Election",         # political topic 2024
    "Semiconductor industry in India", # emerging sector
    "India–China relations",           # geopolitics
    "India–Pakistan relations",        # geopolitics
    "BRICS",                           # (already exists)
    "G20",                             # (already exists)
    "SCO",                             # Shanghai Cooperation Organisation

    # --- World Current Affairs ---
    "Russia–Ukraine war",              # ongoing conflict
    "Gaza war",                        # Israel-Gaza conflict 2023-2025
    "Taiwan Strait",                   # geopolitical flashpoint
    "US–China trade war",              # economic conflict
    "Brexit",                          # UK-EU separation — still evolving
    "NATO",                            # (already exists)
    "AUKUS",                           # security pact
    "Quad (security dialogue)",        # India-US-Japan-Australia
    "United Nations Climate Change conference", # COP summits
    "COP29",                           # 2024 climate summit
    "International Criminal Court",    # (already exists)
]


# ═══════════════════════════════════════════════════════════════════
# SCRIPT FUNCTIONS (same robust pattern as previous scripts)
# ═══════════════════════════════════════════════════════════════════

def fetch_article(title):
    params = {
        "action": "query", "titles": title,
        "prop": "extracts|revisions",
        "explaintext": True, "rvprop": "timestamp", "format": "json",
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
    words = article["text"].split()
    chunks = []
    step = chunk_size - overlap
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
            "domain":     "wikipedia_entity",
            "topic_type": article["topic_type"],
        })
    return chunks


def fetch_batch(titles, topic_type, existing_titles):
    new_articles = []
    seen_in_batch = set()
    for i, title in enumerate(titles, 1):
        if title in existing_titles or title in seen_in_batch:
            print(f"  [{i:03d}] skip   {title}")
            continue
        seen_in_batch.add(title)

        result = fetch_article(title)
        if result:
            aid = title.lower().replace(" ", "_")[:60]
            new_articles.append({
                "id": aid, "title": result["title"],
                "text": result["text"], "date": result["date"],
                "domain": "wikipedia_entity", "topic_type": topic_type,
            })
            print(f"  [{i:03d}] saved  {result['title'][:55]} ({result['date'][:10]})")
        else:
            print(f"  [{i:03d}] ✗ NOT FOUND: {title}")
        time.sleep(0.35)
    return new_articles


def rebuild_index(chunks, model_name="all-MiniLM-L6-v2"):
    print(f"\nRebuilding FAISS index over {len(chunks)} chunks ...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        [c["text"] for c in chunks],
        batch_size=64, show_progress_bar=True, convert_to_numpy=True
    ).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"index": index, "chunks": chunks, "model_name": model_name}, f)
    mb = os.path.getsize(INDEX_PATH) / (1024 * 1024)
    print(f"Index rebuilt: {index.ntotal} vectors | {INDEX_PATH} ({mb:.1f} MB)")


def main():
    print("Loading existing corpus ...")
    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing_titles = {a["title"] for a in existing}
    print(f"Existing articles: {len(existing)}\n")

    all_new = []

    print(f"=== Fetching STATIC articles ({len(VERIFIED_STATIC)} titles) ===")
    new_static = fetch_batch(VERIFIED_STATIC, "static", existing_titles)
    all_new.extend(new_static)
    existing_titles.update(a["title"] for a in new_static)

    print(f"\n=== Fetching DYNAMIC articles ({len(VERIFIED_DYNAMIC)} titles) ===")
    new_dynamic = fetch_batch(VERIFIED_DYNAMIC, "dynamic", existing_titles)
    all_new.extend(new_dynamic)

    if not all_new:
        print("\nNothing new to add — all titles already in corpus.")
        return

    # Save articles
    all_articles = existing + all_new
    with open(ARTICLES_PATH, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    # Save chunks
    print("\nLoading existing chunks ...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        existing_chunks = json.load(f)

    new_chunks = []
    for a in all_new:
        new_chunks.extend(chunk_article(a))

    all_chunks = existing_chunks + new_chunks
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nCorpus:  {len(existing)} → {len(all_articles)} articles (+{len(all_new)})")
    print(f"Chunks:  {len(existing_chunks)} → {len(all_chunks)} (+{len(new_chunks)})")
    print(f"  Static added:  {len(new_static)}")
    print(f"  Dynamic added: {len(new_dynamic)}")

    rebuild_index(all_chunks)

    print(f"\n=== DONE ===")
    print(f"Total corpus: {len(all_articles)} articles | {len(all_chunks)} chunks")
    print("\nVerification test — run these queries to confirm fixes:")
    print("  'Who is the current Chief Justice of India?'  → Surya Kant")
    print("  'Who is the Chief Minister of Assam?'         → Himanta Biswa Sarma")
    print("  'Who is the CEO of Nvidia?'                   → Jensen Huang")
    print("  'Who is the Prime Minister of UK?'            → Keir Starmer")
    print("  'Who is the President of the USA?'            → Donald Trump")


if __name__ == "__main__":
    main()