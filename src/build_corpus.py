import requests
import json
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

WIKI_API = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "TemporalRAGProject/1.0 (research internship project; nikitadas6901@gmail.com)"
}

# ── STATIC categories — facts that never change ────────────────────────────
# Each entry is (category_name, max_articles_to_pull)
STATIC_CATEGORIES = [
    # mathematics & constants
    ("Mathematical constants", 25),
    ("Physical constants", 25),
    ("History of mathematics", 20),
    ("Mathematical theorems", 20),
    ("Number theory", 15),
    # ancient & settled history
    ("Ancient history", 25),
    ("Ancient Rome", 20),
    ("Ancient Greece", 20),
    ("Mughal Empire", 20),
    ("World War II", 25),
    ("World War I", 20),
    # science — settled, textbook facts
    ("Chemical elements", 25),
    ("Classical mechanics", 20),
    ("Laws of thermodynamics", 15),
    ("Geometric shapes", 20),
    ("Astronomical objects", 20),
    # geography — fixed natural facts
    ("Mountains", 20),
    ("Rivers", 20),
    ("Deserts", 15),
    ("Islands", 15),
    # inventions & discoveries — completed events
    ("Inventions", 20),
    ("Scientific discoveries", 20),
    ("Nobel Prize in Physics", 20),
    ("Nobel Prize in Chemistry", 20),
]

# ── DYNAMIC categories — facts that change over time ──────────────────────
DYNAMIC_CATEGORIES = [
    # Indian political leadership — changes with elections
    ("Chief Ministers of Assam", 15),
    ("Chief Ministers of Maharashtra", 15),
    ("Chief Ministers of Tamil Nadu", 15),
    ("Chief Ministers of Karnataka", 15),
    ("Chief Ministers of Uttar Pradesh", 15),
    ("Chief Ministers of West Bengal", 15),
    ("Chief Ministers of Gujarat", 15),
    ("Chief Ministers of Rajasthan", 15),
    ("Chief Ministers of Delhi", 10),
    ("Prime Ministers of India", 15),
    ("Presidents of India", 15),
    # sports — records and champions change
    ("Cricket records", 20),
    ("Olympic records", 15),
    ("FIFA World Cup", 15),
    ("Indian Premier League seasons", 15),
    # technology — versions and leadership change
    ("Python programming language", 10),
    ("Artificial intelligence", 15),
    ("Mobile operating systems", 10),
    # international leadership
    ("Secretaries-General of the United Nations", 10),
    ("Heads of government", 15),
]

# ── STATIC direct titles — high-value articles fetched individually ────────
# These are timeless historical facts. Category pulls sometimes miss them.
STATIC_DIRECT_TITLES = [
    # Indian history — completed, never changes
    "Indian independence movement",
    "Partition of India",
    "Mahatma Gandhi",
    "Jawaharlal Nehru",
    "B. R. Ambedkar",
    "Rabindranath Tagore",
    "Subhas Chandra Bose",
    "Constitution of India",
    "Battle of Plassey",
    "Maurya Empire",
    "Indus Valley Civilisation",
    # world history — completed events
    "World War II",
    "World War I",
    "French Revolution",
    "Renaissance",
    "Industrial Revolution",
    "American Civil War",
    "Cold War",
    "Moon landing",
    # science — permanent textbook facts
    "Theory of relativity",
    "Quantum mechanics",
    "Evolution",
    "DNA",
    "Periodic table",
    "Photosynthesis",
    "Boiling point",
    "Speed of light",
    "Pythagorean theorem",
    "Pi",
    "Euler's number",
    "Newton's laws of motion",
    # geography — fixed natural facts
    "Mount Everest",
    "Amazon River",
    "Sahara Desert",
    "Pacific Ocean",
    "Himalayas",
    "Nile",
    "Atlantic Ocean",
]

# ── DYNAMIC direct titles — high-value evolving-fact articles ─────────────
# These are the most important for Evolving Semantic Conflict.
# Old Wikipedia revisions of these articles give DIFFERENT answers than new ones.
DYNAMIC_DIRECT_TITLES = [
    # Indian current leadership — answer changes with elections/appointments
    "Prime Minister of India",
    "President of India",
    "Chief Minister of Assam",
    "Chief Minister of Maharashtra",
    "Chief Minister of Tamil Nadu",
    "Chief Minister of Karnataka",
    "Chief Minister of Uttar Pradesh",
    "Chief Minister of West Bengal",
    "Chief Minister of Gujarat",
    "Chief Minister of Delhi",
    "Chief Justice of India",
    "Speaker of the Lok Sabha",
    "Vice President of India",
    # list articles — show history of changing role holders
    "List of Chief Ministers of Assam",
    "List of Chief Ministers of Maharashtra",
    "List of Chief Ministers of Tamil Nadu",
    "List of Prime Ministers of India",
    "List of Presidents of India",
    # sports — current records and champion status change
    "India national cricket team",
    "ICC Cricket World Cup",
    "Indian Premier League",
    "List of Test cricket records",
    "List of One Day International cricket records",
    "FIFA World Cup",
    "Olympic Games",
    "List of world records in athletics",
    # current sports personalities — team affiliations change
    "Virat Kohli",
    "Rohit Sharma",
    "MS Dhoni",
    "Sachin Tendulkar",
    # technology — versions and current state change
    "Python (programming language)",
    "TensorFlow",
    "PyTorch",
    "ChatGPT",
    "GPT-4",
    "Android (operating system)",
    "iOS",
    "Linux kernel",
    "Wikipedia",
    # organisations whose leadership changes
    "United Nations",
    "World Health Organization",
    "International Monetary Fund",
    "Reserve Bank of India",
    "ISRO",
    "NASA",
    "International Cricket Council",
    # evolving world situation articles
    "India",
    "Economy of India",
    "Demographics of India",
    "Climate change",
    "COVID-19 pandemic",
    "Artificial intelligence",
    "Large language model",
]


# ── session setup (keep exactly from previous code) ───────────────────────

def make_session(retries=3, backoff_factor=1.5, timeout=30):
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session, timeout


def check_connectivity(session, timeout):
    print("Checking connectivity to en.wikipedia.org ...", end=" ", flush=True)
    try:
        resp = session.get(
            "https://en.wikipedia.org/w/api.php?action=query&meta=siteinfo&format=json",
            timeout=timeout,
        )
        resp.raise_for_status()
        print("OK")
        return True
    except requests.exceptions.ConnectTimeout:
        print("FAILED (connection timed out)")
    except requests.exceptions.ConnectionError as e:
        print(f"FAILED (connection error: {e})")
    except requests.exceptions.HTTPError as e:
        print(f"FAILED (HTTP {e.response.status_code})")
    except Exception as e:
        print(f"FAILED ({e})")

    print("\n[ERROR] Cannot reach en.wikipedia.org.")
    print("Possible causes and fixes:")
    print("  1. No internet connection  — check your network.")
    print("  2. Firewall / proxy        — try a different network or VPN.")
    print("  3. Wikipedia blocked       — use a VPN.")
    print("  4. Temporary outage        — wait a few minutes and retry.")
    print("  5. DNS issue               — try: ipconfig /flushdns  (Windows)")
    return False


# ── fetchers (keep exactly from previous code) ────────────────────────────

def get_category_members(session, timeout, category_name, max_articles=50):
    titles = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category_name}",
        "cmlimit": max_articles,
        "cmtype": "page",
        "format": "json",
    }
    try:
        response = session.get(WIKI_API, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        members = data.get("query", {}).get("categorymembers", [])
        for member in members:
            titles.append(member["title"])
        print(f"    category '{category_name}': found {len(titles)} titles")
    except requests.exceptions.ConnectTimeout:
        print(f"  [WARN] Timeout fetching category '{category_name}' — skipping.")
    except requests.exceptions.ConnectionError as e:
        print(f"  [WARN] Connection error for category '{category_name}': {e} — skipping.")
    except requests.exceptions.HTTPError as e:
        print(f"  [WARN] HTTP {e.response.status_code} for '{category_name}' — skipping.")
    except Exception as e:
        print(f"  [WARN] Unexpected error for '{category_name}': {e} — skipping.")
    return titles


def fetch_article(session, timeout, title):
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|revisions",
        "explaintext": True,
        "rvprop": "timestamp",
        "format": "json",
    }
    try:
        response = session.get(WIKI_API, params=params, timeout=timeout)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1":
                return None
            text = page.get("extract", "")
            revisions = page.get("revisions", [])
            date = revisions[0]["timestamp"] if revisions else None
            if not text or not date or len(text) < 200:
                return None
            return {
                "title": page.get("title", title),
                "text": text,
                "date": date,
            }
    except requests.exceptions.ConnectTimeout:
        print(f"    [WARN] Timeout fetching article '{title}' — skipping.")
    except requests.exceptions.ConnectionError as e:
        print(f"    [WARN] Connection error fetching '{title}': {e} — skipping.")
    except Exception as e:
        print(f"    [WARN] Error fetching '{title}': {e} — skipping.")
    return None


# ── builders ──────────────────────────────────────────────────────────────

def build_from_categories(session, timeout, category_list, topic_type,
                           seen_titles, domain="wikipedia"):
    """
    Pull articles from a list of (category_name, max_articles) tuples.
    seen_titles is shared across all calls to prevent duplicates.
    """
    articles = []
    for category_name, max_articles in category_list:
        print(f"\n  Processing category: {category_name}")
        titles = get_category_members(session, timeout, category_name,
                                      max_articles=max_articles)
        time.sleep(0.5)

        for title in titles:
            if title in seen_titles:
                continue
            seen_titles.add(title)

            result = fetch_article(session, timeout, title)
            time.sleep(0.3)

            if result:
                article_id = title.lower().replace(" ", "_")[:60]
                articles.append({
                    "id": article_id,
                    "title": result["title"],
                    "text": result["text"],
                    "date": result["date"],
                    "domain": domain,
                    "topic_type": topic_type,
                })
                print(f"    saved [{len(articles):>3}] {result['title'][:55]}")
            else:
                print(f"    skipped: {title[:55]}")

    return articles


def build_from_direct_titles(session, timeout, title_list, topic_type,
                              seen_titles, domain="wikipedia"):
    """
    Fetch specific articles by title — used for high-value pages that
    category pulls often miss or return inconsistently.
    seen_titles is shared to prevent duplicates with category results.
    """
    articles = []
    for title in title_list:
        if title in seen_titles:
            print(f"    [dup]  {title[:55]}")
            continue
        seen_titles.add(title)

        result = fetch_article(session, timeout, title)
        time.sleep(0.3)

        if result:
            article_id = title.lower().replace(" ", "_")[:60]
            articles.append({
                "id": article_id,
                "title": result["title"],
                "text": result["text"],
                "date": result["date"],
                "domain": domain,
                "topic_type": topic_type,
            })
            print(f"    saved [{len(articles):>3}] {result['title'][:55]}")
        else:
            print(f"    [skip] {title[:55]}")

    return articles


# ── main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs("data/corpus", exist_ok=True)

    session, timeout = make_session(retries=3, backoff_factor=1.5, timeout=30)

    if not check_connectivity(session, timeout):
        return

    # seen_titles is shared across ALL four build calls below
    # so the same article never appears twice regardless of source
    seen_titles = set()
    all_articles = []

    # ── STATIC ──
    print("\n=== Building STATIC corpus — from categories ===")
    static_cat = build_from_categories(
        session, timeout, STATIC_CATEGORIES, "static", seen_titles)
    all_articles.extend(static_cat)
    print(f"\n  Static from categories: {len(static_cat)}")

    print("\n=== Building STATIC corpus — from direct titles ===")
    static_dir = build_from_direct_titles(
        session, timeout, STATIC_DIRECT_TITLES, "static", seen_titles)
    all_articles.extend(static_dir)
    print(f"\n  Static from direct titles: {len(static_dir)}")

    static_total = len(static_cat) + len(static_dir)
    print(f"\n  STATIC TOTAL: {static_total}")

    # ── DYNAMIC ──
    print("\n=== Building DYNAMIC corpus — from categories ===")
    dynamic_cat = build_from_categories(
        session, timeout, DYNAMIC_CATEGORIES, "dynamic", seen_titles)
    all_articles.extend(dynamic_cat)
    print(f"\n  Dynamic from categories: {len(dynamic_cat)}")

    print("\n=== Building DYNAMIC corpus — from direct titles ===")
    dynamic_dir = build_from_direct_titles(
        session, timeout, DYNAMIC_DIRECT_TITLES, "dynamic", seen_titles)
    all_articles.extend(dynamic_dir)
    print(f"\n  Dynamic from direct titles: {len(dynamic_dir)}")

    dynamic_total = len(dynamic_cat) + len(dynamic_dir)
    print(f"\n  DYNAMIC TOTAL: {dynamic_total}")

    # ── save ──
    output_path = "data/corpus/articles_full.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    print(f"\n=== Done ===")
    print(f"Total articles saved: {len(all_articles)}")
    print(f"  Static:  {static_total}")
    print(f"  Dynamic: {dynamic_total}")
    print(f"Saved to: {output_path}")

    if len(all_articles) == 0:
        print("\nWARNING: No articles collected. Check network connection.")
    elif len(all_articles) < 300:
        print(f"\nWARNING: Only {len(all_articles)} articles collected.")
        print("Some categories may have been empty. Check [WARN] lines above.")
    elif len(all_articles) >= 450:
        print("\nTarget reached: 450+ articles collected.")
    else:
        print(f"\nPartial: {len(all_articles)} articles. Target was 450-500.")


if __name__ == "__main__":
    main()