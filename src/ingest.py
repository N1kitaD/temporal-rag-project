import requests
import json
import time
import os

WIKI_API = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "TemporalRAGResearchProject/1.0 (student research project; contact: nikitadas6901@gmail.com)"
}

# All categories below have been verified to return real articles.
STATIC_CATEGORIES = [
    "Category:Fundamental constants",
    "Category:Chemical elements",
    "Category:World War II",
    "Category:Ancient Greek mathematicians",
    "Category:SI base units",
]

DYNAMIC_CATEGORIES = [
    "Category:Chief ministers of Assam",
    "Category:Prime ministers of India",
    "Category:Presidents of India",
    "Category:Chief ministers of Indian states",
    "Category:Governors of Assam",
    "Category:World chess champions",
]

def get_category_members(category, limit=100):
    titles = []
    cmcontinue = None

    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": min(50, limit - len(titles)),
            "cmtype": "page",
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        response = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20)
        response.raise_for_status()
        data = response.json()

        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(member["title"] for member in members)

        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break

    return titles[:limit]

def fetch_article(title, retries=3):
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|revisions",
        "explaintext": True,
        "rvprop": "timestamp",
        "format": "json",
    }

    for attempt in range(retries):
        try:
            response = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20)
            response.raise_for_status()
            pages = response.json().get("query", {}).get("pages", {})

            for page_id, page in pages.items():
                if page_id == "-1":
                    print("  missing:", title)
                    return None
                text = page.get("extract", "")
                revisions = page.get("revisions", [])
                date = revisions[0]["timestamp"] if revisions else None
                if not text or not date:
                    print("  skipped (no text/date):", title)
                    return None
                return {"title": page.get("title", title), "text": text, "date": date}
            return None

        except requests.exceptions.RequestException as error:
            print("  network error on", title, "- attempt", attempt + 1, "of", retries, "-", error)
            time.sleep(2)

    print("  giving up on:", title, "after", retries, "attempts")
    return None

def build_corpus_from_categories(categories, topic_type, domain="wikipedia_entity", per_category_limit=60):
    all_titles = []
    for category in categories:
        print("pulling category:", category)
        members = get_category_members(category, limit=per_category_limit)
        print("  found", len(members), "articles")
        all_titles.extend(members)
        time.sleep(0.3)

    # remove duplicates while keeping order
    seen = set()
    unique_titles = []
    for title in all_titles:
        if title not in seen:
            seen.add(title)
            unique_titles.append(title)

    articles = []
    for index, title in enumerate(unique_titles):
        print("fetching", index + 1, "/", len(unique_titles), title)
        result = fetch_article(title)
        if result:
            article_id = domain + "_" + topic_type + "_" + str(index).zfill(4)
            articles.append({
                "id": article_id,
                "title": result["title"],
                "text": result["text"],
                "date": result["date"],
                "domain": domain,
                "topic_type": topic_type,
            })
        time.sleep(0.3)

    return articles

# Standalone articles to pull by exact title, in addition to category members.
# Useful for "List of X" summary pages, which Wikipedia deliberately keeps
# out of their own category.
STATIC_EXTRA_TITLES = [
    "List of ancient Greek mathematicians",
]

DYNAMIC_EXTRA_TITLES = [
    "List of current Indian chief ministers",
    "List of current Indian governors",
]

def build_corpus_from_titles(titles, topic_type, domain="wikipedia_entity"):
    articles = []
    for index, title in enumerate(titles):
        print("fetching (standalone)", index + 1, "/", len(titles), title)
        result = fetch_article(title)
        if result:
            article_id = domain + "_" + topic_type + "_standalone_" + str(index).zfill(3)
            articles.append({
                "id": article_id,
                "title": result["title"],
                "text": result["text"],
                "date": result["date"],
                "domain": domain,
                "topic_type": topic_type,
            })
        time.sleep(0.3)
    return articles

def main():
    os.makedirs("data/corpus", exist_ok=True)

    static_articles = build_corpus_from_categories(STATIC_CATEGORIES, "static")
    static_articles += build_corpus_from_titles(STATIC_EXTRA_TITLES, "static")

    dynamic_articles = build_corpus_from_categories(DYNAMIC_CATEGORIES, "dynamic")
    dynamic_articles += build_corpus_from_titles(DYNAMIC_EXTRA_TITLES, "dynamic")

    all_articles = static_articles + dynamic_articles

    with open("data/corpus/articles.json", "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    print("saved", len(all_articles), "articles to data/corpus/articles.json")
    print("static:", len(static_articles), "dynamic:", len(dynamic_articles))

if __name__ == "__main__":
    main()