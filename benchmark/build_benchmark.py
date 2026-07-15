"""
Benchmark Builder — Phase 9 (110 questions)
Builds the controlled evaluation set for the four-way comparison.

Groups:
  1. Static (30 questions)
  2. Dynamic-Current (25 questions) — Wikipedia revision pairs
  3. Dynamic-Historical (25 questions)
  4. Undated-but-Dynamic (15 questions)
  5. No-meaningful-date control (10 questions) -- but we target 5 here since
     group 5 is illustrative only; remainder added to groups 1-4.

Total target: 110 questions

Run with: python benchmark/build_benchmark.py
Output:   data/benchmark/benchmark.json
"""

import json
import os
import requests
import time

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS  = {"User-Agent": "TemporalRAGProject/1.0 (research internship; nikitadas6901@gmail.com)"}
OUTPUT   = "data/benchmark/benchmark.json"


def get_old_revision(title, before_year):
    params = {
        "action":  "query",
        "titles":  title,
        "prop":    "revisions",
        "rvprop":  "content|timestamp",
        "rvlimit": "50",
        "rvend":   f"{before_year}-01-01T00:00:00Z",
        "rvdir":   "older",
        "format":  "json",
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for pid, page in pages.items():
            if pid == "-1": return None
            revs = page.get("revisions", [])
            if not revs: return None
            rev  = revs[0]
            text = rev.get("*", rev.get("slots", {}).get("main", {}).get("*", ""))
            if len(text) < 100: return None
            return {"title": page.get("title", title),
                    "text":  text[:2000], "date": rev["timestamp"]}
    except Exception as e:
        print(f"    [error] {title}: {e}")
        return None


def get_current_revision(title):
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
            if pid == "-1": return None
            text = page.get("extract", "")
            revs = page.get("revisions", [])
            date = revs[0]["timestamp"] if revs else None
            if not text or not date: return None
            return {"title": page.get("title", title),
                    "text":  text[:2000], "date": date}
    except Exception as e:
        print(f"    [error] {title}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────
# GROUP 1 — STATIC (30 questions)
# ─────────────────────────────────────────────────────────────────────────
STATIC_QUESTIONS = [
    # Science & Math
    {"question": "What is the chemical formula of water?",
     "correct_answer": "H2O", "doc_title": "Water"},
    {"question": "What is the Pythagorean theorem?",
     "correct_answer": "a squared plus b squared equals c squared",
     "doc_title": "Pythagorean theorem"},
    {"question": "What is the speed of light?",
     "correct_answer": "299792458", "doc_title": "Speed of light"},
    {"question": "What is the boiling point of water?",
     "correct_answer": "100", "doc_title": "Water"},
    {"question": "What is the chemical symbol for gold?",
     "correct_answer": "Au", "doc_title": "Gold"},
    {"question": "What is the chemical symbol for oxygen?",
     "correct_answer": "O", "doc_title": "Oxygen"},
    {"question": "What is the atomic number of carbon?",
     "correct_answer": "6", "doc_title": "Carbon"},
    {"question": "What is Newton's first law of motion?",
     "correct_answer": "inertia", "doc_title": "Newton's laws of motion"},
    {"question": "What is the value of pi?",
     "correct_answer": "3.14159", "doc_title": "Pi"},
    {"question": "What is the powerhouse of the cell?",
     "correct_answer": "mitochondria", "doc_title": "Mitochondrion"},
    {"question": "How many chromosomes does a human have?",
     "correct_answer": "46", "doc_title": "Chromosome"},
    {"question": "What is the theory of evolution?",
     "correct_answer": "natural selection", "doc_title": "Evolution"},
    {"question": "What is photosynthesis?",
     "correct_answer": "chlorophyll", "doc_title": "Photosynthesis"},
    {"question": "What is the largest planet in the solar system?",
     "correct_answer": "Jupiter", "doc_title": "Jupiter"},
    {"question": "What is the freezing point of water in Celsius?",
     "correct_answer": "0", "doc_title": "Water"},
    # History & Geography
    {"question": "When did India gain independence?",
     "correct_answer": "1947", "doc_title": "Indian independence movement"},
    {"question": "When did World War II end?",
     "correct_answer": "1945", "doc_title": "World War II"},
    {"question": "When did World War I begin?",
     "correct_answer": "1914", "doc_title": "World War I"},
    {"question": "When was the Constitution of India adopted?",
     "correct_answer": "1950", "doc_title": "Constitution of India"},
    {"question": "Who wrote the Indian national anthem?",
     "correct_answer": "Rabindranath Tagore",
     "doc_title": "Rabindranath Tagore"},
    {"question": "What is the capital of France?",
     "correct_answer": "Paris", "doc_title": "Paris"},
    {"question": "What is the capital of Japan?",
     "correct_answer": "Tokyo", "doc_title": "Tokyo"},
    {"question": "What is the longest river in the world?",
     "correct_answer": "Nile", "doc_title": "Nile"},
    {"question": "What is the tallest mountain in the world?",
     "correct_answer": "Everest", "doc_title": "Mount Everest"},
    {"question": "Who discovered gravity?",
     "correct_answer": "Isaac Newton", "doc_title": "Isaac Newton"},
    {"question": "Who was the first President of India?",
     "correct_answer": "Rajendra Prasad", "doc_title": "Rajendra Prasad"},
    {"question": "Who invented the telephone?",
     "correct_answer": "Alexander Graham Bell",
     "doc_title": "Alexander Graham Bell"},
    {"question": "When did the French Revolution begin?",
     "correct_answer": "1789", "doc_title": "French Revolution"},
    {"question": "What was India's capital before New Delhi?",
     "correct_answer": "Calcutta", "doc_title": "New Delhi"},
    {"question": "How many states does India have?",
     "correct_answer": "28", "doc_title": "States and union territories of India"},
]

# ─────────────────────────────────────────────────────────────────────────
# GROUP 2 — DYNAMIC-CURRENT (25 questions, Wikipedia revision pairs)
# ─────────────────────────────────────────────────────────────────────────
DYNAMIC_CURRENT_QUESTIONS = [
    # Indian leadership
    {"question": "Who is the Chief Minister of Assam?",
     "correct_answer": "Himanta Biswa Sarma",
     "wiki_title": "Chief Minister of Assam",
     "old_before_year": 2021, "old_wrong_answer": "Sarbananda Sonowal"},
    {"question": "Who is the Prime Minister of India?",
     "correct_answer": "Narendra Modi",
     "wiki_title": "Prime Minister of India",
     "old_before_year": 2010, "old_wrong_answer": "Manmohan Singh"},
    {"question": "Who is the President of India?",
     "correct_answer": "Droupadi Murmu",
     "wiki_title": "President of India",
     "old_before_year": 2020, "old_wrong_answer": "Ram Nath Kovind"},
    {"question": "Who is the Chief Minister of West Bengal?",
     "correct_answer": "Mamata Banerjee",
     "wiki_title": "Chief Minister of West Bengal",
     "old_before_year": 2010, "old_wrong_answer": "Buddhadeb Bhattacharjee"},
    {"question": "Who is the Chief Minister of Tamil Nadu?",
     "correct_answer": "M. K. Stalin",
     "wiki_title": "Chief Minister of Tamil Nadu",
     "old_before_year": 2020, "old_wrong_answer": "Edappadi K. Palaniswami"},
    {"question": "Who is the Chief Minister of Karnataka?",
     "correct_answer": "Siddaramaiah",
     "wiki_title": "Chief Minister of Karnataka",
     "old_before_year": 2022, "old_wrong_answer": "Basavaraj Bommai"},
    {"question": "Who is the Chief Minister of Maharashtra?",
     "correct_answer": "Devendra Fadnavis",
     "wiki_title": "Chief Minister of Maharashtra",
     "old_before_year": 2019, "old_wrong_answer": "Uddhav Thackeray"},
    {"question": "Who is the Chief Justice of India?",
     "correct_answer": "Surya Kant",
     "wiki_title": "Chief Justice of India",
     "old_before_year": 2020, "old_wrong_answer": "Sharad Arvind Bobde"},
    # World leadership
    {"question": "Who is the Prime Minister of the United Kingdom?",
     "correct_answer": "Keir Starmer",
     "wiki_title": "Prime Minister of the United Kingdom",
     "old_before_year": 2022, "old_wrong_answer": "Boris Johnson"},
    {"question": "Who is the President of the United States?",
     "correct_answer": "Donald Trump",
     "wiki_title": "President of the United States",
     "old_before_year": 2021, "old_wrong_answer": "Joe Biden"},
    {"question": "Who is the Chancellor of Germany?",
     "correct_answer": "Friedrich Merz",
     "wiki_title": "Chancellor of Germany",
     "old_before_year": 2021, "old_wrong_answer": "Angela Merkel"},
    {"question": "Who is the President of France?",
     "correct_answer": "Emmanuel Macron",
     "wiki_title": "President of France",
     "old_before_year": 2016, "old_wrong_answer": "Francois Hollande"},
    {"question": "Who is the Prime Minister of Canada?",
     "correct_answer": "Mark Carney",
     "wiki_title": "Prime Minister of Canada",
     "old_before_year": 2020, "old_wrong_answer": "Justin Trudeau"},
    {"question": "Who is the Prime Minister of Australia?",
     "correct_answer": "Anthony Albanese",
     "wiki_title": "Prime Minister of Australia",
     "old_before_year": 2021, "old_wrong_answer": "Scott Morrison"},
    {"question": "Who is the Secretary-General of the United Nations?",
     "correct_answer": "Antonio Guterres",
     "wiki_title": "Secretary-General of the United Nations",
     "old_before_year": 2016, "old_wrong_answer": "Ban Ki-moon"},
     {"question": "Who is the President of Brazil?",
     "correct_answer": "Luiz Inácio Lula da Silva",
     "wiki_title": "President of Brazil",
     "old_before_year": 2022, "old_wrong_answer": "Jair Bolsonaro"},
    # Tech leadership
    {"question": "Who is the CEO of Google?",
     "correct_answer": "Sundar Pichai",
     "wiki_title": "Sundar Pichai",
     "old_before_year": 2015, "old_wrong_answer": "Larry Page"},
    {"question": "Who is the CEO of Microsoft?",
     "correct_answer": "Satya Nadella",
     "wiki_title": "Satya Nadella",
     "old_before_year": 2013, "old_wrong_answer": "Steve Ballmer"},
    {"question": "Who is the CEO of Apple?",
     "correct_answer": "Tim Cook",
     "wiki_title": "Tim Cook",
     "old_before_year": 2010, "old_wrong_answer": "Steve Jobs"},
    {"question": "Who is the CEO of OpenAI?",
     "correct_answer": "Sam Altman",
     "wiki_title": "OpenAI",
     "old_before_year": 2023, "old_wrong_answer": "Greg Brockman"},
    # Sports
    {"question": "Who is the world number one ranked chess player?",
     "correct_answer": "Magnus Carlsen",
     "wiki_title": "Magnus Carlsen",
     "old_before_year": 2010, "old_wrong_answer": "Viswanathan Anand"},
    {"question": "Who is the world number one men's tennis player?",
     "correct_answer": "Jannik Sinner",
     "wiki_title": "Jannik Sinner",
     "old_before_year": 2023, "old_wrong_answer": "Novak Djokovic"},
    {"question": "Who is the Formula One world champion?",
     "correct_answer": "Max Verstappen",
     "wiki_title": "Max Verstappen",
     "old_before_year": 2020, "old_wrong_answer": "Lewis Hamilton"},
    {"question": "Who is the current captain of the Indian cricket team?",
     "correct_answer": "Rohit Sharma",
     "wiki_title": "India national cricket team",
     "old_before_year": 2020, "old_wrong_answer": "Virat Kohli"},
    {"question": "Who won the 2023 Cricket World Cup?",
     "correct_answer": "Australia",
     "wiki_title": "2023 Cricket World Cup",
     "old_before_year": 2022, "old_wrong_answer": "England"},
]

# ─────────────────────────────────────────────────────────────────────────
# GROUP 3 — DYNAMIC-HISTORICAL (25 questions)
# ─────────────────────────────────────────────────────────────────────────
DYNAMIC_HISTORICAL_QUESTIONS = [
    # Indian leadership — past
    {"question": "Who was the Chief Minister of Assam in 2018?",
     "correct_answer": "Sarbananda Sonowal",
     "doc_title": "Chief Minister of Assam"},
    {"question": "Who was the Prime Minister of India in 2005?",
     "correct_answer": "Manmohan Singh",
     "doc_title": "Manmohan Singh"},
    {"question": "Who was the President of India in 2010?",
     "correct_answer": "Pratibha Patil",
     "doc_title": "Pratibha Patil"},
    {"question": "Who was the Chief Minister of West Bengal in 2008?",
     "correct_answer": "Buddhadeb Bhattacharjee",
     "doc_title": "Buddhadeb Bhattacharjee"},
    {"question": "Who was the Chief Minister of Tamil Nadu in 2015?",
     "correct_answer": "Jayalalithaa",
     "doc_title": "Jayalalithaa"},
    {"question": "Who was the Chief Minister of Karnataka in 2010?",
     "correct_answer": "B. S. Yediyurappa",
     "doc_title": "B. S. Yediyurappa"},
    {"question": "Who was the Chief Minister of Maharashtra in 2009?",
     "correct_answer": "Ashok Chavan",
     "doc_title": "Ashok Chavan"},
    {"question": "Who was the Chief Justice of India in 2015?",
     "correct_answer": "H. L. Dattu",
     "doc_title": "H. L. Dattu"},
    # World leadership — past
    {"question": "Who was the Prime Minister of UK in 2010?",
     "correct_answer": "David Cameron",
     "doc_title": "David Cameron"},
    {"question": "Who was the President of USA in 2010?",
     "correct_answer": "Barack Obama",
     "doc_title": "Barack Obama"},
    {"question": "Who was the Chancellor of Germany in 2010?",
     "correct_answer": "Angela Merkel",
     "doc_title": "Angela Merkel"},
    {"question": "Who was the President of France in 2010?",
     "correct_answer": "Nicolas Sarkozy",
     "doc_title": "Nicolas Sarkozy"},
    {"question": "Who was the Prime Minister of Canada in 2010?",
     "correct_answer": "Stephen Harper",
     "doc_title": "Stephen Harper"},
    {"question": "Who was the Secretary-General of the UN in 2010?",
     "correct_answer": "Ban Ki-moon",
     "doc_title": "Ban Ki-moon"},
    # Tech — past
    {"question": "Who was the CEO of Apple when the iPhone was launched?",
     "correct_answer": "Steve Jobs",
     "doc_title": "Steve Jobs"},
    {"question": "Who was the CEO of Microsoft in 2010?",
     "correct_answer": "Steve Ballmer",
     "doc_title": "Steve Ballmer"},
    {"question": "Who was the CEO of Google in 2010?",
     "correct_answer": "Eric Schmidt",
     "doc_title": "Eric Schmidt"},
    # Sports — past
    {"question": "Who was the world number one men's tennis player in 2015?",
     "correct_answer": "Novak Djokovic",
     "doc_title": "Novak Djokovic"},
    {"question": "Who won the 2011 Cricket World Cup?",
     "correct_answer": "India",
     "doc_title": "2011 Cricket World Cup"},
    {"question": "Who won the 2019 Cricket World Cup?",
     "correct_answer": "England",
     "doc_title": "2019 Cricket World Cup"},
    {"question": "Who won the 2018 FIFA World Cup?",
     "correct_answer": "France",
     "doc_title": "2018 FIFA World Cup"},
    {"question": "Who was the Formula One world champion in 2020?",
     "correct_answer": "Lewis Hamilton",
     "doc_title": "Lewis Hamilton"},
    # History with specific time
    {"question": "Who led India during the 1971 war?",
     "correct_answer": "Indira Gandhi",
     "doc_title": "Indira Gandhi"},
    {"question": "Who was the President of USA during World War II?",
     "correct_answer": "Franklin D. Roosevelt",
     "doc_title": "Franklin D. Roosevelt"},
    {"question": "Who was the Prime Minister of UK during World War II?",
     "correct_answer": "Winston Churchill",
     "doc_title": "Winston Churchill"},
]

# ─────────────────────────────────────────────────────────────────────────
# GROUP 4 — UNDATED-BUT-DYNAMIC (15 questions)
# ─────────────────────────────────────────────────────────────────────────
UNDATED_DYNAMIC_QUESTIONS = [
    {"question": "Who heads ISRO?",
     "correct_answer": "V. Narayanan", "doc_title": "Indian Space Research Organisation"},
    {"question": "Who runs Anthropic?",
     "correct_answer": "Dario Amodei", "doc_title": "Anthropic"},
    {"question": "Who leads OpenAI?",
     "correct_answer": "Sam Altman", "doc_title": "OpenAI"},
    {"question": "Who manages the Indian cricket team?",
     "correct_answer": "Gautam Gambhir", "doc_title": "India national cricket team"},
    {"question": "Who is the richest person in the world?",
     "correct_answer": "Elon Musk", "doc_title": "Elon Musk"},
    {"question": "Who is the world chess champion?",
     "correct_answer": "Gukesh", "doc_title": "World Chess Championship 2024"},
    {"question": "Who commands the Indian Army?",
     "correct_answer": "Upendra Dwivedi", "doc_title": "Indian Army"},
    {"question": "Who is handling the opposition in the Assam assembly?",
     "correct_answer": "Congress", "doc_title": "Assam Legislative Assembly"},
    {"question": "Who is the governor of the Reserve Bank of India?",
     "correct_answer": "Sanjay Malhotra", "doc_title": "Reserve Bank of India"},
    {"question": "Who is the chairman of SEBI?",
     "correct_answer": "Tuhin Kanta Pandey",
     "doc_title": "Securities and Exchange Board of India"},
    {"question": "Who is the CEO of SpaceX?",
     "correct_answer": "Elon Musk", "doc_title": "SpaceX"},
    {"question": "Who is the CEO of Nvidia?",
     "correct_answer": "Jensen Huang", "doc_title": "Nvidia"},
    {"question": "Who is the CEO of Amazon?",
     "correct_answer": "Andy Jassy", "doc_title": "Amazon (company)"},
    {"question": "Who is the world number one women's tennis player?",
     "correct_answer": "Aryna Sabalenka", "doc_title": "Aryna Sabalenka"},
    {"question": "Who is the reigning Wimbledon men's singles champion?",
     "correct_answer": "Carlos Alcaraz", "doc_title": "Carlos Alcaraz"},
]

# ─────────────────────────────────────────────────────────────────────────
# GROUP 5 — NO-MEANINGFUL-DATE CONTROL (10 questions)
# ─────────────────────────────────────────────────────────────────────────
NO_DATE_CONTROL_QUESTIONS = [
    {"question": "How does photosynthesis work?",
     "correct_answer": "chlorophyll", "doc_title": "Photosynthesis"},
    {"question": "What causes earthquakes?",
     "correct_answer": "tectonic", "doc_title": "Earthquake"},
    {"question": "How do vaccines work?",
     "correct_answer": "immune", "doc_title": "Vaccine"},
    {"question": "What is DNA made of?",
     "correct_answer": "nucleotide", "doc_title": "DNA"},
    {"question": "Why is the sky blue?",
     "correct_answer": "scattering", "doc_title": "Rayleigh scattering"},
    {"question": "What is the theory of relativity?",
     "correct_answer": "Einstein", "doc_title": "Theory of relativity"},
    {"question": "How does a black hole form?",
     "correct_answer": "gravity", "doc_title": "Black hole"},
    {"question": "What is machine learning?",
     "correct_answer": "algorithm", "doc_title": "Machine learning"},
    {"question": "How does encryption work?",
     "correct_answer": "key", "doc_title": "Encryption"},
    {"question": "What is the greenhouse effect?",
     "correct_answer": "atmosphere", "doc_title": "Greenhouse effect"},
]


# ─────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────

def build_benchmark():
    os.makedirs("data/benchmark", exist_ok=True)
    benchmark = []
    qid = 1

    # Group 1
    print("\n=== Group 1: Static (30 questions) ===")
    for q in STATIC_QUESTIONS:
        benchmark.append({
            "question_id":    f"q{str(qid).zfill(3)}",
            "question":       q["question"],
            "correct_answer": q["correct_answer"],
            "true_type":      "STATIC",
            "group":          1,
            "doc_title":      q["doc_title"],
            "notes":          "Time-invariant. Penalty must not fire.",
        })
        print(f"  [q{str(qid).zfill(3)}] {q['question'][:60]}")
        qid += 1

    # Group 2
    print("\n=== Group 2: Dynamic-Current (25 questions, fetching revision pairs) ===")
    revision_ok = 0
    for q in DYNAMIC_CURRENT_QUESTIONS:
        print(f"  Fetching: {q['wiki_title']}")
        old_doc = get_old_revision(q["wiki_title"], q["old_before_year"])
        time.sleep(0.4)
        new_doc = get_current_revision(q["wiki_title"])
        time.sleep(0.4)
        has_pair = old_doc is not None and new_doc is not None
        if has_pair:
            revision_ok += 1
        benchmark.append({
            "question_id":       f"q{str(qid).zfill(3)}",
            "question":          q["question"],
            "correct_answer":    q["correct_answer"],
            "true_type":         "DYNAMIC_CURRENT",
            "group":             2,
            "doc_title":         q["wiki_title"],
            "old_wrong_answer":  q.get("old_wrong_answer", ""),
            "old_doc_date":      old_doc["date"][:10] if old_doc else None,
            "new_doc_date":      new_doc["date"][:10] if new_doc else None,
            "has_revision_pair": has_pair,
            "notes": f"Old ({q['old_before_year']}) has wrong answer. "
                     f"New has correct. Penalty must boost new doc.",
        })
        status = "✓" if has_pair else "✗ missing"
        print(f"  [q{str(qid).zfill(3)}] {q['question'][:45]} — {status}")
        qid += 1

    # Group 3
    print("\n=== Group 3: Dynamic-Historical (25 questions) ===")
    for q in DYNAMIC_HISTORICAL_QUESTIONS:
        benchmark.append({
            "question_id":    f"q{str(qid).zfill(3)}",
            "question":       q["question"],
            "correct_answer": q["correct_answer"],
            "true_type":      "DYNAMIC_HISTORICAL",
            "group":          3,
            "doc_title":      q["doc_title"],
            "notes":          "Past-time question. Penalty must NOT fire.",
        })
        print(f"  [q{str(qid).zfill(3)}] {q['question'][:60]}")
        qid += 1

    # Group 4
    print("\n=== Group 4: Undated-but-Dynamic (15 questions) ===")
    for q in UNDATED_DYNAMIC_QUESTIONS:
        benchmark.append({
            "question_id":    f"q{str(qid).zfill(3)}",
            "question":       q["question"],
            "correct_answer": q["correct_answer"],
            "true_type":      "DYNAMIC_CURRENT",
            "group":          4,
            "doc_title":      q["doc_title"],
            "notes":          "No temporal keyword. Tests semantic vs syntactic gate.",
        })
        print(f"  [q{str(qid).zfill(3)}] {q['question'][:60]}")
        qid += 1

    # Group 5
    print("\n=== Group 5: No-meaningful-date control (10 questions) ===")
    for q in NO_DATE_CONTROL_QUESTIONS:
        benchmark.append({
            "question_id":    f"q{str(qid).zfill(3)}",
            "question":       q["question"],
            "correct_answer": q["correct_answer"],
            "true_type":      "STATIC",
            "group":          5,
            "doc_title":      q["doc_title"],
            "notes":          "Purely timeless. Classifier must say STATIC.",
        })
        print(f"  [q{str(qid).zfill(3)}] {q['question'][:60]}")
        qid += 1

    # Save
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"BENCHMARK BUILT — {len(benchmark)} questions")
    print(f"{'='*60}")
    groups = {}
    for q in benchmark:
        g = q["group"]
        groups[g] = groups.get(g, 0) + 1
    names = {
        1: "Static",
        2: "Dynamic-Current (revision pairs)",
        3: "Dynamic-Historical",
        4: "Undated-but-Dynamic",
        5: "No-meaningful-date control",
    }
    for g, c in sorted(groups.items()):
        print(f"  Group {g} ({names[g]}): {c}")
    print(f"\nRevision pairs fetched: {revision_ok}/{len(DYNAMIC_CURRENT_QUESTIONS)}")
    print(f"Saved to: {OUTPUT}")
    print(f"\nNext: python benchmark/run_evaluation.py")


if __name__ == "__main__":
    build_benchmark()