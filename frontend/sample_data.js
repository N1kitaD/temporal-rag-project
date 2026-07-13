// Hardcoded sample response (DYNAMIC_CURRENT) auto-loaded on page open.
window.SAMPLE_RESPONSE = {
  "question": "Who is the current Chief Minister of Assam?",
  "query_type": "DYNAMIC_CURRENT",
  "p_dynamic": 0.92,
  "penalty_status": "Full adaptive penalty applied",
  "answer": "Himanta Biswa Sarma has been the Chief Minister of Assam since May 2021.",
  "source_title": "Chief Minister of Assam",
  "source_date": "2021-06-01",
  "top5_before_rerank": [
    {"rank": 1, "title": "Chief Minister of Assam", "date": "2018-03-01", "score": 0.824},
    {"rank": 2, "title": "Chief Minister of Assam", "date": "2021-06-01", "score": 0.801},
    {"rank": 3, "title": "Chief Minister of Uttar Pradesh", "date": "2022-01-01", "score": 0.743},
    {"rank": 4, "title": "Assam government", "date": "2019-05-01", "score": 0.698},
    {"rank": 5, "title": "List of Chief Ministers", "date": "2023-01-01", "score": 0.671}
  ],
  "top5_after_rerank": [
    {"rank": 1, "title": "Chief Minister of Assam", "date": "2021-06-01", "score": 0.731},
    {"rank": 2, "title": "List of Chief Ministers", "date": "2023-01-01", "score": 0.694},
    {"rank": 3, "title": "Chief Minister of Assam", "date": "2018-03-01", "score": 0.621},
    {"rank": 4, "title": "Assam government", "date": "2019-05-01", "score": 0.598},
    {"rank": 5, "title": "Chief Minister of Uttar Pradesh", "date": "2022-01-01", "score": 0.541}
  ],
  "full_log": {
    "query": "Who is the current Chief Minister of Assam?",
    "classifier": {
      "predicted_type": "DYNAMIC_CURRENT",
      "p_dynamic": 0.92,
      "p_static": 0.05,
      "p_historical": 0.03
    },
    "timestamps": {
      "received_at": "2025-04-11T09:14:22Z",
      "responded_at": "2025-04-11T09:14:24Z"
    },
    "reranker": {
      "lambda": 0.55,
      "reference_date": "2025-04-11",
      "penalty_applied": true
    },
    "candidates": [
      {"title": "Chief Minister of Assam", "date": "2018-03-01", "cosine": 0.824, "freshness": 0.34, "final": 0.621},
      {"title": "Chief Minister of Assam", "date": "2021-06-01", "cosine": 0.801, "freshness": 0.68, "final": 0.731},
      {"title": "Chief Minister of Uttar Pradesh", "date": "2022-01-01", "cosine": 0.743, "freshness": 0.72, "final": 0.541},
      {"title": "Assam government", "date": "2019-05-01", "cosine": 0.698, "freshness": 0.48, "final": 0.598},
      {"title": "List of Chief Ministers", "date": "2023-01-01", "cosine": 0.671, "freshness": 0.83, "final": 0.694}
    ]
  }
};
