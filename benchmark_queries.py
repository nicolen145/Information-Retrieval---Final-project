import json
import time
import requests
import pandas as pd

# ===== CONFIG =====
ENGINE_URL = "http://127.0.0.1:8080/search"
QUERIES_FILE = "queries_train.json"
OUT_CSV = "benchmark_results.csv"

# ===== LOAD QUERIES =====
with open(QUERIES_FILE, "r", encoding="utf-8") as f:
    queries = json.load(f)  # dict: {query_text: [relevant_doc_ids_as_str]}

rows = []

print(f"Running {len(queries)} queries...")

for i, (query, rel_docs) in enumerate(queries.items(), start=1):
    true_docs = set(str(d) for d in rel_docs)  # ensure strings

    t0 = time.time()
    try:
        r = requests.get(ENGINE_URL, params={"query": query}, timeout=30)
        elapsed = time.time() - t0
        r.raise_for_status()
        results = r.json()
    except Exception as e:
        elapsed = time.time() - t0
        results = []
        print(f"[ERROR] query #{i}: {query!r} -> {e}")

    # results expected: list of [doc_id, title] pairs
    retrieved = [str(doc_id) for doc_id, _ in results] if results else []

    # Precision@10
    top10 = retrieved[:10]
    p10 = (sum(1 for d in top10 if d in true_docs) / 10) if top10 else 0.0

    rows.append({
        "query": query,
        "latency_sec": elapsed,
        "precision_at_10": p10,
        "num_results": len(retrieved),
    })

    if i % 10 == 0 or i == len(queries):
        print(f"{i}/{len(queries)} done")

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print("Saved:", OUT_CSV)
print(df.describe(include="all"))
