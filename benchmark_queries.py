import json
import time
import requests
import pandas as pd

# ===== CONFIG =====
ENGINE_URL = "http://127.0.0.1:8080/search"
QUERIES_FILE = "queries_train_as_list_int.json"   
OUT_CSV = "benchmark_results.csv"

# ===== LOAD QUERIES =====
with open(QUERIES_FILE, "r") as f:
    queries = json.load(f)

rows = []

print(f"Running {len(queries)} queries...")

for i, q in enumerate(queries):
    query = q["query"]
    true_docs = set(q.get("relevant_docs", []))  

    t0 = time.time()
    r = requests.get(ENGINE_URL, params={"query": query})
    elapsed = time.time() - t0

    results = r.json()
    retrieved = [doc_id for doc_id, _ in results]

    # Precision@10
    if retrieved[:10]:
        p10 = len([d for d in retrieved[:10] if d in true_docs]) / 10
    else:
        p10 = 0.0

    rows.append({
        "query": query,
        "latency_sec": elapsed,
        "precision_at_10": p10
    })

    if i % 10 == 0:
        print(f"{i}/{len(queries)} done")

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print("Saved:", OUT_CSV)
print(df.describe())
