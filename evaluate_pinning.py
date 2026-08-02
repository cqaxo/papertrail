"""
Follow-up 2: does pinning strong first-stage hits recover the reranker's
regressions without giving up its wins?

The idea: the cross-encoder is trained on web search, not policy prose, so it
demotes some passages plain vector search ranked correctly (q3, q4, q7, q9 on
the original 13). Pinning reserves the first pin_n slots for the first-stage
top pin_n, in first-stage order, then fills the rest with reranked candidates.

pin_n=0 is exactly your current pipeline, so it doubles as the control.
A rerank-off row is printed as the other reference point.

Run from the repo root, in the venv:
    python evaluate_pinning.py
"""
import json, re
from ingest import load_vector_store, get_reranker, load_documents, chunk_documents

EVAL_SET_PATH = "eval_set.json"
K = 10
FETCH_K = 40
PIN_VALUES = [0, 1, 2, 3]
ORIGINAL_13 = {f"q{i}" for i in range(1, 14)}

def normalize(text):
    return re.sub(r"\s+", " ", text).lower().strip()

with open(EVAL_SET_PATH) as f:
    eval_data = json.load(f)

corpus_text = normalize(" ".join(c.page_content for c in chunk_documents(load_documents())))
for item in eval_data:
    for ref in item["references"]:
        if normalize(ref) not in corpus_text:
            print(f"WARNING: reference missing from corpus ({item['id']}): {ref!r}")

answerable = [i for i in eval_data if i["answerable"]]
vs = load_vector_store()

def retrieve_pinned(query, k=K, fetch_k=FETCH_K, pin_n=0):
    """First-stage top pin_n keep their slots; everything else is reranked."""
    candidates = vs.similarity_search(query, k=fetch_k)
    pinned, rest = candidates[:pin_n], candidates[pin_n:]
    if not rest:
        return pinned[:k]
    scores = get_reranker().predict([[query, c.page_content] for c in rest])
    ranked = sorted(zip(rest, scores), key=lambda x: x[1], reverse=True)
    return (pinned + [d for d, _ in ranked])[:k]

def first_hit_rank(chunks, refs):
    for i, chunk in enumerate(chunks, start=1):
        if any(r in normalize(chunk.page_content) for r in refs):
            return i
    return None

def score_all(get_chunks):
    out = {}
    for item in answerable:
        refs = [normalize(r) for r in item["references"]]
        out[item["id"]] = first_hit_rank(get_chunks(item["question"]), refs)
    return out

def metrics(rank_map, ids):
    ids = [i for i in ids if i in rank_map]
    n = len(ids)
    rs = [rank_map[i] for i in ids]
    rec = lambda k: sum(1 for r in rs if r is not None and r <= k) / n
    mrr = sum(1 / r for r in rs if r is not None) / n
    return rec(3), rec(6), mrr

all_ids = [i["id"] for i in answerable]
orig_ids = [i for i in all_ids if i in ORIGINAL_13]
added_ids = [i for i in all_ids if i not in ORIGINAL_13]

runs = {}
print("\nscoring rerank-off reference...")
runs["off"] = score_all(lambda q: vs.similarity_search(q, k=K))
for p in PIN_VALUES:
    print(f"scoring pin_n={p}...")
    runs[f"pin{p}"] = score_all(lambda q, p=p: retrieve_pinned(q, pin_n=p))

def row(label, rank_map):
    a, o, n = metrics(rank_map, all_ids), metrics(rank_map, orig_ids), metrics(rank_map, added_ids)
    print(f"{label:<14}{a[0]:>7.2f}{a[1]:>7.2f}{a[2]:>8.3f}   |{o[0]:>7.2f}{o[2]:>8.3f}   |{n[0]:>7.2f}{n[2]:>8.3f}")

print("\n=== pinning sweep ===")
print(f"{'config':<14}{'R@3':>7}{'R@6':>7}{'MRR':>8}   |{'13 R@3':>7}{'13 MRR':>8}   |{'17 R@3':>7}{'17 MRR':>8}")
row("rerank off", runs["off"])
for p in PIN_VALUES:
    label = f"pin_n={p}" + (" (current)" if p == 0 else "")
    row(label, runs[f"pin{p}"])

print("\n=== per-question ranks across configs ===")
header = f"{'id':<5}{'off':>6}" + "".join(f"{'pin'+str(p):>7}" for p in PIN_VALUES)
print(header)
fmt = lambda r: "miss" if r is None else str(r)
for qid in all_ids:
    line = f"{qid:<5}{fmt(runs['off'][qid]):>6}"
    line += "".join(f"{fmt(runs['pin'+str(p)][qid]):>7}" for p in PIN_VALUES)
    print(line)

print("\n=== what pinning changed vs current pipeline (pin_n=0) ===")
base = runs["pin0"]
for p in PIN_VALUES[1:]:
    cur = runs[f"pin{p}"]
    moved = [q for q in all_ids if cur[q] != base[q]]
    print(f"\npin_n={p}: {len(moved)} question(s) moved")
    for q in moved:
        print(f"  {q}: {fmt(base[q])} -> {fmt(cur[q])}")
