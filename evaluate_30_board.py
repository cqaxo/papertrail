"""
Follow-up 1: per-question board on the FULL 30-question set, reranking on vs off.

Answers two things:
  - which questions the cross-encoder promotes and which it demotes
  - whether the headline gain really lives in the 17 questions added after
    the original 13 (the arithmetic claim, now measured instead of derived)

Run from the repo root, in the venv:
    python evaluate_30_board.py
"""
import json, re
from ingest import retrieve, load_vector_store, load_documents, chunk_documents

EVAL_SET_PATH = "eval_set.json"
K = 10
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

def first_hit_rank(chunks, refs):
    for i, chunk in enumerate(chunks, start=1):
        if any(r in normalize(chunk.page_content) for r in refs):
            return i
    return None

vs = load_vector_store()

ranks_off, ranks_on = {}, {}
for item in answerable:
    refs = [normalize(r) for r in item["references"]]
    ranks_off[item["id"]] = first_hit_rank(vs.similarity_search(item["question"], k=K), refs)
    ranks_on[item["id"]] = first_hit_rank(retrieve(item["question"], k=K), refs)

def fmt(r):
    return "miss" if r is None else str(r)

def delta_label(off, on):
    if off == on:
        return "same"
    if off is None:
        return "FIXED"
    if on is None:
        return "BROKE"
    return ("up " if on < off else "down ") + str(abs(on - off))

print("\n=== per-question board (rank of first correct passage, k=10) ===")
print(f"{'id':<5} {'off':>6} {'on':>6}   {'change':<8} set")
for item in answerable:
    qid = item["id"]
    off, on = ranks_off[qid], ranks_on[qid]
    which = "original-13" if qid in ORIGINAL_13 else "added-17"
    print(f"{qid:<5} {fmt(off):>6} {fmt(on):>6}   {delta_label(off, on):<8} {which}")

def metrics(rank_map, ids):
    ids = [i for i in ids if i in rank_map]
    n = len(ids)
    if n == 0:
        return None
    rs = [rank_map[i] for i in ids]
    rec = lambda k: sum(1 for r in rs if r is not None and r <= k) / n
    mrr = sum(1 / r for r in rs if r is not None) / n
    hits3 = sum(1 for r in rs if r is not None and r <= 3)
    return n, rec(3), rec(6), mrr, hits3

def report(label, ids):
    off, on = metrics(ranks_off, ids), metrics(ranks_on, ids)
    if not off:
        return
    n = off[0]
    print(f"\n--- {label} (n={n}) ---")
    print(f"{'':<12}{'R@3':>7}{'R@6':>7}{'MRR':>8}{'hits@3':>9}")
    print(f"{'rerank off':<12}{off[1]:>7.2f}{off[2]:>7.2f}{off[3]:>8.3f}{off[4]:>9}")
    print(f"{'rerank on':<12}{on[1]:>7.2f}{on[2]:>7.2f}{on[3]:>8.3f}{on[4]:>9}")
    print(f"{'delta':<12}{on[1]-off[1]:>+7.2f}{on[2]-off[2]:>+7.2f}{on[3]-off[3]:>+8.3f}{on[4]-off[4]:>+9}")

all_ids = [i["id"] for i in answerable]
report("ALL answerable", all_ids)
report("Original 13", [i for i in all_ids if i in ORIGINAL_13])
report("Added 17", [i for i in all_ids if i not in ORIGINAL_13])

print("\n=== regressions caused by reranking (candidates for pinning) ===")
regressed = [i["id"] for i in answerable
             if ranks_off[i["id"]] is not None
             and (ranks_on[i["id"]] is None or ranks_on[i["id"]] > ranks_off[i["id"]])]
for qid in regressed:
    print(f"  {qid}: {fmt(ranks_off[qid])} -> {fmt(ranks_on[qid])}")
print(f"  total regressed: {len(regressed)}")
