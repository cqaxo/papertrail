import json, re
from ingest import retrieve, load_documents, chunk_documents

EVAL_SET_PATH = "eval_set.json"
K = 10 # retrieve this deep, then read recall at shallower cutoffs

# Normalize text by removing extra whitespace, converting to lowercase, and stripping leading/trailing spaces
def normalize(text):
    return re.sub(r"\s+", " ", text).lower().strip()

# 1. Load the eval set (JSON: questions + gold phrases). Not chunked.
with open(EVAL_SET_PATH) as f:
    eval_data = json.load(f)

# 2. Build the corpus text once, only to verify the gold phrases exist in it.
#    This is the data/ PDF, loaded and chunked the same way the app ingests it.
corpus_text = normalize(" ".join(c.page_content for c in chunk_documents(load_documents())))

# 3. Verification: warn if any reference phrase is missing from the corpus entirely.
for item in eval_data:
    for ref in item["references"]:
        if normalize(ref) not in corpus_text:
            print(f"WARNING: reference missing from corpus ({item['id']}): {ref!r}")

# 4. Score each answerable question.
answerable = [item for item in eval_data if item["answerable"]]
ranks = []  # first-hit rank per question, or None for a miss

for item in answerable:
    refs = [normalize(r) for r in item["references"]]
    rank = None
    for i, chunk in enumerate(retrieve(item["question"], k=K), start=1):
        text = normalize(chunk.page_content)
        if any(ref in text for ref in refs):
            rank = i
            break
    ranks.append(rank)
    print(f"{item['id']}: {'rank ' + str(rank) if rank else 'miss'}")

# 5. Aggregate.
n = len(answerable)
recall_at = lambda k: sum(1 for r in ranks if r is not None and r <= k) / n
mrr = sum(1 / r for r in ranks if r is not None) / n

print(f"\nAnswerable questions: {n}")
print(f"Recall@3: {recall_at(3):.2f}")
print(f"Recall@6: {recall_at(6):.2f}")
print(f"MRR:      {mrr:.3f}")