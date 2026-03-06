# rag.py
import argparse
from retrieve import retrieve

def parse_query_to_filters(q: str):
    ql = q.lower()
    filters = {}

    # dataset
    if "imagenet" in ql or "in1k" in ql:
        filters["dataset"] = "ImageNet-1k"
    if "inshop" in ql:
        filters["dataset"] = "InShop"

    # task
    if "classification" in ql or "top-1" in ql or "top1" in ql:
        filters["task"] = "Image Classification"
    if "retrieval" in ql or "recall@1" in ql or "map@" in ql:
        filters["task"] = "Image Retrieval"

    # collections (optional quick rules)
    for col in ["beit", "beitv2", "barlowtwins", "arcface"]:
        if col in ql:
            # your UIR uses "BEiT", "BEiTv2", ...
            mapping = {"beit": "BEiT", "beitv2": "BEiTv2", "barlowtwins": "BarlowTwins", "arcface": "ArcFace"}
            filters["collection"] = mapping[col]
            break

    return filters

def build_prompt(query: str, hits: list[dict]) -> str:
    lines = []
    lines.append("You are a helpful assistant. Answer using ONLY the provided context.")
    lines.append("If the context is insufficient, say what is missing.")
    lines.append("")
    lines.append(f"Question: {query}")
    lines.append("")
    lines.append("Context:")
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] name: {h.get('name')}")
        lines.append(f"    collection: {h.get('collection')}")
        lines.append(f"    task/dataset: {h.get('task')} / {h.get('dataset')}")
        if h.get("metrics"):
            lines.append(f"    metrics: {h.get('metrics')}")
        lines.append(f"    config: {h.get('config_repo_path')}")
        if h.get("weights_url"):
            lines.append(f"    weights: {h.get('weights_url')}")
        lines.append(f"    score: {h.get('score'):.4f}")
        lines.append(f"    summary: {h.get('context_text')}")
        lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uir_path", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    filters = parse_query_to_filters(args.query)
    hits = retrieve(uir_path=args.uir_path, query=args.query, filters=filters, topk=args.topk)
    prompt = build_prompt(args.query, hits)
    print(prompt)

if __name__ == "__main__":
    main()