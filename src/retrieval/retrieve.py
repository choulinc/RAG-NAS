# Output: list[dict] hits with score + useful fields for RAG prompt building
# python -m retrieval.retrieve --uir_path data/processed/uir/mmpretrain_uir.jsonl --query "beit imagenet top1" --topk 5
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from rank_bm25 import BM25Okapi

# token utils
TOKEN_RE = re.compile(r"[A-Za-z0-9_@./:+-]+")

def tokenize(s: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(s or "")]

def safe_get(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

# flatten key val
def flatten_kv(x: Any, prefix: str = "") -> List[str]:
    """
    "arch": {
        "components": {
            "model.head.loss": "CrossEntropyLoss",
            "model.backbone": "ResNet"
        }
    }
    =>
    arch.components.model.head.loss: CrossEntropyLoss
    arch.components.model.backbone: ResNet
    metadata.Parameters: 31693888
    results[0].dataset: ImageNet-1k
    ...
    """
    out: List[str] = []
    if isinstance(x, dict):
        for k, v in x.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.extend(flatten_kv(v, p))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            p = f"{prefix}[{i}]"
            out.extend(flatten_kv(v, p))
    else:
        if prefix:
            out.append(f"{prefix}: {x}")
    return out


# UIR -> views for retrieval
def extract_primary_result(u: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """
    Return (task, dataset, metrics) for the first result if exists.
    You can later improve this by choosing the best-matching result per query.
    """
    results = u.get("results", []) or []
    if not results:
        return None, None, {}
    r0 = results[0] or {}
    return r0.get("task"), r0.get("dataset"), (r0.get("metrics", {}) or {})


def uir_to_views(u: Dict[str, Any]) -> Dict[str, str]:
    """
    two text views:
        narrative: short semantic summary (for dense embeddings)
        kv: structure-heavy text (for keyword, BM25-like matching)
    """
    name = u.get("name", "") or ""
    collection = u.get("collection", "") or ""
    cfg = u.get("config_repo_path", "") or ""
    weights = u.get("weights_url", "") or ""
    paper = u.get("paper_url", "") or ""

    arch = u.get("arch", {}) or {}
    arch_bits = []
    for k in ["model_type", "detector", "backbone", "neck", "head"]:
        v = arch.get(k)
        if v:
            arch_bits.append(f"{k}={v}")
    arch_text = "; ".join(arch_bits)

    results = u.get("results", []) or []
    res_bits = []
    for r in results:
        task = r.get("task")
        dataset = r.get("dataset")
        metrics = r.get("metrics", {}) or {}
        metric_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        if task or dataset or metric_str:
            res_bits.append(f"{task} on {dataset} ({metric_str})".strip())
    res_text = " | ".join(res_bits)

    # narrative
    narrative = (
        f"Model: {name} | Collection: {collection} | {arch_text} | "
        f"Config: {cfg} | "
        f"{('Paper: ' + paper + ' | ') if paper else ''}"
        f"{('Weights: ' + weights + ' | ') if weights else ''}"
        f"{('Result: ' + res_text) if res_text else ''}"
    ).strip()

    # kv
    kv_lines: List[str] = []
    kv_lines += [f"name: {name}", f"collection: {collection}", f"config_repo_path: {cfg}"]
    if paper:
        kv_lines.append(f"paper_url: {paper}")
    if weights:
        kv_lines.append(f"weights_url: {weights}")
    kv_lines += flatten_kv(u.get("metadata", {}), "metadata")
    kv_lines += flatten_kv(u.get("arch", {}), "arch")
    kv_lines += flatten_kv(u.get("results", {}), "results")
    kv_lines += flatten_kv(u.get("source", {}), "source")

    kv = "\n".join(kv_lines)
    return {"narrative": narrative, "kv": kv}

# filter
def passes_filters(u: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    # filters results[].task, results[].dataset, collection, source.repo, source.provider
    if not filters:
        return True

    # collection
    col = filters.get("collection")
    if col and (u.get("collection") or "").lower() != str(col).lower():
        return False

    # provider & repo
    provider = filters.get("provider")
    if provider and (safe_get(u, "source.provider", "") or "").lower() != str(provider).lower():
        return False
    repo = filters.get("repo")
    if repo and (safe_get(u, "source.repo", "") or "").lower() != str(repo).lower():
        return False

    # task & dataset
    task = filters.get("task")
    dataset = filters.get("dataset")
    if task or dataset:
        results = u.get("results", []) or []
        ok = False
        for r in results:
            if task and r.get("task") != task:
                continue
            if dataset and r.get("dataset") != dataset:
                continue
            ok = True
            break
        if not ok:
            return False

    return True


# Scoring, Hybrid + Rerank
"""
def keyword_score(query_tokens: List[str], doc_tokens: List[str]) -> float:
    # keyword overlap scoring
    # rank-bm25 / Elastic / OpenSearch
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_set = set(doc_tokens)
    hits = sum(1 for t in query_tokens if t in doc_set)
    return hits / math.sqrt(len(doc_set) + 1.0)
"""

def bm25_scores(query_tokens: List[str], corpus_tokens: List[List[str]]) -> List[float]:
    # BM25 scores for tokenized query against a tokenized corpus
    if not query_tokens or not corpus_tokens:
        return [0.0] * len(corpus_tokens)
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(query_tokens)  # numpy array
    return [float(s) for s in scores]

def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)

# normalize bm25
def minmax_norm(xs: List[float]) -> List[float]:
    if not xs: return xs
    mn, mx = min(xs), max(xs)
    if mx == mn: return [0.0 for _ in xs]
    return [(x - mn) / (mx - mn) for x in xs]

class Embedder:
    """
    sentence-transformers embedder (normalized embeddings).
    """

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError(
                "Missing dependency. Install: pip install sentence-transformers"
            ) from e
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


def make_cache_key(uir_path: str, embed_model: str, view_name: str) -> str:
    # Invalidate cache if uir_path changes (mtime + size) or model/view changes
    st = os.stat(uir_path)
    raw = f"{os.path.abspath(uir_path)}|{st.st_mtime_ns}|{st.st_size}|{embed_model}|{view_name}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_or_build_embedding_cache(
    *,
    uirs: List[Dict[str, Any]],
    uir_path: str,
    embed_model: str,
    view_name: str = "narrative",
    cache_dir: str = ".cache/rag_embeddings",
) -> Tuple[List[List[float]], List[str]]:
    """
    Build embeddings for each UIR (chosen view) once and cache them on disk.
    Returns (embeddings, doc_ids_in_same_order).
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = make_cache_key(uir_path, embed_model, view_name)
    cache_path = os.path.join(cache_dir, f"{key}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            obj = pickle.load(f)
        return obj["embeddings"], obj["doc_ids"]

    embedder = Embedder(embed_model)
    texts: List[str] = []
    doc_ids: List[str] = []
    for u in uirs:
        views = uir_to_views(u)
        texts.append(views[view_name])
        doc_ids.append(u.get("doc_id", ""))

    embeddings = embedder.encode(texts)

    with open(cache_path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "doc_ids": doc_ids}, f)

    return embeddings, doc_ids


# -------------------------
# Public API
# -------------------------
def retrieve(
    *,
    uir_path: str,
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    topk: int = 5,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    dense_weight: float = 0.55,
    kw_weight: float = 0.45,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k OpenMMLab UIR entries for a query.

    - filters supports: task, dataset, collection, provider, repo
    - dense uses the narrative view
    - keyword uses the kv view
    - rerank uses simple rule boosts (dataset/task exact matches, mention of collection/name)

    Returns list of hits, each hit is a dict:
      {doc_id, name, collection, task, dataset, metrics, config_repo_path, weights_url, score, context_text}
    """
    # Load UIR jsonl
    uirs: List[Dict[str, Any]] = []
    with open(uir_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            uirs.append(json.loads(line))

    # Filter
    cand_uirs = [u for u in uirs if passes_filters(u, filters)]
    if not cand_uirs:
        return []

    # Prepare candidates and keyword scores
    q_tokens = tokenize(query)
    candidates: List[Dict[str, Any]] = []
    kv_token_corpus: List[List[str]] = []
    for u in cand_uirs:
        views = uir_to_views(u)
        task0, dataset0, metrics0 = extract_primary_result(u)

        kv_tokens = tokenize(views["kv"])
        kv_token_corpus.append(kv_tokens)

        candidates.append(
            {
                "uir": u,
                "narrative": views["narrative"],
                "kv": views["kv"],
                "task": task0,
                "dataset": dataset0,
                "metrics": metrics0,
                "kw_score": 0.0, # will be modified after BM25
                "dense_score": 0.0,
                "final_score": 0.0,
            }
        )

    # BM25 scores on kv
    kw_scores = minmax_norm(bm25_scores(q_tokens, kv_token_corpus))
    for c, s in zip(candidates, kw_scores):
        c["kw_score"] = s

    # Dense scores
    if dense_weight > 0:
        if use_cache:
            # build/load cached embeddings for ALL filtered candidates, aligned by doc_id order in cand_uirs
            # (cache is based on uir_path; filtering doesn't change the cached full-file order)
            # To keep MVP simple: we cache for the full file, then map doc_id -> embedding.
            full_embeddings, full_doc_ids = load_or_build_embedding_cache(
                uirs=uirs, uir_path=uir_path, embed_model=embed_model, view_name="narrative"
            )
            emb_by_id = {did: emb for did, emb in zip(full_doc_ids, full_embeddings)}
            embedder = Embedder(embed_model)  # for query only
            q_vec = embedder.encode([query])[0]
            for c in candidates:
                did = c["uir"].get("doc_id", "")
                c["dense_score"] = cosine(q_vec, emb_by_id.get(did, []))
        else:
            embedder = Embedder(embed_model)
            q_vec = embedder.encode([query])[0]
            doc_vecs = embedder.encode([c["narrative"] for c in candidates])
            for c, v in zip(candidates, doc_vecs):
                c["dense_score"] = cosine(q_vec, v)

    # Combine + rule-based rerank boosts
    ql = query.lower()
    task_f = (filters or {}).get("task")
    dataset_f = (filters or {}).get("dataset")

    for c in candidates:
        score = kw_weight * c["kw_score"] + dense_weight * c["dense_score"]

        # boosts
        if task_f and c["task"] == task_f:
            score *= 1.08
        if dataset_f and c["dataset"] == dataset_f:
            score *= 1.08

        u = c["uir"]
        collection = (u.get("collection") or "")
        name = (u.get("name") or "")
        if collection and collection.lower() in ql:
            score *= 1.05
        if name and name.lower() in ql:
            score *= 1.05

        # If query mentions a metric name that appears in metrics, boost slightly
        metrics = c["metrics"] or {}
        for mk in metrics.keys():
            if str(mk).lower() in ql:
                score *= 1.03
                break

        c["final_score"] = score

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    top = candidates[: max(1, topk)]

    hits: List[Dict[str, Any]] = []
    for c in top:
        u = c["uir"]
        hits.append(
            {
                "doc_id": u.get("doc_id", ""),
                "name": u.get("name", ""),
                "collection": u.get("collection", ""),
                "task": c["task"],
                "dataset": c["dataset"],
                "metrics": c["metrics"],
                "config_repo_path": u.get("config_repo_path", ""),
                "weights_url": u.get("weights_url", ""),
                "paper_url": u.get("paper_url", ""),
                "score": float(c["final_score"]),
                "context_text": c["narrative"] or (u.get("text", "") or ""),
                # Optional debug fields:
                "debug": {
                    "dense_score": float(c["dense_score"]),
                    "kw_score": float(c["kw_score"]),
                },
            }
        )

    return hits


# CLI 
    ap.add_argument("--task", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--collection", default=None)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--repo", default=None)

    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--dense_weight", type=float, default=0.55)
    ap.add_argument("--kw_weight", type=float, default=0.45)
    ap.add_argument("--no_cache", action="store_true")
    args = ap.parse_args()

    filters = {
        k: v
        for k, v in {
            "task": args.task,
            "dataset": args.dataset,
            "collection": args.collection,
            "provider": args.provider,
            "repo": args.repo,
        }.items()
        if v is not None
    }

    hits = retrieve(
        uir_path=args.uir_path,
        query=args.query,
        filters=filters or None,
        topk=args.topk,
        embed_model=args.embed_model,
        dense_weight=args.dense_weight,
        kw_weight=args.kw_weight,
        use_cache=(not args.no_cache),
    )

    if not hits:
        print("No results.")
        return

    print("\n=== Top Results ===")
    for i, h in enumerate(hits, 1):
        d = h.get("debug", {})
        print(f"\n[{i}] score={h['score']:.4f} (dense={d.get('dense_score',0):.4f}, kw={d.get('kw_score',0):.4f})")
        print(f"    name: {h['name']}")
        print(f"    collection: {h['collection']}")
        print(f"    task/dataset: {h.get('task')} / {h.get('dataset')}")
        print(f"    config: {h.get('config_repo_path')}")
        if h.get("paper_url"):
            print(f"    paper: {h.get('paper_url')}")
        if h.get("weights_url"):
            print(f"    weights: {h.get('weights_url')}")
        print(f"    doc_id: {h.get('doc_id')}")
        print(f"    summary: {h.get('context_text')}")

if __name__ == "__main__":
    _cli()