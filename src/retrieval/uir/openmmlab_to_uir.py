from __future__ import annotations
import json
import yaml
import hashlib
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable, Tuple
import re
import ast

@dataclass
class UIRSource:
    provider: str # openmmlab
    repo: str # mmdetection
    branch: str # main
    metafile_local: str # local path to metafile.yml
    metafile_repo_path: Optional[str] = None # configs/atss/metafile.yml
    config_local: Optional[str] = None # local path to config .pys
    config_repo_path: Optional[str] = None #c onfigs/atss/atss_r50_fpn_1x_coco.py

@dataclass
class UIRMetrics:
    task: str
    dataset: str
    metrics: Dict[str, float] # {"box AP": 39.4}


@dataclass
class UIRArch:
    model_type: Optional[str] = None # "ATSS"
    detector: Optional[str] = None # model_type
    backbone: Optional[str] = None # "ResNet"
    neck: Optional[str] = None # "FPN"
    head: Optional[str] = None # "ATSSHead", "RPNHead", "ROIHead"
    roi_head: Optional[str] = None
    bbox_head: Optional[str] = None
    mask_head: Optional[str] = None
    rpn_head: Optional[str] = None
    components: Dict[str, str] = field(default_factory=dict)


@dataclass
class UIRRecord:
    doc_id: str
    name: str
    collection: Optional[str]
    paper_url: Optional[str]
    weights_url: Optional[str]
    config_repo_path: Optional[str]
    config_local: Optional[str]
    metadata: Dict[str, Any]
    results: List[UIRMetrics]
    arch: UIRArch
    text: str # text description for embedding & LLM
    source: UIRSource

def find_repo_root(start: Path) -> Path:
    cur = start if start.is_dir() else start.parent
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return cur

# sha1 hash
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def safe_load_yaml_file(path: Path) -> Any:
    return yaml.safe_load(read_text(path))

def normalize_slashes(s: str) -> str:
    return s.replace("\\", "/")

def infer_metafile_repo_path(metafile_path: Path) -> Optional[str]:
    # .../metafiles/configs/atss/metafile.yml -> configs/atss/metafile.yml
    s = normalize_slashes(metafile_path.as_posix())
    m = re.search(r"/metafiles/(configs/.+/metafile\.ya?ml)$", s)
    if m:
        return m.group(1)
    return None

# parse yml
def parse_metafile(metafile_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # return 1. collection map 2. model list
    yml_obj = safe_load_yaml_file(metafile_path)
    if not isinstance(yml_obj, dict):
        return ({}, [])
    collections = yml_obj.get("Collections") or []
    
    col_map: Dict[str, Dict[str, Any]] = {} # 2-stage dict col_map
    if isinstance(collections, list):
        for c in collections:
            if isinstance(c, dict) and isinstance(c.get("Name"), str):
                col_map[c["Name"]] = c

    models = yml_obj.get("Models") or []
    if not isinstance(models, list):
        models = []

    return ({"collections_by_name": col_map, "raw": yml_obj}, models)

def normalize_config_py_path(metafile_path: Path, config_field: str) -> str:
    """
    config_field e.g.:
      - "configs/atss/atss_r50_fpn_1x_coco.py"
      - "mask-rcnn_r50_fpn_albu-1x_coco.py"
    """
    cf = config_field.strip().replace("\\", "/")
    if cf.startswith("./"):
        cf = cf[2:]
    if cf.startswith("configs/"):
        return cf

    mf = metafile_path.as_posix().replace("\\", "/")
    m = re.search(r"/configs/([^/]+)/metafile\.ya?ml$", mf)
    if m:
        folder = m.group(1)
        return f"configs/{folder}/{cf}"

    return f"configs/unknown/{cf}"


# ast
def _ast_get_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None

def _literal_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None

def _literal_str_list(node: ast.AST) -> Optional[List[str]]:
    if isinstance(node, (ast.List, ast.Tuple)):
        out: List[str] = []
        for elt in node.elts:
            s = _literal_str(elt)
            if s is None:
                return None
            out.append(s)
        return out
    return None

# open base
def extract_base_paths_from_config(cfg_text: str) -> List[str]:
    try:
        tree = ast.parse(cfg_text)
    except SyntaxError:
        return []

    for node in tree.body:
        # _base_ = 
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_base_":
                    s = _literal_str(node.value)
                    if s is not None:
                        return [s]
                    lst = _literal_str_list(node.value)
                    if lst is not None:
                        return lst
    return []

def resolve_config_local_path(configs_root: Path, cfg_repo_path: str) -> Optional[Path]:
    # case 1: configs_root = repo root 
    # .../mmdetection
    p1 = configs_root / cfg_repo_path

    # case 2: configs_root = .../mmdetection/configs
    stripped = cfg_repo_path.replace("configs/", "", 1) if cfg_repo_path.startswith("configs/") else cfg_repo_path
    p2 = configs_root / stripped

    if p1.exists():
        return p1
    if p2.exists():
        return p2
    return None

def normalize_base_config_repo_path(curr_cfg_repo_path: str, base_ref: str) -> str:
    """
    curr_cfg_repo_path: configs/atss/atss_r101_fpn_1x_coco.py
    base_ref: ./atss_r50_fpn_1x_coco.py or ../_base_/models/faster-rcnn_r50_fpn.py
    """
    curr = Path(curr_cfg_repo_path)
    if base_ref.startswith("configs/"):
        return normalize_slashes(base_ref)

    # resolve repo-style relative
    resolved = (curr.parent / base_ref).as_posix()
    return normalize_slashes(str(Path(resolved)))

# DFS _base_ tree, extract 'type'
def extract_types_with_bases(
    cfg_repo_path: str,
    configs_root: Path,
    visited: Optional[set[str]] = None,
) -> Dict[str, str]:
# child overrides base
    if visited is None:
        visited = set()
    if cfg_repo_path in visited:
        return {}
    visited.add(cfg_repo_path)

    local_path = resolve_config_local_path(configs_root, cfg_repo_path)
    if local_path is None:
        return {}

    cfg_text = read_text(local_path)

    # parse current file types
    current = extract_all_types_from_config(cfg_text)

    # parse bases
    merged: Dict[str, str] = {}
    base_refs = extract_base_paths_from_config(cfg_text)

    for b in base_refs:
        b_repo = normalize_base_config_repo_path(cfg_repo_path, b)
        base_types = extract_types_with_bases(b_repo, configs_root, visited)
        merged.update(base_types)

    # current overrides base
    merged.update(current)
    return merged

def _extract_types_from_node(node: ast.AST, path: str, out: Dict[str, str]) -> None:
    # dict()
    if isinstance(node, ast.Call) and _ast_get_name(node.func) == "dict":
        # keyword
        kw_map: Dict[str, ast.AST] = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            kw_map[kw.arg] = kw.value
        # e.g.
        # input: dict(type='ResNet', depth=50)
        # keyword(arg="type", value=Constant("ResNet"))
        # keyword(arg="depth", value=Constant(50))

        # "type="
        tnode = kw_map.get("type")
        tval = _literal_str(tnode) if tnode is not None else None
        if tval:
            out[path] = tval

        # recurse
        for k, v in kw_map.items():
            _extract_types_from_node(v, f"{path}.{k}", out)
        return

    if isinstance(node, ast.Dict):
        pairs: List[Tuple[str, ast.AST]] = []
        for k_node, v_node in zip(node.keys, node.values):
            if k_node is None:
                continue
            k = _literal_str(k_node)
            if k is None:
                continue
            pairs.append((k, v_node))

        for k, v in pairs:
            if k == "type":
                tval = _literal_str(v)
                if tval:
                    out[path] = tval
                break

        for k, v in pairs:
            _extract_types_from_node(v, f"{path}.{k}", out)
        return

    # list / tuple
    if isinstance(node, (ast.List, ast.Tuple)):
        for i, elt in enumerate(node.elts):
            _extract_types_from_node(elt, f"{path}[{i}]", out)
        return

    return


# "type="
def extract_all_types_from_config(cfg_text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        tree = ast.parse(cfg_text)
    except SyntaxError:
        return out

    for node in tree.body: # find model dict
        # model = dict()
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "model":
                    _extract_types_from_node(node.value, "model", out)
                    return out
        # model: dict = 
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "model":
                if node.value is not None:
                    _extract_types_from_node(node.value, "model", out)
                return out

    return out

def parse_arch_from_config(
    cfg_text: str,
    cfg_repo_path: Optional[str],
    configs_root: Optional[Path],
) -> UIRArch:
    arch = UIRArch()

    # fallback: only current file
    comps = extract_all_types_from_config(cfg_text)

    # better: recursive base resolution
    if cfg_repo_path and configs_root is not None:
        try:
            comps2 = extract_types_with_bases(cfg_repo_path, configs_root)
            if comps2:
                comps = comps2
        except Exception:
            pass

    arch.components = comps
    arch.model_type = comps.get("model")
    arch.detector = arch.model_type
    arch.backbone = comps.get("model.backbone")
    arch.neck = comps.get("model.neck")
    arch.rpn_head = comps.get("model.rpn_head")
    arch.roi_head = comps.get("model.roi_head")
    arch.bbox_head = comps.get("model.bbox_head") or comps.get("model.roi_head.bbox_head")
    arch.mask_head = comps.get("model.mask_head") or comps.get("model.roi_head.mask_head")
    arch.head = arch.bbox_head or arch.roi_head or arch.rpn_head or arch.mask_head
    return arch

def model_entry_to_uir(
    model_entry: Dict[str, Any],
    metafile_path: Path,
    collections_by_name: Dict[str, Dict[str, Any]],
    repo: str,
    branch: str,
    configs_root: Optional[Path],
) -> UIRRecord:
    name = str(model_entry.get("Name", "")).strip()

    collection = model_entry.get("In Collection")
    if not isinstance(collection, str):
        collection = None
    else:
        collection = collection.strip()

    cfg_field = model_entry.get("Config")
    cfg_repo_path = None
    cfg_local = None

    if isinstance(cfg_field, str) and cfg_field.strip():
        cfg_repo_path = normalize_config_py_path(metafile_path, cfg_field)

        if configs_root is not None:
            # case 1: configs_root = repo root (.../mmdetection)
            p1 = configs_root / cfg_repo_path

            # case 2: configs_root = repo_root/configs
            stripped = cfg_repo_path.replace("configs/", "", 1) if cfg_repo_path.startswith("configs/") else cfg_repo_path
            p2 = configs_root / stripped

            if p1.exists():
                cfg_local = str(p1)
            elif p2.exists():
                cfg_local = str(p2)

    weights_url = model_entry.get("Weights") if isinstance(model_entry.get("Weights"), str) else None

    metadata = model_entry.get("Metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    # results
    results_obj = model_entry.get("Results") or []
    results: List[UIRMetrics] = []
    if isinstance(results_obj, list):
        for rr in results_obj:
            if not isinstance(rr, dict):
                continue
            task = str(rr.get("Task", "unknown"))
            dataset = str(rr.get("Dataset", "unknown"))
            metrics_obj = rr.get("Metrics") or {}
            metric_map: Dict[str, float] = {}
            if isinstance(metrics_obj, dict):
                for k, v in metrics_obj.items():
                    try:
                        metric_map[str(k)] = float(v)
                    except Exception:
                        continue
            results.append(UIRMetrics(task=task, dataset=dataset, metrics=metric_map))

    # parse config -> arch
    arch = UIRArch()
    if cfg_local:
        cfg_text = read_text(Path(cfg_local))
        arch = parse_arch_from_config(cfg_text, cfg_repo_path, configs_root)

    # merge collection info into metadata
    paper_url = None
    if collection and collection in collections_by_name:
        c = collections_by_name[collection]
        metadata = {
            **metadata,
            "_collection_metadata": c.get("Metadata"),
            "_collection_paper": c.get("Paper"),
            "_collection_readme": c.get("README"),
            "_collection_code": c.get("Code"),
        }
        cp = c.get("Paper")
        if isinstance(cp, dict) and cp.get("URL"):
            paper_url = cp.get("URL")
        elif isinstance(cp, str):
            paper_url = str(cp)

    mp = model_entry.get("Paper")
    if isinstance(mp, dict) and mp.get("URL"):
        paper_url = mp.get("URL")
    elif isinstance(mp, str):
        paper_url = str(mp)

    src = UIRSource(
        provider="openmmlab",
        repo=repo,
        branch=branch,
        metafile_local=str(metafile_path),
        metafile_repo_path=infer_metafile_repo_path(metafile_path),
        config_local=cfg_local,
        config_repo_path=cfg_repo_path,
    )

    doc_id = sha1(f"{repo}|{branch}|{metafile_path}|{name}|{cfg_repo_path}")

    # retrieval text v0
    text_parts = [f"Model: {name}"]
    if collection:
        text_parts.append(f"Collection: {collection}")

    if arch.model_type:
        text_parts.append(f"Detector: {arch.model_type}")
    if arch.backbone:
        text_parts.append(f"Backbone: {arch.backbone}")
    if arch.neck:
        text_parts.append(f"Neck: {arch.neck}")
    if arch.head:
        text_parts.append(f"Head: {arch.head}")

    if "Epochs" in metadata:
        text_parts.append(f"Epochs: {metadata.get('Epochs')}")
    if "Training Memory (GB)" in metadata:
        text_parts.append(f"Training Memory (GB): {metadata.get('Training Memory (GB)')}")

    if cfg_repo_path:
        text_parts.append(f"Config: {cfg_repo_path}")
    if weights_url:
        text_parts.append(f"Weights: {weights_url}")
    if paper_url:
        text_parts.append(f"Paper: {paper_url}")

    for r in results:
        mtxt = ", ".join([f"{k}={v}" for k, v in r.metrics.items()])
        text_parts.append(f"Result: {r.task} on {r.dataset} ({mtxt})")

    text = " | ".join(text_parts)

    return UIRRecord(
        doc_id=doc_id,
        name=name,
        collection=collection,
        paper_url=paper_url,
        weights_url=weights_url,
        config_repo_path=cfg_repo_path,
        config_local=cfg_local,
        metadata=metadata,
        results=results,
        arch=arch,
        text=text,
        source=src,
    )


def build_openmmlab_uir_jsonl(
    metafiles_dir: Path,
    out_jsonl: Path,
    repo: str = "mmdetection",
    branch: str = "main",
    configs_root: Optional[Path] = None,
    limit_models: Optional[int] = None,
) -> Tuple[int, List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    n = 0

    # find all metafile.yml or metafile.yaml
    metafile_paths = sorted(
        list(metafiles_dir.rglob("metafile.yml")) + list(metafiles_dir.rglob("metafile.yaml"))
    )

    for mf in metafile_paths:
        try:
            info, models = parse_metafile(mf)
        except Exception as e:
            failures.append({
                "stage": "parse_metafile",
                "metafile": str(mf),
                "error": repr(e),
            })
            continue

        col_map = info.get("collections_by_name", {})
        if not isinstance(col_map, dict):
            col_map = {}

        if not isinstance(models, list):
            failures.append({
                "stage": "invalid_models_list",
                "metafile": str(mf),
                "error": f"Models is not a list: {type(models).__name__}",
            })
            continue

        for idx, me in enumerate(models):
            if not isinstance(me, dict):
                failures.append({
                    "stage": "invalid_model_entry",
                    "metafile": str(mf),
                    "model_index": idx,
                    "error": f"model entry is not dict: {type(me).__name__}",
                })
                continue

            try:
                rec = model_entry_to_uir(
                    model_entry=me,
                    metafile_path=mf,
                    collections_by_name=col_map,
                    repo=repo,
                    branch=branch,
                    configs_root=configs_root,
                )

                # missing data
                if rec.config_repo_path is None:
                    failures.append({
                        "stage": "missing_config_repo_path",
                        "metafile": str(mf),
                        "model_name": rec.name,
                        "error": "Config field missing or invalid",
                    })

                if rec.config_repo_path and rec.config_local is None:
                    failures.append({
                        "stage": "config_file_not_found",
                        "metafile": str(mf),
                        "model_name": rec.name,
                        "config_repo_path": rec.config_repo_path,
                        "error": "Config .py not found locally",
                    })

                # AST fail to get model type
                if rec.config_local and rec.arch.model_type is None:
                    failures.append({
                        "stage": "ast_parse_no_model_type",
                        "metafile": str(mf),
                        "model_name": rec.name,
                        "config_local": rec.config_local,
                        "config_repo_path": rec.config_repo_path,
                        "error": "Parsed config but no model.type extracted",
                    })

                records.append(asdict(rec))
                n += 1

            except Exception as e:
                failures.append({
                    "stage": "model_entry_to_uir",
                    "metafile": str(mf),
                    "model_index": idx,
                    "model_name": str(me.get("Name", "")) if isinstance(me, dict) else None,
                    "config_field": me.get("Config") if isinstance(me, dict) else None,
                    "error": repr(e),
                })
                continue

            if limit_models is not None and n >= limit_models:
                break

        if limit_models is not None and n >= limit_models:
            break

    write_jsonl(out_jsonl, records)
    return n, failures


if __name__ == "__main__":
    REPO_ROOT = find_repo_root(Path(__file__).resolve())
    metafiles_dir = REPO_ROOT / "data" / "raw" / "openMMLab" / "mmdetection" / "metafiles"
    configs_root  = REPO_ROOT / "data" / "raw" / "openMMLab" / "mmdetection"
    out_jsonl     = REPO_ROOT / "data" / "processed" / "uir" / "mmdetection_uir.jsonl"
    out_fail_json = REPO_ROOT / "data" / "processed" / "uir" / "mmdetection_uir_failures.json"

    print("metafiles_dir:", metafiles_dir)
    print("configs_root :", configs_root)
    print("metafiles exists?", metafiles_dir.exists())

    n, failures = build_openmmlab_uir_jsonl(
        metafiles_dir=metafiles_dir,
        out_jsonl=out_jsonl,
        repo="mmdetection",
        branch="main",
        configs_root=configs_root,
        limit_models=None,
    )
    print(f"Wrote {n} records to {out_jsonl}")

    # failure list
    failure_path = out_fail_json    
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Failure count: {len(failures)}")
    print(f"Failure log saved to: {failure_path}")

    for f in failures[:10]:
        print(" -", f)

    # mmpretrain for image classification
    print("\n" + "=" * 60)
    print("Building UIR for mmpretrain ...")
    print("=" * 60)
    metafiles_dir2 = REPO_ROOT / "data" / "raw" / "openMMLab" / "mmpretrain" / "metafiles"
    configs_root2  = REPO_ROOT / "data" / "raw" / "openMMLab" / "mmpretrain"
    out_jsonl2     = REPO_ROOT / "data" / "processed" / "uir" / "mmpretrain_uir.jsonl"
    out_fail_json2 = REPO_ROOT / "data" / "processed" / "uir" / "mmpretrain_uir_failures.json"

    print("metafiles_dir:", metafiles_dir2)
    print("configs_root :", configs_root2)
    print("metafiles exists?", metafiles_dir2.exists())

    n2, failures2 = build_openmmlab_uir_jsonl(
        metafiles_dir=metafiles_dir2,
        out_jsonl=out_jsonl2,
        repo="mmpretrain",
        branch="main",
        configs_root=configs_root2,
        limit_models=None,
    )
    print(f"Wrote {n2} records to {out_jsonl2}")

    failure_path2 = out_fail_json2
    failure_path2.parent.mkdir(parents=True, exist_ok=True)
    failure_path2.write_text(json.dumps(failures2, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Failure count: {len(failures2)}")
    print(f"Failure log saved to: {failure_path2}")

    for f in failures2[:10]:
        print(" -", f)
