from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from pathlib import Path
from tqdm import tqdm
import json
import requests
import hashlib
import yaml
import re
import ast

GITHUB_RAW = "https://raw.githubusercontent.com"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "RAGNAS/0.1"})

@dataclass
class RepoSpec:
    author: str = "open-mmlab"
    repo: str = "mmdetection"
    branch: str = "main"

class OpenMMLabSource:
    def __init__(self, out_root: Path, repo: RepoSpec = RepoSpec()):
        self.out_root = out_root
        self.repo = repo
    def crawl_modelzoo(self, max_mfs: Optional[int] = None) -> list[str]:
        base = self.out_root / self.repo.repo
        base.mkdir(parents=True, exist_ok = True)
        url = raw_url(self.repo, "model-index.yml")
        text = http_get_text(url)
        save_text(base / "model-index.yml", text)
        
        metafiles = get_metafiles(text)
        if max_mfs is not None:
            metafiles = metafiles[:max_mfs]
        return metafiles

def raw_url(spec: RepoSpec, path: str) -> str:
    # input spec & path -> github raw url
    path = path.lstrip("/")
    return f"{GITHUB_RAW}/{spec.author}/{spec.repo}/{spec.branch}/{path}"

def http_get_text(url: str, timeout: int = 30) -> str:
    # input url then request it -> text
    response = SESSION.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text

def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def load_yaml(text: str) ->Any:
    return yaml.safe_load(text)

def find_repo_root(start: Path) -> Path:
    cur = start if start.is_dir() else start.parent
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return cur

def download_one_metafile(spec: RepoSpec, mf_path: str, save_root: Path) -> Path:
    local_path = save_root / mf_path
    if local_path.exists():
        return local_path
    url = raw_url(spec, mf_path)
    text = http_get_text(url)
    save_text(local_path, text)
    return local_path

def get_metafiles(model_index_yaml) -> list[str]:
    obj: Any = yaml.safe_load(model_index_yaml)
    if not isinstance(obj, dict):
        # ensure if it is dict
        return []
    
    imports = obj.get("Import") or obj.get("Imports")
    if not isinstance(imports, list):
        # ensure if it is list
        return []

    out: list[str] = []
    for item in imports:
        if not isinstance(item, str):
            continue
        metafile = item.strip().replace("\\", "/")
        if metafile.endswith((".yml", "yaml")):
            out.append(metafile.lstrip("./"))
    return out

def download_metafiles(spec: RepoSpec, metafiles: list[str], base_dir = Path) -> tuple[list[Path], list[dict]]:
    dest_dir = base_dir / "metafiles"
    saved: list[Path] = []
    failed: list[dict] = []
    seen: set[str] = set()
    print(f"dest_dir:{dest_dir}")
    for mf in tqdm(metafiles, desc="Downloading metafiles", unit="file"):
        if mf in seen:
            continue
        seen.add(mf)

        try:
            p = download_one_metafile(spec, mf, dest_dir)
            saved.append(p)
        except Exception as e:
            failed.append({"metafile": mf, "error": repr(e)})

    return saved, failed

def collect_config_paths_from_metafiles(
    metafile_paths: list[Path],
) -> tuple[list[str], list[dict]]:
    all_cfgs: list[str] = []
    failed: list[dict] = []
    seen: set[str] = set()

    for mf in tqdm(metafile_paths, desc="Parsing metafiles", unit="file"):
        try:
            cfg_paths = get_config_paths_from_metafile_file(mf)
        except Exception as e:
            failed.append({"metafile": str(mf), "error": f"parse metafile failed: {repr(e)}"})
            continue

        for cfg in cfg_paths:
            if cfg not in seen:
                seen.add(cfg)
                all_cfgs.append(cfg)

    return all_cfgs, failed

def download_configs(
    spec: RepoSpec,
    config_repo_paths: list[str],
    base_dir: Path,
    max_configs: Optional[int] = None,
) -> tuple[list[Path], list[dict]]:
    save_root = base_dir
    saved: list[Path] = []
    failed: list[dict] = []

    paths = config_repo_paths
    if max_configs is not None:
        paths = paths[:max_configs]

    for cfg in tqdm(paths, desc="Downloading config .py", unit="file"):
        try:
            p = download_one_config_py(spec, cfg, save_root)
            saved.append(p)
        except Exception as e:
            failed.append({"config": cfg, "error": repr(e)})

    return saved, failed

def normalize_config_path_from_metafile(metafile_local_path: Path, config_field: str) -> str:
    # normalize_config_path into repo path configs/.../*.py
    cf = config_field.strip().replace("\\", "/")
    if cf.startswith("./"):
        cf = cf[2:]

    if cf.startswith("configs/"):
        return cf

    # e.g. .../metafiles/configs/foveabox/metafile.yml + "fovea_xxx.py"
    # -> configs/foveabox/fovea_xxx.py
    s = metafile_local_path.as_posix().replace("\\", "/")
    m = re.search(r"/metafiles/(configs/.+)/metafile\.ya?ml$", s)
    if m:
        folder = m.group(1)
        return f"{folder}/{cf}"

    return f"configs/unknown/{cf}"


def get_config_paths_from_metafile_file(metafile_local_path: Path) -> list[str]:
    # read metafile.yml, get Models[*].Config repo path
    text = metafile_local_path.read_text(encoding="utf-8", errors="ignore")
    obj = yaml.safe_load(text)

    if not isinstance(obj, dict):
        return []

    models = obj.get("Models") or []
    if not isinstance(models, list):
        return []

    out: list[str] = []
    seen: set[str] = set()

    for model in models:
        if not isinstance(model, dict):
            continue
        cfg = model.get("Config")
        if not isinstance(cfg, str):
            continue

        cfg_repo_path = normalize_config_path_from_metafile(metafile_local_path, cfg)
        if not cfg_repo_path.endswith(".py"):
            continue

        if cfg_repo_path not in seen:
            seen.add(cfg_repo_path)
            out.append(cfg_repo_path)

    return out

def download_one_config_py(spec: RepoSpec, config_repo_path: str, save_root: Path) -> Path:
    local_path = save_root / config_repo_path
    if local_path.exists():
        return local_path
    url = raw_url(spec, config_repo_path)
    text = http_get_text(url)
    save_text(local_path, text)
    return local_path

def download_configs_from_metafiles(
    spec: RepoSpec,
    metafile_paths: list[Path],
    base_dir: Path,
    max_configs: Optional[int] = None,
) -> tuple[list[Path], list[dict]]:
    # config.py from metafile.yml to base_dir/configs/...
    save_root = base_dir

    saved: list[Path] = []
    failed: list[dict] = []
    seen: set[str] = set()
    n = 0

    for mf in metafile_paths:
        try:
            cfg_paths = get_config_paths_from_metafile_file(mf)
        except Exception as e:
            failed.append({"metafile": str(mf), "error": f"parse metafile failed: {repr(e)}"})
            continue

        for cfg in cfg_paths:
            if cfg in seen:
                continue
            seen.add(cfg)

            try:
                p = download_one_config_py(spec, cfg, save_root)
                saved.append(p)
                n += 1
                if max_configs is not None and n >= max_configs:
                    return saved, failed
            except Exception as e:
                failed.append({"config": cfg, "error": repr(e)})

    return saved, failed

def _parse_base_refs(py_path: Path) -> list[str]:
    """Parse _base_ = [...] from a config .py and return the string literals."""
    try:
        text = py_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(text)
    except Exception:
        return []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_base_":
                    # single string
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return [node.value.value]
                    # list of strings
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        out = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                out.append(elt.value)
                        return out
    return []

def _resolve_base_to_repo_path(cfg_repo_path: str, base_ref: str) -> str:
    """Resolve a relative _base_ ref to a repo-level path.
    e.g. cfg_repo_path='configs/resnet/resnet50_8xb16_cifar10.py'
         base_ref='../_base_/models/resnet50.py'
      -> 'configs/_base_/models/resnet50.py'
    """
    if base_ref.startswith("configs/"):
        return base_ref.replace("\\", "/")
    resolved = (Path(cfg_repo_path).parent / base_ref).as_posix()
    # normalize '../' etc.
    parts = []
    for part in resolved.replace("\\", "/").split("/"):
        if part == "..":
            if parts:
                parts.pop()
        elif part and part != ".":
            parts.append(part)
    return "/".join(parts)

def download_base_configs(
    spec: RepoSpec,
    base_dir: Path,
) -> tuple[int, list[dict]]:
    """Recursively scan all downloaded .py configs for _base_ refs and download missing ones."""
    failed: list[dict] = []
    total_new = 0
    round_num = 0

    while True:
        round_num += 1
        # collect all .py files under configs/
        configs_dir = base_dir / "configs"
        if not configs_dir.exists():
            break
        all_py = list(configs_dir.rglob("*.py"))

        new_downloads = 0
        for py_path in all_py:
            base_refs = _parse_base_refs(py_path)
            if not base_refs:
                continue
            # determine this file's repo path
            try:
                rel = py_path.relative_to(base_dir)
            except ValueError:
                continue
            cfg_repo_path = str(rel).replace("\\", "/")

            for ref in base_refs:
                repo_path = _resolve_base_to_repo_path(cfg_repo_path, ref)
                local_path = base_dir / repo_path
                if local_path.exists():
                    continue
                # download
                try:
                    download_one_config_py(spec, repo_path, base_dir)
                    new_downloads += 1
                except Exception as e:
                    failed.append({"base_config": repo_path, "referenced_by": cfg_repo_path, "error": repr(e)})

        total_new += new_downloads
        print(f"  _base_ round {round_num}: downloaded {new_downloads} new files")
        if new_downloads == 0:
            break  # no new files, done

    return total_new, failed

if __name__ == "__main__":
    REPO_ROOT = find_repo_root(Path(__file__).resolve())
    out_root = REPO_ROOT / "data" / "raw" / "openMMLab"
    # mmdetection
    repo = RepoSpec(repo="mmdetection", branch="main")
    source = OpenMMLabSource(out_root=out_root, repo=repo)
    metafiles = source.crawl_modelzoo()
    print("metafiles:", metafiles)
    
    base_dir = out_root / repo.repo
    saved_metafiles, metafile_failures = download_metafiles(repo, metafiles, base_dir)
    print(f"Downloaded metafiles: {len(saved_metafiles)}")

    all_cfgs, parse_failures = collect_config_paths_from_metafiles(saved_metafiles)
    print(f"config paths found: {len(all_cfgs)}")

    saved_cfgs, cfg_failures = download_configs_from_metafiles(
        repo,
        saved_metafiles,
        base_dir,
    )
    print(f"Downloaded config .py files: {len(saved_cfgs)}")

    print("\nDownloading _base_ configs ...")
    base_new, base_failures = download_base_configs(repo, base_dir)
    print(f"Downloaded {base_new} _base_ config files")

    print("\nSample saved config files:")
    for p in saved_cfgs[:10]:
        print(" -", p)

    all_failures = metafile_failures + parse_failures + cfg_failures + base_failures
    if all_failures:
        print(f"\nFailures: {len(all_failures)}")
        for f in all_failures[:20]:
            print(" -", f)
        if len(all_failures) > 20:
            print(f" ... and {len(all_failures) - 20} more")

    # mmpretrain for image classification
    print("\n" + "=" * 60)
    print("Syncing mmpretrain ...")
    print("=" * 60)
    repo2 = RepoSpec(repo="mmpretrain", branch="main")
    source2 = OpenMMLabSource(out_root=out_root, repo=repo2)
    metafiles2 = source2.crawl_modelzoo()
    print("metafiles:", metafiles2)

    base_dir2 = out_root / repo2.repo
    saved_metafiles2, metafile_failures2 = download_metafiles(repo2, metafiles2, base_dir2)
    print(f"Downloaded metafiles: {len(saved_metafiles2)}")

    all_cfgs2, parse_failures2 = collect_config_paths_from_metafiles(saved_metafiles2)
    print(f"config paths found: {len(all_cfgs2)}")

    saved_cfgs2, cfg_failures2 = download_configs_from_metafiles(
        repo2,
        saved_metafiles2,
        base_dir2,
    )
    print(f"Downloaded config .py files: {len(saved_cfgs2)}")

    print("\nDownloading _base_ configs ...")
    base_new2, base_failures2 = download_base_configs(repo2, base_dir2)
    print(f"Downloaded {base_new2} _base_ config files")

    print("\nSample saved config files:")
    for p in saved_cfgs2[:10]:
        print(" -", p)

    all_failures2 = metafile_failures2 + parse_failures2 + cfg_failures2 + base_failures2
    if all_failures2:
        print(f"\nFailures: {len(all_failures2)}")
        for f in all_failures2[:20]:
            print(" -", f)
        if len(all_failures2) > 20:
            print(f" ... and {len(all_failures2) - 20} more")