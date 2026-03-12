"""
End-to-End Integration Test for RAG-NAS Retrieval Pipeline

完整展示從「使用者輸入資料集」到「產生 EA search space templates」的全流程。
LLM 使用 Mock 版本，不需要真的呼叫 OpenAI API。

Run:
    python test/test_e2e_pipeline.py
"""
import os
import sys
import json
import shutil
import tempfile
import random
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

# ============================================================================
# Helpers: 建構 mock 資料
# ============================================================================

def create_mock_dataset(root: str) -> str:
    """
    建構一個假的 CIFAR-like classification dataset 目錄結構。
    模擬使用者拿到一個新資料集要分析的場景。
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # README
    (root / "README.md").write_text(
        "# CIFAR-100 Subset\n"
        "A subset of CIFAR-100 for image classification.\n"
        "Contains 5 classes with 32x32 RGB images.\n"
    )

    # labels.txt
    class_names = ["airplane", "automobile", "bird", "cat", "deer"]
    (root / "labels.txt").write_text("\n".join(class_names) + "\n")

    # train/class/images
    try:
        from PIL import Image
        for cls in class_names:
            cls_dir = root / "train" / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(10):
                img = Image.new("RGB", (32, 32), color=(
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ))
                img.save(str(cls_dir / f"{i:03d}.png"))
    except ImportError:
        # Fallback: create empty directories
        for cls in class_names:
            (root / "train" / cls).mkdir(parents=True, exist_ok=True)

    return str(root)


def create_mock_uir(uir_path: str) -> str:
    """
    建構假的 UIR JSONL，模擬 RAG DB 中的 OpenMMLab 模型資訊。
    """
    entries = [
        {
            "doc_id": "resnet50_cifar100",
            "name": "ResNet-50 on CIFAR-100",
            "collection": "ResNet",
            "config_repo_path": "configs/resnet/resnet50_cifar100.py",
            "weights_url": "https://download.openmmlab.com/resnet50_cifar100.pth",
            "paper_url": "https://arxiv.org/abs/1512.03385",
            "arch": {"backbone": "ResNet", "head": "LinearClsHead"},
            "results": [
                {"task": "Image Classification", "dataset": "CIFAR-100",
                 "metrics": {"top-1": 79.34, "top-5": 94.67}}
            ],
            "metadata": {"Parameters": 23520842, "FLOPs": 4112000000},
        },
        {
            "doc_id": "vit_base_imagenet",
            "name": "ViT-Base on ImageNet",
            "collection": "ViT",
            "config_repo_path": "configs/vit/vit-base_imagenet.py",
            "weights_url": "https://download.openmmlab.com/vit_base_imagenet.pth",
            "paper_url": "https://arxiv.org/abs/2010.11929",
            "arch": {"backbone": "VisionTransformer", "head": "LinearClsHead"},
            "results": [
                {"task": "Image Classification", "dataset": "ImageNet-1k",
                 "metrics": {"top-1": 81.07, "top-5": 95.32}}
            ],
            "metadata": {"Parameters": 86567656, "FLOPs": 17581000000},
        },
        {
            "doc_id": "convnext_tiny_cifar10",
            "name": "ConvNeXt-Tiny on CIFAR-10",
            "collection": "ConvNeXt",
            "config_repo_path": "configs/convnext/convnext-tiny_cifar10.py",
            "weights_url": "https://download.openmmlab.com/convnext_tiny_cifar10.pth",
            "paper_url": "https://arxiv.org/abs/2201.03545",
            "arch": {"backbone": "ConvNeXt", "head": "LinearClsHead"},
            "results": [
                {"task": "Image Classification", "dataset": "CIFAR-10",
                 "metrics": {"top-1": 96.21}}
            ],
            "metadata": {"Parameters": 28589128, "FLOPs": 4500000000},
        },
        {
            "doc_id": "mobilenet_v2_cifar100",
            "name": "MobileNet-V2 on CIFAR-100",
            "collection": "MobileNet",
            "config_repo_path": "configs/mobilenet/mobilenet_v2_cifar100.py",
            "weights_url": "https://download.openmmlab.com/mobilenet_v2_cifar100.pth",
            "paper_url": "https://arxiv.org/abs/1801.04381",
            "arch": {"backbone": "MobileNetV2", "head": "LinearClsHead"},
            "results": [
                {"task": "Image Classification", "dataset": "CIFAR-100",
                 "metrics": {"top-1": 73.21, "top-5": 91.45}}
            ],
            "metadata": {"Parameters": 3504872, "FLOPs": 319000000},
        },
        {
            "doc_id": "swin_tiny_imagenet",
            "name": "Swin-Tiny on ImageNet",
            "collection": "Swin",
            "config_repo_path": "configs/swin/swin-tiny_imagenet.py",
            "weights_url": "https://download.openmmlab.com/swin_tiny_imagenet.pth",
            "paper_url": "https://arxiv.org/abs/2103.14030",
            "arch": {"backbone": "SwinTransformer", "head": "LinearClsHead"},
            "results": [
                {"task": "Image Classification", "dataset": "ImageNet-1k",
                 "metrics": {"top-1": 81.18, "top-5": 95.61}}
            ],
            "metadata": {"Parameters": 28288354, "FLOPs": 4510000000},
        },
    ]

    Path(uir_path).parent.mkdir(parents=True, exist_ok=True)
    with open(uir_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return uir_path


def create_mock_feature_store(store_dir: str, uir_path: str) -> str:
    """
    建構假的 FeatureStore，為每個 UIR entry 產生 random embedding。
    正式環境中這些 embedding 會由 contrastive encoder 計算。
    """
    from src.retrieval.feature_store import FeatureStore

    with open(uir_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    store = FeatureStore(dim=128)
    for entry in entries:
        vec = np.random.randn(128).astype(np.float32)
        store.add(entry["doc_id"], vec, {
            "name": entry["name"],
            "task": entry["results"][0]["task"],
            "dataset": entry["results"][0]["dataset"],
        })

    store.save(store_dir)
    return store_dir


def mock_generate_templates(query: str, hits: list, profile) -> list:
    """
    Mock LLM template generator。
    根據 retrieval 結果產生「假的」EA search space templates。
    """
    # 從 hits 中提取 backbone 資訊
    backbones = set()
    for h in hits[:3]:
        ctx = h.get("context_text", "")
        for bb in ["ResNet", "ConvNeXt", "MobileNet", "ViT", "Swin"]:
            if bb.lower() in ctx.lower() or bb in str(h):
                backbones.add(bb)

    if not backbones:
        backbones = {"ResNet"}

    templates = [
        {
            "paradigm": "Heavy Convolutional (from RAG retrieval)",
            "task": profile.task if profile else "Image Classification",
            "dataset": profile.domain if profile else "unknown",
            "evidence": [
                {
                    "doc_id": h.get("doc_id", ""),
                    "config_repo_path": h.get("config_repo_path", ""),
                    "paper_url": h.get("paper_url", ""),
                    "why": f"Retrieved with score {h.get('score', 0):.4f}",
                }
                for h in hits[:2]
            ],
            "macro": {
                "backbone": list(backbones),
                "neck": ["GlobalAveragePooling"],
                "head": ["LinearClsHead"],
                "hparams": {"lr": ["log", 0.0001, 0.05]},
            },
            "micro": {
                "nb201": {
                    "allowed_ops": ["nor_conv_3x3", "nor_conv_1x1", "skip_connect", "avg_pool_3x3", "none"],
                    "op_prior": {
                        "nor_conv_3x3": 0.45,
                        "nor_conv_1x1": 0.15,
                        "skip_connect": 0.30,
                        "avg_pool_3x3": 0.07,
                        "none": 0.03,
                    },
                    "constraints": [
                        {"type": "max_count", "op": "none", "value": 1, "reason": "Avoid disconnection"},
                        {"type": "min_count", "op": "nor_conv_3x3", "value": 1, "reason": "Feature extraction"},
                    ],
                }
            },
        },
        {
            "paradigm": "Lightweight Mobile (from RAG retrieval)",
            "task": profile.task if profile else "Image Classification",
            "dataset": profile.domain if profile else "unknown",
            "evidence": [
                {
                    "doc_id": h.get("doc_id", ""),
                    "config_repo_path": h.get("config_repo_path", ""),
                    "paper_url": h.get("paper_url", ""),
                    "why": f"Retrieved with score {h.get('score', 0):.4f}",
                }
                for h in hits[1:3]
            ],
            "macro": {
                "backbone": ["MobileNetV2", "EfficientNet"],
                "neck": ["GlobalAveragePooling"],
                "head": ["LinearClsHead"],
                "hparams": {"lr": ["log", 0.0005, 0.01]},
            },
            "micro": {
                "nb201": {
                    "allowed_ops": ["nor_conv_3x3", "nor_conv_1x1", "skip_connect", "avg_pool_3x3", "none"],
                    "op_prior": {
                        "nor_conv_3x3": 0.10,
                        "skip_connect": 0.40,
                        "nor_conv_1x1": 0.40,
                        "avg_pool_3x3": 0.05,
                        "none": 0.05,
                    },
                    "constraints": [
                        {"type": "max_count", "op": "nor_conv_3x3", "value": 2, "reason": "Keep lightweight"},
                    ],
                }
            },
        },
    ]
    return templates


# ============================================================================
# Main E2E Test
# ============================================================================

def run_e2e_test():
    """End-to-end test of the full RAG-NAS retrieval pipeline."""

    tmpdir = tempfile.mkdtemp(prefix="ragnas_e2e_")

    try:
        print("=" * 70)
        print("  RAG-NAS End-to-End Pipeline Test (Mock LLM)")
        print("=" * 70)

        # ─────────────────────────────────────────────────────────
        # STEP 1: 建構 Mock 資料
        # ─────────────────────────────────────────────────────────
        print("\n" + "─" * 50)
        print("STEP 1: 建構測試資料")
        print("─" * 50)

        dataset_dir = create_mock_dataset(os.path.join(tmpdir, "my_dataset"))
        print(f"  Mock dataset: {dataset_dir}")

        uir_path = create_mock_uir(os.path.join(tmpdir, "uir", "mock_uir.jsonl"))
        print(f"  Mock UIR:     {uir_path}")

        store_dir = create_mock_feature_store(
            os.path.join(tmpdir, "feature_store"), uir_path
        )
        print(f"  Mock store:   {store_dir}")

        # ─────────────────────────────────────────────────────────
        # STEP 2: DatasetAnalyzer — 分析資料集
        # ─────────────────────────────────────────────────────────
        print("\n" + "─" * 50)
        print("STEP 2: DatasetAnalyzer — 分析使用者資料集")
        print("─" * 50)

        from src.retrieval.dataset_analyzer import DatasetAnalyzer

        analyzer = DatasetAnalyzer()
        profile = analyzer.analyze(dataset_dir)

        print(f"  Task:        {profile.task}")
        print(f"  Domain:      {profile.domain}")
        print(f"  Keywords:    {profile.keywords}")
        print(f"  Num classes: {profile.num_classes}")
        print(f"  Class names: {profile.class_names[:5]}")
        print(f"  Image size:  {profile.image_stats.median_height}x{profile.image_stats.median_width}")
        print(f"  Channels:    {profile.image_stats.channels}")
        print(f"  Total imgs:  {profile.image_stats.total_count}")
        print(f"  Query:       \"{profile.to_query()}\"")

        assert profile.task == "Image Classification", f"Expected classification, got {profile.task}"
        assert profile.num_classes == 5

        # ─────────────────────────────────────────────────────────
        # STEP 3: Text Retrieval — 用 query 搜尋 RAG DB
        # ─────────────────────────────────────────────────────────
        print("\n" + "─" * 50)
        print("STEP 3: Text Retrieval — 在 RAG DB 中搜尋")
        print("─" * 50)

        from src.retrieval.retrieve import retrieve as text_retrieve

        query = profile.to_query()
        text_hits = text_retrieve(
            uir_path=uir_path,
            query=query,
            topk=5,
        )

        print(f"\n  Query: \"{query}\"")
        print(f"  Results: {len(text_hits)} hits\n")
        for i, h in enumerate(text_hits, 1):
            d = h.get("debug", {})
            print(f"  [{i}] {h['name']}")
            print(f"      score={h['score']:.4f} (dense={d.get('dense_score',0):.4f}, kw={d.get('kw_score',0):.4f})")
            print(f"      task={h.get('task')} | dataset={h.get('dataset')}")
            print(f"      config={h.get('config_repo_path')}")
            print()

        assert len(text_hits) > 0, "No retrieval results"

        # ─────────────────────────────────────────────────────────
        # STEP 4: Image Retrieval — Contrastive Encoder 比對
        # ─────────────────────────────────────────────────────────
        print("─" * 50)
        print("STEP 4: Image Retrieval — Contrastive Encoder 比對")
        print("─" * 50)

        from src.retrieval.feature_store import FeatureStore
        from src.retrieval.contrastive_encoder import SiameseEncoder, ImageRetriever

        # 嘗試用真正訓練好的 encoder，否則用 random
        ckpt_path = "checkpoints/contrastive_encoder_best.pt"
        if os.path.exists(ckpt_path):
            print(f"  Loading trained encoder from {ckpt_path}")
            from src.retrieval.contrastive_encoder import ContrastiveTrainer
            encoder = ContrastiveTrainer.load_checkpoint(ckpt_path, device="cpu")
        else:
            print("  No trained checkpoint found, using untrained encoder")
            encoder = SiameseEncoder(embed_dim=128, pretrained=False)
            encoder.eval()

        image_retriever = ImageRetriever(encoder, device="cpu")

        # 收集 sample images
        sample_images = []
        for dirpath, _, fnames in os.walk(dataset_dir):
            for fn in fnames:
                if fn.endswith((".png", ".jpg")):
                    sample_images.append(os.path.join(dirpath, fn))
                    if len(sample_images) >= 10:
                        break
            if len(sample_images) >= 10:
                break

        if sample_images:
            store = FeatureStore.load(store_dir)
            image_hits = image_retriever.retrieve(
                query_image_paths=sample_images,
                store_vectors=store.get_all_vectors(),
                store_doc_ids=store.get_all_doc_ids(),
                topk=3,
            )

            print(f"\n  Sampled {len(sample_images)} images for comparison")
            print(f"  Image retrieval results:\n")
            for i, ih in enumerate(image_hits, 1):
                meta = store.metadata[store.doc_ids.index(ih["doc_id"])]
                print(f"  [{i}] {meta.get('name', ih['doc_id'])}")
                print(f"      image_score={ih['image_score']:.4f}")
                print(f"      dataset={meta.get('dataset', 'N/A')}")
                print()
        else:
            print("  No images found, skipping image retrieval")
            image_hits = []

        # ─────────────────────────────────────────────────────────
        # STEP 5: Score Fusion — 融合 text + image
        # ─────────────────────────────────────────────────────────
        print("─" * 50)
        print("STEP 5: Score Fusion — 融合 text + image scores")
        print("─" * 50)

        alpha = 0.7  # text weight
        print(f"\n  alpha = {alpha} (text={alpha}, image={1-alpha})")

        # Normalize text scores
        text_by_id = {}
        if text_hits:
            max_ts = max(h["score"] for h in text_hits)
            for h in text_hits:
                text_by_id[h["doc_id"]] = h["score"] / max_ts if max_ts > 0 else 0

        # Normalize image scores
        img_by_id = {}
        if image_hits:
            max_is = max(abs(ih["image_score"]) for ih in image_hits)
            for ih in image_hits:
                img_by_id[ih["doc_id"]] = ih["image_score"] / max_is if max_is > 0 else 0

        # Fuse
        all_ids = set(text_by_id.keys()) | set(img_by_id.keys())
        fused = []
        for did in all_ids:
            ts = text_by_id.get(did, 0)
            ims = img_by_id.get(did, 0)
            final = alpha * ts + (1 - alpha) * ims
            fused.append({"doc_id": did, "final": final, "text": ts, "image": ims})

        fused.sort(key=lambda x: x["final"], reverse=True)

        print(f"\n  {'Rank':<5} {'doc_id':<30} {'Text':>7} {'Image':>7} {'Final':>7}")
        print("  " + "-" * 60)
        for i, f in enumerate(fused, 1):
            print(f"  {i:<5} {f['doc_id']:<30} {f['text']:>7.4f} {f['image']:>7.4f} {f['final']:>7.4f}")

        # ─────────────────────────────────────────────────────────
        # STEP 6: LLM Template Generation (auto: Qwen / Mock)
        # ─────────────────────────────────────────────────────────
        print("\n" + "─" * 50)
        print("STEP 6: LLM Template Generation")
        print("─" * 50)

        import torch
        use_local_llm = torch.cuda.is_available()

        if use_local_llm:
            print("\n  CUDA detected → using local Qwen model")
            from src.retrieval.llm_template_generator import get_template_generator
            generator = get_template_generator(use_local=True)
            result = generator.generate_templates(query, text_hits, profile=profile)
            templates = result if isinstance(result, list) else [result]
        else:
            print("\n  No CUDA → using Mock templates")
            templates = mock_generate_templates(query, text_hits, profile)

        print(f"\n  Generated {len(templates)} templates:\n")
        for i, tmpl in enumerate(templates, 1):
            print(f"  Template {i}: \"{tmpl.get('paradigm', 'unnamed')}\"")
            print(f"    Task:      {tmpl.get('task', 'N/A')}")
            print(f"    Backbones: {tmpl.get('macro', {}).get('backbone', [])}")
            print(f"    Heads:     {tmpl.get('macro', {}).get('head', [])}")
            nb201 = tmpl.get("micro", {}).get("nb201", {})
            if nb201.get("op_prior"):
                print(f"    NB201 op_prior:")
                for op, p in nb201["op_prior"].items():
                    bar = "█" * int(p * 30)
                    print(f"      {op:<16} {p:.2f} {bar}")
            if nb201.get("constraints"):
                print(f"    Constraints:")
                for c in nb201["constraints"]:
                    print(f"      {c['type']}: {c['op']} <= {c.get('value')} ({c.get('reason', '')})")
            if tmpl.get("evidence"):
                print(f"    Evidence:")
                for ev in tmpl["evidence"][:2]:
                    paper = ev.get('paper_url', '')
                    paper_str = f" | paper: {paper}" if paper else ""
                    print(f"      - {ev.get('doc_id')}: {ev.get('why')}{paper_str}")
            print()

        # ─────────────────────────────────────────────────────────
        # STEP 7: EA Sampling (Quick Demo)
        # ─────────────────────────────────────────────────────────
        print("─" * 50)
        print("STEP 7: EA Sampling Preview — 從 template 取樣 NB201 架構")
        print("─" * 50)

        from src.nas.evolutionary_search import sample_gene_from_template, gene_to_string

        # 存起來供 Step 8 查表
        sampled_archs = []  # list of (paradigm, arch_str)

        print()
        for tmpl in templates:
            print(f"  Paradigm: \"{tmpl['paradigm']}\"")
            for j in range(3):
                gene = sample_gene_from_template(tmpl)
                arch_str = gene_to_string(gene)
                sampled_archs.append((tmpl["paradigm"], arch_str))
                print(f"    Sample {j+1}: {arch_str}")
            print()

        # ─────────────────────────────────────────────────────────
        # STEP 8: NAS-Bench-201 Evaluation — 查表取得真實準確率
        # ─────────────────────────────────────────────────────────
        print("─" * 50)
        print("STEP 8: NAS-Bench-201 Evaluation — 查表取得真實準確率")
        print("─" * 50)

        NB201_API_PATH = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..",
            "data", "NAS-Bench", "NAS-Bench-201-v1_1-096897.pth"
        )
        # Also try from project root
        if not os.path.exists(NB201_API_PATH):
            NB201_API_PATH = "data/NAS-Bench/NAS-Bench-201-v1_1-096897.pth"
        NB201_DATASETS = ["cifar10-valid", "cifar100", "ImageNet16-120"]

        use_real_api = os.path.exists(NB201_API_PATH)

        if use_real_api:
            try:
                print(f"\n  Loading NAS-Bench-201 API from {NB201_API_PATH} ...")
                from src.nas.nasbench201_evaluator import NASBench201Evaluator
                evaluator = NASBench201Evaluator(NB201_API_PATH)
            except Exception as e:
                print(f"\n  Failed to load NB201 API: {e}")
                print("  Falling back to Mock Evaluator")
                evaluator = None
                use_real_api = False
        else:
            print(f"\n  NB201 API not found at {NB201_API_PATH}")
            print("  Using Mock Evaluator (random scores)")
            evaluator = None

        # 表頭
        print()
        header = f"  {'#':<3} {'Paradigm':<45} {'Arch String':<75}"
        for ds in NB201_DATASETS:
            header += f" {ds:>16}"
        print(header)
        print("  " + "─" * (3 + 45 + 75 + 16 * len(NB201_DATASETS)))

        best_arch = None
        best_acc = -1.0

        for idx, (paradigm, arch_str) in enumerate(sampled_archs, 1):
            short_paradigm = paradigm[:43] + ".." if len(paradigm) > 45 else paradigm

            row = f"  {idx:<3} {short_paradigm:<45} {arch_str:<75}"

            for ds in NB201_DATASETS:
                if evaluator:
                    acc = evaluator.evaluate(arch_str, dataset=ds)
                else:
                    # Mock: 隨機分數，但加一些 pattern-based bias
                    base = random.uniform(60, 95)
                    if "nor_conv_3x3" in arch_str:
                        base += arch_str.count("nor_conv_3x3") * 2
                    if "none" in arch_str:
                        base -= arch_str.count("none") * 10
                    acc = min(97.0, max(10.0, base))

                row += f" {acc:>15.2f}%"

                # Track best on cifar100
                if ds == "cifar100" and acc > best_acc:
                    best_acc = acc
                    best_arch = arch_str

            print(row)

        print()
        print(f"  Best architecture (by cifar100): {best_arch}")
        print(f"  Best accuracy: {best_acc:.2f}%")
        if not use_real_api:
            print("  (Note: scores are mock-generated, use real NB201 API for actual results)")

        # ─────────────────────────────────────────────────────────
        # Summary
        # ─────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("  Pipeline Summary")
        print("=" * 70)
        print(f"""
  1. Dataset Analyzed:
     - Path:    {dataset_dir}
     - Task:    {profile.task}
     - Domain:  {profile.domain}
     - Classes: {profile.num_classes}

  2. Text Retrieval:
     - Query:   "{query}"
     - Top hit: {text_hits[0]['name']} (score={text_hits[0]['score']:.4f})

  3. Image Retrieval:
     - Samples: {len(sample_images)} images encoded
     - Top hit: {image_hits[0]['doc_id'] if image_hits else 'N/A'}

  4. Templates Generated: {len(templates)}
     - LLM: {'Local Qwen' if use_local_llm else 'Mock'}
     - {templates[0].get('paradigm', 'unnamed') if templates else 'N/A'}
     - {templates[1].get('paradigm', 'unnamed') if len(templates) > 1 else 'N/A'}

  5. NB201 Evaluation:
     - API: {'Real' if use_real_api else 'Mock'}
     - Architectures evaluated: {len(sampled_archs)}
     - Best (cifar100): {best_acc:.2f}% — {best_arch}
""")
        print("  All steps completed successfully!")
        print("=" * 70)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    run_e2e_test()
