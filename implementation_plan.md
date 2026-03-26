# Production Contrastive Encoder Training

重寫 [scripts/train_contrastive_encoder.py](file:///home/choulin/RAG-NAS/scripts/train_contrastive_encoder.py)，移除 100 張限制，加入多個資料集，並記錄完整訓練 log。

## Proposed Changes

### [MODIFY] [train_contrastive_encoder.py](file:///home/choulin/RAG-NAS/scripts/train_contrastive_encoder.py)

#### 1. 移除 100 張限制

[download_and_organize_cifar()](file:///home/choulin/RAG-NAS/scripts/train_contrastive_encoder.py#53-122) 中的 `max_per_class = 100` 改為 **無上限**（CIFAR-10 每類 5000 張，CIFAR-100 每類 500 張，全部使用）。

#### 2. 新增多個資料集

| Dataset | 來源 | 類別數 | 圖片數 | 用途 |
|---------|------|--------|--------|------|
| CIFAR-10 | torchvision | 10 | 50,000 | NB201 目標 dataset |
| CIFAR-100 | torchvision | 100 | 50,000 | NB201 目標 dataset |
| STL-10 | torchvision | 10 | 5,000 (train) | 較高解析度 (96×96) |
| SVHN | torchvision | 10 | 73,257 | 街景門牌數字 |
| FashionMNIST | torchvision | 10 | 60,000 | 服飾灰階 |
| Flowers102 | torchvision | 102 | ~2,040 | 花卉細粒度 |
| Food101 | torchvision | 101 | 75,750 (train) | 食物識別 |
| ImageFolder | 自訂路徑 | 動態 | 動態 | 使用者可擴充 |

CLI 新增 `--dataset all` 一鍵全部訓練，以及 `--extra_dirs` 允許指定更多 ImageFolder 路徑。

#### 3. 正式訓練配置（Production defaults）

| 參數 | 舊值 | 新值 | 理由 |
|------|------|------|------|
| epochs | 20 | 100 | 充分收斂 |
| pairs_per_epoch | 4,000 | 20,000 | 更多 pair 覆蓋 |
| batch_size | 64 | 128 | GPU 記憶體足夠 |
| lr | 1e-4 | 3e-4 | 配合 warmup |
| scheduler | CosineAnnealing | Warmup 5 epochs + CosineAnnealing | 穩定起步 |
| num_workers | 2 | 4 | 加速 data loading |
| image_size | 32 | 32 | 保持與 NB201 一致 |

#### 4. Training Log

每次訓練結束後寫入 `logs/contrastive_training_YYYYMMDD_HHMMSS.json`，包含：

```json
{
  "timestamp": "2026-03-19T16:12:00+08:00",
  "config": { "epochs": 100, "lr": 3e-4, ... },
  "datasets": {
    "cifar10": { "classes": 10, "images": 50000 },
    "cifar100": { "classes": 100, "images": 50000 },
    ...
  },
  "total_images": 316047,
  "total_classes": 343,
  "history": [0.523, 0.412, ...],
  "best_epoch": 87,
  "best_loss": 0.089,
  "checkpoint_best": "checkpoints/contrastive_encoder_best.pt",
  "checkpoint_final": "checkpoints/contrastive_encoder_final.pt"
}
```

## Verification Plan

```bash
# 乾跑 1 epoch 確認所有 dataset 可載入
python scripts/train_contrastive_encoder.py --dataset all --epochs 1 --pairs_per_epoch 100
```

```bash
# 確認 log 檔案產出
ls -la logs/contrastive_training_*.json
cat logs/contrastive_training_*.json | python -m json.tool | head -30
```

還有多加一個 --csv_label 參數，自動把Kaggle格式轉成 ImageFolder