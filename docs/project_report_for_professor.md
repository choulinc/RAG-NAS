# RAG-NAS 專案技術報告

## 1. 專案目標

本專案的核心目標是：**當使用者拿到一個新的影像資料集時，系統能先理解資料集特性，再從既有模型知識庫中找出相近設計，進一步自動產生 NAS 搜尋模板，最後在 NAS-Bench-201 上搜尋出適合的微架構。**

換句話說，這不是單純的影像分類模型，也不是單純的 RAG 系統，而是一個把下列幾件事串起來的系統：

1. **資料集理解**
2. **多模態檢索**
3. **RAG 輔助的搜尋空間生成**
4. **演化式 NAS 搜尋**
5. **Benchmark 驗證**

---

## 2. 系統總覽

整個流程可以概括成：

```text
使用者資料集
   ↓
DatasetAnalyzer
   ↓
DatasetProfile + Query
   ↓
Text Retrieval + Image Retrieval
   ↓
Fusion
   ↓
Top-K 相關模型 / 設計範式
   ↓
LLM Template Generator
   ↓
EA Search Templates
   ↓
Regularized Evolution on NAS-Bench-201
   ↓
最佳架構 + Benchmark 指標
```

如果用比較口語的方式說，就是：

- 先看使用者的資料長什麼樣子
- 把資料集翻譯成可搜尋的描述
- 去模型知識庫裡找「跟這種資料集最像、最值得參考」的模型
- 把這些模型的設計經驗整理成 NAS 搜尋模板
- 再用演化搜尋真的去找一個好架構

---

## 3. 資料進來之後，系統做了什麼

## 3.1 DatasetAnalyzer：先把資料集轉成機器可理解的描述

第一步不是直接訓練模型，而是先分析使用者提供的資料夾。這部分由 `DatasetAnalyzer` 負責，會輸出一個 `DatasetProfile`。

它主要做四件事：

1. **讀取 README 與 metadata**
   - 例如 `README.md`、`labels.txt`、`classes.json`、`data.yaml`
   - 從這些文字資訊推測 task、類別數、類別名稱

2. **看目錄結構**
   - 若有 `train/class_x/*.png`，傾向判定為分類
   - 若有 `annotations/` 且裡面有 COCO / VOC / YOLO 格式訊號，傾向判定為偵測
   - 若有 `masks/`，傾向判定為 segmentation

3. **抽樣影像統計**
   - 估計影像總數
   - 抽樣取得中位數高寬
   - 判斷通道數（灰階 / RGB / RGBA）

4. **推測 domain 與 keyword**
   - 例如從 `cifar`、`imagenet`、`medical`、`coco` 等字眼推測資料領域
   - 從類別名稱抽出關鍵字

最後會得到：

- `task`
- `domain`
- `keywords`
- `num_classes`
- `class_names`
- `image_stats`

並進一步轉成一條搜尋 query，例如：

```text
Image Classification cifar 5 classes airplane automobile bird cat deer
```

這一步的意義很重要：**它把原始資料夾，轉成後續 retrieval 與 template generation 可以使用的語意表示。**

---

## 3.2 Text Retrieval：從模型知識庫中找語意上最接近的模型

系統中有一個 UIR（Unified Intermediate Representation）格式的模型知識庫，每一筆資料包含：

- 模型名稱
- collection
- config 路徑
- paper URL
- weights URL
- architecture 資訊
- results（task / dataset / metrics）

### 這裡怎麼做 retrieval？

每一筆 UIR 會被轉成兩種文字視圖：

1. **narrative view**
   - 比較像自然語言摘要
   - 適合做 dense embedding

2. **kv view**
   - 比較像 key-value 結構化文本
   - 適合做 BM25 關鍵字匹配

然後 query 會同時走兩條路：

1. **Dense retrieval**
   - 使用 `sentence-transformers/all-MiniLM-L6-v2`
   - 將 query 和 narrative view 做 embedding
   - 用 cosine similarity 算語意相似度

2. **Keyword retrieval**
   - 對 kv view 做 tokenize
   - 使用 BM25 計分

最後做 hybrid score：

```text
final_score = 0.55 * dense_score + 0.45 * kw_score
```

此外還有一些 rule-based rerank：

- query 若明確命中 task，分數加強
- query 若明確命中 dataset，分數加強
- query 若命中 collection / model name / metric，也會有小幅 boost

### 為什麼不只用 embedding？

因為這個任務不是一般 open-domain QA，而是**結構化模型知識檢索**。  
例如：

- `ImageNet-1k`
- `top-1`
- `LinearClsHead`
- `ResNet`

這些字串本身就是非常有用的訊號。若只做 dense retrieval，容易漏掉精確欄位匹配；若只做 BM25，又會缺乏語意泛化能力。所以此處採用 **dense + keyword hybrid retrieval**。

---

## 3.3 Image Encoder：把影像資料集本身也轉成向量

除了文字檢索之外，本專案還有一條影像路徑，目的是讓系統不只依賴 README 或標籤文字，也能從資料本身的視覺分布來找相近模型。

### Encoder 架構

這部分使用的是一個 **SiameseEncoder**：

```text
Image
  ↓
ResNet-18 backbone
  ↓
Projector (Linear → ReLU → Linear)
  ↓
128-d embedding
  ↓
L2 normalization
```

也就是說，輸入影像後會先經過 ResNet-18 抽 feature，再投影成 128 維向量。

### 這個 encoder 是怎麼訓練的？

它不是用分類 cross-entropy，而是用 **contrastive learning**。

訓練資料會形成成對樣本：

- **positive pair**：來自同一個 dataset 的兩張圖
- **negative pair**：來自不同 dataset 的兩張圖

loss 為 margin-based contrastive loss：

```text
L = y * ||v1 - v2||^2 + (1-y) * max(0, margin - ||v1 - v2||)^2
```

這代表：

- 同資料集的 embedding 要靠近
- 不同資料集的 embedding 要被拉開

### 這樣學到的是什麼？

這個 encoder 學到的不是「貓狗分類器」，而是更偏向：

- CIFAR-10 類型資料的視覺分布
- STL-10 類型資料的視覺分布
- SVHN 類型資料的視覺分布

也就是說，它更像是在學 **dataset/domain-level representation**，其目的不是做最終分類，而是支援「這個新資料集看起來比較像哪些已知資料集 / 模型適用場景」。

### Reconstruction regularization

系統還另外實作了 `ImageDecoder` 與 `CombinedLoss`，把 embedding 反解回影像，加入 reconstruction loss。  
其直觀目的，是避免 encoder 只學到過於粗糙的分群訊號，希望保留更多細節。

總 loss 概念為：

```text
L_total = λ_c * L_contrastive + λ_r * L_reconstruction
```

---

## 3.4 Feature Store：把資料庫中的影像特徵先存起來

為了讓 image retrieval 不用每次都重算整個資料庫，系統設計了 `FeatureStore`：

- 每個資料庫 entry 對應一個向量
- 向量先做 L2 normalize
- 使用 FAISS `IndexFlatIP` 建索引

這樣在查詢時，只要：

1. 將使用者資料集抽樣幾張圖
2. 用 encoder 編碼成向量
3. 與資料庫中的向量做內積搜尋

就可以快速得到最相近的 doc_id。

---

## 3.5 Fusion：文字證據與影像證據怎麼合併

文字路徑與影像路徑各自會產生分數：

- `text_score`
- `image_score`

最後在 `MultiModalRetriever` 中做加權融合：

```text
final_score = α * text_score + (1 - α) * image_score
```

其中 `α` 不是完全固定，而是會依資料集特性調整：

- 若 README 充足、keyword 多，表示文字資訊豐富，`α` 變高
- 若沒有 image pathway，則直接退化成 text-only
- 若幾乎沒有文字資訊，則降低 `α`，讓 image 分支權重提高

### 為什麼要 fusion？

因為實際情況中，使用者資料集的品質很不一致：

- 有些資料集 README 很完整，但圖片內容很雜
- 有些資料集幾乎沒有文字說明，只能靠影像判斷

所以 fusion 的目的，是讓系統能同時利用：

- **顯式語意訊號**：task、dataset、class names、README
- **隱式視覺訊號**：圖片風格、解析度、物體分布、域別差異

---

## 4. 對齊模組：讓 image embedding 與 text embedding 說同一種語言

光有影像 encoder 還不夠，因為影像向量空間和文字向量空間本來是分開的。  
因此專案額外實作了 `alignment.py`，目的是做 **image-text embedding alignment**。

### 4.1 整體設計

此模組包含：

1. 冷凍的 image encoder
2. 冷凍的 SentenceTransformer text encoder
3. 兩個可訓練的 projection heads

流程如下：

```text
image → frozen image encoder → image projection head → shared space
text  → frozen text encoder  → text projection head  → shared space
```

最終兩者都投影到同一個 256 維 shared embedding space。

### 4.2 為什麼 backbone 要 frozen？

因為這個專案的重點不是重新大規模預訓練，而是用較低成本方式把既有 representation 對齊：

- image encoder 已經有 dataset/domain 表示能力
- text encoder 已經有一般語意表示能力
- 只要再學一層 projection，就能建立跨模態對齊

這樣訓練量比較小，也比較穩定。

### 4.3 損失函數：SigLIP Loss

alignment 採用的是 **SigLIP-style pairwise contrastive loss**，而不是傳統 softmax InfoNCE。

核心概念是：

- 同一筆 image-text pair 應該相似
- batch 中其他配對視為負樣本

它的優點是：

- 不依賴整體 softmax
- 小 batch 也能訓練
- 每對 pair 都獨立貢獻梯度

### 4.4 對齊資料怎麼來？

系統使用 `ImageTextPairDataset`，從 ImageFolder 形式資料夾讀入：

```text
root/class_name/image.png
```

影像配對的文字不是只有類別名稱，而是用多種 prompt template 產生，例如：

- `a photo of a cat`
- `a picture of a cat`
- `an image of a cat`

這樣做的目的，是避免文字 supervision 太單薄，讓 text encoder 看到稍微多樣化的自然語言形式。

### 4.5 這個 alignment 對系統的價值

對齊之後，系統就更有機會做到：

- 用文字查詢影像概念
- 用影像去找文字描述相近的模型
- 建立真正跨模態的相似度空間

這對於「資料集理解 → 模型檢索」這種任務特別重要，因為使用者提供的資訊可能本來就跨模態。

---

## 5. RAG 的角色：不是直接回答問題，而是產生 NAS 搜尋空間

這個專案中 RAG 最有特色的地方是：**檢索結果不是直接拿來生成一段答案，而是拿來生成 NAS 搜尋模板。**

### 5.1 檢索回來的是什麼？

檢索回來的是一些 OpenMMLab / UIR 模型條目，例如：

- 模型名稱
- backbone / head 類型
- 用在哪個 dataset
- 指標表現
- 對應 config 與 paper

### 5.2 LLM 讀完這些 context 後，要產生什麼？

LLM 不只是摘要，而是要輸出結構化 JSON template，內容包含：

- `paradigm`
- `evidence`
- `macro` 空間
- `micro.nb201` 空間

其中：

- **macro** 代表大方向設計，例如偏向 ResNet、ConvNeXt、MobileNet
- **micro.nb201** 代表 NAS-Bench-201 cell 的操作機率分布與限制條件

例如：

- 哪些 operation 機率高
- 某些 edge 偏好 `nor_conv_3x3` 還是 `skip_connect`
- `none` 最多出現幾次

所以這裡的 LLM 角色，是一個 **retrieval-conditioned search space designer**。

---

## 6. Evolutionary Search：真正去找架構

有了 template 之後，系統就進入 NAS 階段。這部分用的是 `Regularized Evolution Algorithm (REA)`，搜尋空間是 NAS-Bench-201。

### 6.1 NAS-Bench-201 的 cell 表示

NB201 一個 cell 有 6 條 edge，每條 edge 從 5 個操作中選 1 個：

- `nor_conv_3x3`
- `nor_conv_1x1`
- `skip_connect`
- `avg_pool_3x3`
- `none`

因此一個 gene 是 6 個 operation 的序列，再被轉成官方的 architecture string。

### 6.2 Template 如何影響搜尋？

LLM 產生的 template 不直接指定唯一架構，而是給一個 **帶偏好的搜尋分布**：

- `op_prior`：全域操作機率
- `edge_prior`：某條 edge 的操作機率
- `constraints`：例如 `none` 最多幾次、某操作至少出現幾次

之後 REA 會：

1. 用 template 的 prior 初始化 population
2. 用 tournament selection 選 parent
3. 隨機挑某一條 edge 做 mutation
4. mutation 時仍遵守 template 的 prior 與 constraints
5. 用 validation accuracy 當 fitness
6. 移除最老個體，持續演化

### 6.3 為什麼這樣做合理？

若完全 random search，搜尋空間太大。  
若完全 handcraft，又無法隨資料集動態改變。  
本專案做法介於兩者之間：

- 先用 retrieval 找外部知識
- 再用 LLM 把外部知識轉成帶偏好的搜尋模板
- 最後仍用 benchmark 驗證，而不是只靠 LLM 幻想

也就是：**RAG 負責縮小與引導搜尋空間，EA 負責真正優化。**

---

## 7. 評估方式與實驗流程

專案中有端到端實驗腳本，展示整個流程：

1. 建立或讀入資料集
2. 做 `DatasetAnalyzer`
3. 執行 text retrieval
4. 執行 image retrieval
5. 做 score fusion
6. 用 LLM 產生 template
7. 執行 REA
8. 用 NAS-Bench-201 查表得到 valid / test 指標

### 評估上特別注意的一點

在 `NASBench201Evaluator` 與 `REA` 中，搜尋時使用的是 **validation metric**，不是 test metric。  
也就是：

- 搜尋階段：`x-valid`
- 最後報告階段：`x-test` / `ori-test`

這樣做是為了避免 test set leakage，讓評估流程比較嚴謹。

---

## 8. 本專案的研究亮點

我認為本專案比較有研究價值的地方有五點：

1. **把 RAG 用在 NAS 搜尋空間設計，而非一般問答**
   - 檢索結果不是拿來回答問題，而是用來引導架構搜尋

2. **同時利用文字與影像的資料集表示**
   - 不只看 README，也看資料本身的視覺分布

3. **有明確的 encoder、fusion、alignment 設計**
   - 不是單一路徑，而是完整多模態檢索系統

4. **LLM 並非直接輸出最終架構，而是輸出帶限制的 template**
   - 保留 NAS 搜尋彈性，也降低 hallucination 直接決策的風險

5. **最終仍回到 benchmark 做客觀驗證**
   - 沒有停在 prompt engineering，而是落地到 NAS-Bench-201

---

## 9. 目前實作中的限制與教授可能會問的問題

這一段我刻意寫得誠實一些，因為程式裡其實也有明確標註某些地方還是 heuristic / prototype。

### 9.1 DatasetAnalyzer 仍是 heuristic-based

目前 task、domain、keyword 的推斷主要依賴：

- 檔名
- README 關鍵字
- 資料夾結構
- 常見格式判斷

所以對非標準資料集，可能會誤判。

### 9.2 Retrieval 的融合權重目前是手工設定

例如：

- `dense_weight = 0.55`
- `kw_weight = 0.45`
- rerank boosts

這些都還不是經過完整 ablation search 得出的最優值。

### 9.3 Fusion 的 alpha 調整也是 rule-based

目前是用：

- 有無 README
- 有無 keywords
- 是否有 image pathway

來決定文字分支和影像分支權重。  
這很合理，但還不是 data-driven learned fusion。

### 9.4 Contrastive encoder 學到的是 dataset/domain similarity，不是萬能視覺語意

這個 encoder 的 supervision 主要來自「同資料集 / 不同資料集」，所以更適合做資料集級檢索，而不是一般語意辨識。

### 9.5 Alignment 的負樣本仍可能有 false negatives

SigLIP loss 中，batch 內非對角元素都被當成負樣本。  
若兩個 class name 語意很接近，可能會被錯誤拉遠。

---

## 10. 若要往論文方向走，我會建議下一步做什麼

若目標是口頭報告加研究延伸，我建議後續可以做下面幾項：

1. **做 ablation study**
   - text-only
   - image-only
   - fusion
   - 不同 alpha
   - 不同 dense / BM25 權重

2. **建立標註好的 retrieval validation set**
   - 驗證 query 到底有沒有真的抓到對的模型

3. **比較不同 template source**
   - random template
   - hand-crafted template
   - retrieval-guided template

4. **比較不同 LLM**
   - local Qwen
   - API model
   - 不同 prompt 設計

5. **把 fusion 改成 learnable module**
   - 不只是手工 alpha，而是學習如何整合 text / image score

6. **擴展到不只 NAS-Bench-201**
   - 讓 macro space 與 micro space 的橋接更完整

---

## 11. 結論

本專案的核心貢獻可以總結成一句話：

> **RAG-NAS 不是直接生成神經網路架構，而是先理解新資料集，再利用多模態檢索找出可借鏡的模型設計，最後把這些知識轉成可執行的 NAS 搜尋模板。**

因此，整個系統可以視為一條完整的知識驅動 NAS pipeline：

- 前端是 **dataset understanding**
- 中段是 **text/image encoder + fusion + alignment**
- 後段是 **retrieval-guided template generation + evolutionary NAS**

這樣的設計比單純 random search 更有方向性，比單純 LLM 直接生架構更可控，也比一般 RAG 問答更具有研究上的結構性與可驗證性。

---

## 12. 報告時可直接使用的簡短口語版總結

如果要用 1 分鐘向教授快速說明，我會這樣講：

> 老師，我這個系統的想法是，當一個新資料集進來時，我不是立刻亂搜架構，而是先分析這個資料集的 task、domain、類別數與影像特徵。接著我會讓資料同時走文字檢索和影像檢索兩條路，前者找語意相近的模型，後者找視覺分布相近的模型，再把兩者融合。之後我不是直接讓 LLM 生最終架構，而是讓它根據檢索回來的模型證據，產生 NAS 的搜尋模板，最後再用 evolutionary search 在 NAS-Bench-201 上找出最佳架構。也就是說，這是一個把 dataset understanding、multimodal retrieval、RAG 與 NAS 串成一起的系統。 
