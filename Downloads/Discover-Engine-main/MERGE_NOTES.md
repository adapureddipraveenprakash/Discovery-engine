# Discovery Engine — Merge Notes

> **Status:** Stage 1 complete — scaffold, training, FAISS index, API, tests, and embedding verification all passing.

## Repos cloned (read-only reference)

| Repo | Path |
|------|------|
| [Two-Tower-Retrieval-System](https://github.com/Nikgauttam/Two-Tower-Retrieval-System) | `./external/two-tower-retrieval-system` |
| [fashion-clip](https://github.com/patrickjohncyh/fashion-clip) | `./external/fashion-clip` |

**Pre-existing workspace content:** The root also contains older zip-extracted copies (`Two-Tower-Retrieval-System-main/`, `fashion-clip-master/`). These are **not** the canonical audit sources; use `./external/` going forward.

---

## What we're keeping

### From Two-Tower-Retrieval-System

| Component | Source | Why |
|-----------|--------|-----|
| **Two-tower architecture** | `src/models/two_tower.py` | Clean separable user/item towers with MLP + L2 norm — production retrieval pattern |
| **Training loop** | `src/training/trainer.py` | InfoNCE (in-batch negatives) + BPR (explicit negatives) already implemented |
| **Negative sampling** | `src/data/loader.py` | Popularity-weighted hard negatives (`count^0.75`) |
| **FAISS indexing** | `src/indexing/build_index.py`, `build_index.py` | Pre-compute item vectors → `IndexFlatIP` for sub-ms search |
| **FastAPI serving** | `api/app.py` | `/recommend` endpoint with seen-item filtering |
| **Evaluation** | `src/evaluation/metrics.py` | Recall@k, MRR@k, NDCG@k |
| **Config-driven pipeline** | `configs/config.yaml`, `Makefile` | Reproducible train → index → serve workflow |
| **Time-based split** | `src/data/split.py` | Avoids future-leak in eval |

### From fashion-clip

| Component | Source | Why |
|-----------|--------|-----|
| **`FashionCLIP` class** | `fashion_clip/fashion_clip.py` | Domain-specific CLIP for fashion image + text embeddings |
| **`encode_images()`** | same | Batch image → 512-d vectors via `CLIPModel.get_image_features` |
| **`encode_text()`** | same | Batch text → 512-d vectors via `CLIPModel.get_text_features` |
| **L2 normalization** | `normalize=True` in `__init__` | Makes dot product == cosine similarity (same trick as two-tower) |
| **Text-to-image retrieval** | `retrieval()` method | Zero-shot query → nearest product images |
| **HuggingFace model** | `patrickjohncyh/fashion-clip` | Fine-tuned CLIP ViT-B/32 for fashion |

---

## What we're dropping

### From Two-Tower-Retrieval-System

- **MovieLens-specific data & artifacts** — `data/raw/ml-1m/`, `data/raw/ml-100k/`, pre-bundled `.dat` files
- **MF baseline** — `src/models/mf_baseline.py` (reference only, not part of Discovery Engine)
- **Ablation / comparison scripts** — `ablation.py`, `compare.py`, `benchmark.py`, interview PDF tooling
- **Docker/CI as-is** — will be re-authored for the merged project
- **Pure ID-based item tower** — in Discovery Engine, item features come from FashionCLIP, not `nn.Embedding(num_items, D)`

### From fashion-clip

- **`FCLIPDataset` + S3 catalog plumbing** — `_CATALOGS`, `_VECTORS`, S3 downloads (replace with local fashion catalog)
- **Annoy approximate index** — Discovery Engine will use FAISS (from two-tower repo) for consistency
- **`attention_map` / zero-shot classification helpers** — optional; not core to retrieval merge
- **Demo notebook** — `fashion_clip_api_demo.ipynb` (reference only)

---

## Where the two pieces connect

```
┌─────────────────────────────────────────────────────────────────┐
│                     OFFLINE (training + indexing)               │
├─────────────────────────────────────────────────────────────────┤
│  Fashion catalog (image + text per SKU)                         │
│       │                                                         │
│       ▼                                                         │
│  FashionCLIP.encode_images/text  ──► 512-d content embeddings   │
│       │                                                         │
│       ▼                                                         │
│  Item Tower: Linear(512 → hidden) → ReLU → Linear → 128-d      │
│       │                              (project FCLIP → tower D)  │
│       ▼                                                         │
│  User Tower: user_id → Embed → MLP → 128-d  (from two-tower)     │
│       │                                                         │
│       ▼                                                         │
│  InfoNCE / BPR training on (user, item) interactions            │
│       │                                                         │
│       ▼                                                         │
│  FAISS IndexFlatIP  (all item 128-d vectors, L2-normalized)     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     ONLINE (serving)                            │
├─────────────────────────────────────────────────────────────────┤
│  POST /recommend { user_id, top_k }                             │
│       │                                                         │
│       ├─► User tower → 128-d query vector                       │
│       ├─► FAISS search → top-k×5 candidates                     │
│       └─► Filter seen items → return top-k                      │
│                                                                 │
│  (Future) POST /search { query_text, top_k }                    │
│       │                                                         │
│       ├─► FashionCLIP.encode_text → 512-d                       │
│       ├─► Project to 128-d (shared projection layer)            │
│       └─► FAISS search (same index)                             │
└─────────────────────────────────────────────────────────────────┘
```

**Key integration decision (for Step 2):** FashionCLIP outputs **512-d** vectors; the two-tower system uses **128-d** (config default). A projection layer (`Linear(512, 128)`) or config change to `embedding_dim=512` will be required to align dimensions.

---

## Technical details discovered

### Two-Tower — User & Item Tower Architecture

**File:** `external/two-tower-retrieval-system/src/models/two_tower.py`

| | User Tower | Item Tower |
|---|-----------|------------|
| **Input** | `user_id` (int index) | `item_id` (int index) |
| **Embedding** | `nn.Embedding(num_users, D)` | `nn.Embedding(num_items, D)` |
| **MLP** | `Linear(D, H) → ReLU → Linear(H, D)` | same |
| **Output** | L2-normalized `D`-dim vector | L2-normalized `D`-dim vector |
| **Scoring** | dot product (cosine, since L2-normed) | |

**Config defaults** (`configs/config.yaml`):
- `embedding_dim` (D) = **128**
- `hidden_dim` (H) = **256**
- `temperature` = **0.05**
- `batch_size` = 1024, `epochs` = 25, `lr` = 0.001

(Code-level defaults in `TwoTowerModel.__init__` are D=256, H=512 — config overrides at runtime.)

### Two-Tower — Training Loop

**File:** `external/two-tower-retrieval-system/src/training/trainer.py`

| Loss | Mechanism |
|------|-----------|
| **InfoNCE** | `(user_emb @ item_emb.T) / temperature` → row-wise log-softmax; diagonal entries are positives; **in-batch negatives** (B−1 free hard negs per user) |
| **BPR** | Pairwise: `-log σ(pos_score − neg_score)` with **4 explicit negatives** per positive |

**Negative sampling** (`src/data/loader.py`):
- `"popularity"`: sample proportional to `item_count^0.75` (word2vec-style), reject items user already interacted with
- `"uniform"`: uniform random
- Config default: **`loss: bpr`**, **`negative_sampling: popularity`** (BPR beat InfoNCE on ML-1M per ablation)

### Two-Tower — FAISS Index

**File:** `external/two-tower-retrieval-system/src/indexing/build_index.py`

1. `model.eval()` → `encode_items(torch.arange(num_items))` → `(num_items, D)` float32, L2-normalized
2. `faiss.IndexFlatIP(D)` — exact inner product (= cosine similarity on unit vectors)
3. `index.add(embeddings_np)` → saved to `artifacts/faiss.index`

**Query** (`api/app.py`):
```python
_, indices = index.search(user_emb_np, top_k * 5)  # over-fetch for filtering
recommendations = [i for i in indices[0] if i not in interacted][:top_k]
```

### Two-Tower — FastAPI `/recommend` Contract

**File:** `external/two-tower-retrieval-system/api/app.py`

**Request** (`RecommendationRequest`):
```json
{ "user_id": 1, "top_k": 10 }
```

**Response**:
```json
{
  "user_id": 1,
  "recommendations": [318, 296, 593, 50, 858],
  "latency_ms": 0.43
}
```

**Additional endpoint:** `GET /info` → variant, model file, index file, num_users, num_items.

**Error:** `404` if `user_id` not in `user_map`.

### FashionCLIP — Class & Encoding

**File:** `external/fashion-clip/fashion_clip/fashion_clip.py`

**`FashionCLIP.__init__(model_name, dataset=None, normalize=True, approx=True, auth_token=None)`**
- Loads `CLIPModel` + `CLIPProcessor` from HuggingFace (`patrickjohncyh/fashion-clip` or custom repo)
- Device: cuda → mps → cpu
- If `dataset` provided: pre-encodes all images/text, optionally L2-normalizes, builds Annoy index (512-d, dot metric)

**`encode_images(images, batch_size)`**
- Input: `List[str]` (paths) or `List[PIL.Image.Image]`
- Pipeline: HuggingFace `Dataset` → `CLIPProcessor` → `DataLoader` → `model.get_image_features(**batch)`
- Output: `np.ndarray` shape `(N, 512)`

**`encode_text(texts, batch_size)`**
- Input: `List[str]`
- Pipeline: tokenize (`max_length=77`, pad/truncate) → `model.get_text_features(**batch)`
- Output: `np.ndarray` shape `(N, 512)`

**Normalization** (when `normalize=True` and dataset set):
```python
vectors = vectors / np.linalg.norm(vectors, ord=2, axis=-1, keepdims=True)
```

**Embedding dimension:** **512** (CLIP ViT-B/32; confirmed by `AnnoyIndex(512, "dot")` in source).

---

## Dimension alignment summary

| Component | Dimension |
|-----------|-----------|
| Two-tower user/item output | **128** (config) |
| Two-tower MLP hidden | **256** |
| FashionCLIP image/text output | **512** |
| FAISS index vector size | **128** (matches tower D) |

→ **Gap to resolve in Step 2:** project FashionCLIP 512-d → tower 128-d, or raise tower D to 512.

---

## Issues / blockers noted during audit

1. **API data mismatch:** `api/app.py` loads interaction history from `ml-100k/u.data` for seen-item filtering, while `config.yaml` trains on `ml-1m`. This is a bug in the reference repo — Discovery Engine should use a single consistent interaction source.

2. **No notebook equivalent:** The audit brief mentioned `ttrecsys.ipynb`; this repo uses modular `.py` files instead (`train.py`, `src/`). All required logic is present in Python modules.

3. **FashionCLIP S3 dependencies:** Pre-computed vector caches and FF catalog require S3 access. Discovery Engine should call `encode_images`/`encode_text` locally rather than relying on `_VECTORS` S3 paths.

4. **Windows environment:** The two-tower `Makefile` uses Unix commands (`curl`, `unzip`, `rm`). Step 2 scaffolding should provide cross-platform scripts or document WSL usage.

5. **Pre-existing zip copies:** Root-level `*-main`/`*-master` folders may cause confusion; recommend ignoring or removing them before Step 2.

---

## Recommended Step 2 scope (preview only — do not implement yet)

1. Scaffold `discovery-engine/` project layout (config, src, api, artifacts)
2. Replace ID-based item tower with FashionCLIP-projected features
3. Unify on FAISS `IndexFlatIP` (drop Annoy)
4. Extend API: `/recommend` (collaborative) + `/search` (text query via FashionCLIP)
5. Add fashion interaction dataset loader (replace MovieLens)

---

*Generated: Step 1 audit — awaiting review before proceeding.*
