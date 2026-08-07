# Discovery Engine — Stage 1

Fashion retrieval system merging **Two-Tower** collaborative filtering with **FashionCLIP** multimodal item features.

## Architecture

```
                         OFFLINE                         ONLINE
+---------------------------+              +---------------------------+
| H&M catalog (image+text)  |              | POST /recommend           |
|          |                |              |   session -> user tower   |
|          v                |              |   -> FAISS -> filter seen |
| FashionCLIP encode        |              +---------------------------+
|   image (512) + text (512)|              | POST /search              |
|          |                |              |   text/image -> FCLIP     |
|          v                |              |   -> project -> FAISS     |
| fuse 0.6*img + 0.4*txt    |              +---------------------------+
|          |                |              | POST /complete-the-look   |
|          v                |              |   seed -> FAISS + category|
| Linear(512 -> 128)        |              |   complement + diversity  |
|   (trainable projection)  |              +---------------------------+
|          |                |
| User tower (history pool  |
|   + MLP -> 128-d)         |
|          |                |
| InfoNCE training          |
|          |                |
| FAISS IndexFlatIP (128-d) |
+---------------------------+
```

## Quick start

### Prerequisites

- Python 3.10+
- ~2 GB disk for FashionCLIP model download (first run)

### Setup (Unix / Git Bash / WSL)

```bash
make install
make data      # synthetic H&M-shaped sample data
make train     # FashionCLIP encode + InfoNCE user tower
make index     # build FAISS index
make api       # serve on http://localhost:8000
```

### Setup (Windows PowerShell)

```powershell
.\scripts\install.ps1
.\scripts\data.ps1
.\scripts\train.ps1
.\scripts\index.ps1
.\scripts\api.ps1
```

Or run the full offline pipeline in one step:

```powershell
.\scripts\pipeline.ps1
```

Individual scripts mirror `make` targets (`install`, `data`, `train`, `index`, `api`, `test`, `verify`).

Manual equivalent:

```powershell
python -m pip install -r requirements.txt
python training/prepare_dataset.py
python training/train_user_tower.py --epochs 5
python scripts/build_index.py
python -m uvicorn serving.api:app --reload --host 0.0.0.0 --port 8000
```

Or use `make` if [GNU Make](https://gnuwin32.sourceforge.net/packages/make.htm) is installed.

### Verify embeddings (Step 4 checkpoint)

```bash
make verify
# or: python scripts/verify_embeddings.py
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/recommend` | `{"user_id": "U00001", "top_k": 10}` |
| POST | `/search` | `{"query_text": "blue denim jacket", "top_k": 10}` |
| POST | `/complete-the-look` | `{"seed_item_idx": 42, "top_k": 10}` |

**Recommend response shape:**

```json
{"user_id": "U00001", "recommendations": [12, 45, 78], "latency_ms": 1.23}
```

## Swapping in real H&M data

Place the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) files under `data/`:

```
data/
  articles.csv
  customers.csv
  transactions_train.csv
  images/{article_id}.jpg
```

Then run `make data` (or `python training/prepare_dataset.py`) to rebuild artifacts. No code changes required.

## Project layout

```
models/          item_tower, user_tower, fusion (FashionCLIP wrapper)
index/           FAISS build + FaissStore
agents/          candidate, search, complete-the-look (plain Python)
serving/         FastAPI api.py
training/        prepare_dataset, train_user_tower
eval/            metrics, ablation
tests/           unit tests
scripts/         verify_embeddings, build_index, Windows *.ps1 pipeline scripts
external/        read-only reference repos (do not modify)
```

## Tests

```bash
make test
```

## Evaluation

```bash
make ablation   # writes eval/ablation_results.md
```

Metrics: Recall@10/20, NDCG@10, cold-start Recall@10.

## Key design decisions

- **128-d** shared space with trainable `Linear(512→128)` on FashionCLIP fused items
- **No user_id embedding** — session history pooling + MLP
- **No ID-based item tower** — FashionCLIP fusion only
- **FAISS IndexFlatIP** for exact cosine search
- **InfoNCE** in-batch negative training
- **Stage 2 guardrails** — see [GUARDRAILS_REPORT.md](GUARDRAILS_REPORT.md); audit via `python scripts/audit_workstream_b.py`

See [MERGE_NOTES.md](MERGE_NOTES.md) for the Step 1 audit and source mapping.
