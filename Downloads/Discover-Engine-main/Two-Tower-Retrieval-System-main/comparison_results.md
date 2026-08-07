# InfoNCE vs BPR — Side-by-Side Comparison

_Trained on MovieLens-1M (5378 users, 3468 items) with identical configs except for loss function._

| Loss | Recall@10 | MRR@10 | NDCG@10 | Train (s) | Saved as |
|---|---|---|---|---|---|
| infonce | 0.3363 | 0.1265 | 0.2708 | 423.4 | `model_infonce.pt` |
| **bpr** | **0.5667** | 0.3028 | 0.7988 | 437.5 | `model_bpr.pt` |

## Winner

**BPR** wins with Recall@10 = 0.5667 (+68.5% vs infonce).

## Why one beats the other (hypothesis)

ML-1M is heavily popularity-skewed. With InfoNCE, every other user's positive item in the batch becomes a "negative" — so popular movies appear as negatives for almost every user, including users who would actually like them. The model is biased to demote popular items.

BPR with popularity-weighted random sampling has the same long-tail awareness but doesn't force every popular item to be a negative everywhere. It's a softer, more honest gradient signal on a skewed distribution.

## How to serve each variant

```bash
LOSS=bpr     uvicorn api.app:app --reload --port 8000
LOSS=infonce uvicorn api.app:app --reload --port 8001

curl http://localhost:8000/info     # → {"variant": "bpr", ...}
curl http://localhost:8001/info     # → {"variant": "infonce", ...}
```
