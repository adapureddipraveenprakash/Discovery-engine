# Ablation Results

| Configuration | Recall@10 | MRR@10 | NDCG@10 | Train (s) |
|---|---|---|---|---|
| Baseline (InfoNCE + MLP + popularity negs, D=128) | 0.2722 | 0.1097 | 0.2320 | 78.1 |
| BPR loss (instead of InfoNCE) | 0.5089 | 0.2325 | 0.5713 | 79.8 |
| Uniform negatives (instead of popularity) | 0.2900 | 0.1120 | 0.2274 | 30.6 |
| No MLP (plain embeddings) | 0.0934 | 0.0243 | 0.0451 | 74.7 |
| Embedding dim = 64 | 0.2714 | 0.1024 | 0.1975 | 74.5 |
| Embedding dim = 256 | 0.2963 | 0.1061 | 0.2217 | 81.0 |
