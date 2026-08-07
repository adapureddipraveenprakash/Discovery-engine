# Design Decisions & Lessons Learned

> A log of the architectural choices, dead ends, and bugs I hit while building this project.
> Written so a reader can understand *why* the code looks the way it does, not just *what* it does.

---

## Why this project?

<!-- TODO Nikhil: replace this with 2-3 sentences in your own words.
     e.g. "I picked this project because I wanted to understand how
     YouTube and Spotify recommend content at scale. Two-tower retrieval
     is the actual system Google uses, and I wanted to learn it
     end-to-end — not just train a model, but also serve it." -->

---

## Decision 1 — Two towers vs. one

**What I considered:** A single neural network that takes (user_id, item_id) and outputs a score (DeepFM, NeuMF, etc.).

**What I chose:** Two independent towers.

**Why:** At inference time, I can pre-compute every item embedding **once**, dump them into FAISS, and only run the user tower at query time. With a single combined model I'd have to score every candidate item online — O(N) forward passes per request. With two towers it's O(1) for the user tower + O(log N) for the FAISS search.

This is the same reason Google's YouTube paper (Yi et al. 2019) uses two towers in production.

---

## Decision 2 — BPR loss vs InfoNCE (the surprising result)

**Hypothesis going in:** InfoNCE with in-batch negatives should win — it's what Google's YouTube paper uses, and you get 1023 hard negatives per positive for free instead of 4 random ones.

**What actually happened:** I trained both end-to-end on the same data, same config, same seed:

| Loss | Recall@10 | MRR@10 | NDCG@10 |
|---|---|---|---|
| InfoNCE (in-batch negs) | 0.3363 | 0.1265 | 0.2708 |
| **BPR (random negs)**   | **0.5667** | **0.3028** | **0.7988** |

**BPR won by +68.5%.** This was unexpected, so I dug into why.

**My explanation:** ML-1M is heavily popularity-skewed — a few blockbusters dominate user-item interactions. With InfoNCE, every other user's positive item in the batch becomes a "negative" for the current user. So *Toy Story* shows up as a negative for almost every user — including users who would genuinely love *Toy Story*. The model learns to systematically push down popular items.

BPR with popularity-weighted random sampling has the same long-tail awareness but doesn't force every popular item to be a negative everywhere. It's a softer, more honest gradient signal on a skewed distribution.

**Lesson:** The "obvious right answer" in ML papers assumes a particular data distribution. On a skewed dataset, the more theoretically powerful method can underperform a simpler one. **You have to run the ablation.**

**Final numbers:** Recall@10 = 0.5667, NDCG@10 = 0.7988 — roughly **4.3x the MF baseline** (Recall@10 = 0.13).

---

## Decision 3 — Popularity-weighted negative sampling

**Why:** Uniform random negatives over-sample long-tail items the model would never confuse with a positive anyway. Popular items the user *didn't* interact with are the ones the model is most likely to mistakenly recommend, so they're the most informative negatives.

I used a `count^0.75` smoothing (same as word2vec) instead of raw popularity, because raw popularity over-samples a tiny number of blockbuster movies.

---

## Decision 4 — FAISS `IndexFlatIP` (exact) over IVF/HNSW (approximate)

**What I considered:** `IndexIVFFlat` for sub-linear search.

**What I chose:** Exact `IndexFlatIP`.

**Why:** With ~3500 items in ML-1M, exact search runs in <0.5ms. IVF adds complexity (training the index, tuning `nprobe`) without measurable latency savings at this scale. For production with millions of items, I'd switch to IVF or HNSW — and the swap is a one-line change.

This is a deliberate "right tool for the size" decision, not a shortcut.

---

## Decision 5 — Time-based train/test split

**First attempt:** `train_test_split(df, test_size=0.2, random_state=42)`.
Got Recall@10 = 0.72. Suspicious.

**Realisation:** A random split lets the model see "future" interactions during training. In real deployment, you can only train on past data. The 0.72 was data leakage, not actual quality.

**Fix:** Sort by timestamp, take the first 80% as train and the last 20% as test. Recall@10 dropped to ~0.47 — but that number is *honest*.

> Lesson: if your numbers look too good, they probably are.

---

## Bugs I hit (the real story)

### Bug 1 — The MLP was never being trained

The `TwoTowerModel.forward()` method runs embeddings through MLP layers, but the training loop was directly accessing `model.user_embedding(...)` and `model.item_embedding(...)`, bypassing the MLP entirely. The MLP weights were random and unused for the entire training run — I was effectively training a plain matrix-factorisation model with a giant unused MLP attached.

**How I caught it:** <!-- TODO fill in your real story.
e.g. "I noticed Recall wasn't improving past epoch 2. Started printing layer weight gradients
and saw the MLP gradients were ~1e-9 — basically zero. Then I traced it back." -->

**Fix:** Added `encode_users()` / `encode_items()` methods to the model that go through the full tower, and updated trainer / metrics / FAISS index / API to call them.

### Bug 2 — `MFBaseline` crashed during evaluation

After refactoring metrics to call `model.encode_users()`, the MF baseline broke because it didn't have that method. Added matching encoder methods to `MFBaseline` so the same metric code works for both.

### Bug 3 — `KMP_DUPLICATE_LIB_OK` OpenMP crash

FAISS and PyTorch both ship their own OpenMP runtime on macOS. Loading both segfaults. Workaround: set `KMP_DUPLICATE_LIB_OK=TRUE` early in every entry point. Long-term fix would be to install FAISS and PyTorch from the same conda channel so they share an OpenMP library — left as a TODO.

---

## What I'd do differently next time

<!-- TODO Nikhil: write 3-5 bullets in your own words. Some prompts:
- Would you start with content features (user age, movie genre) instead of pure ID-based?
- Would you add early stopping / a learning-rate scheduler?
- Would you set up MLflow / W&B from day one instead of grepping log files?
- Would you write tests before the code?
- Would you scale to ML-25M directly?
-->

---

## What's missing (and what I'd build next)

- **Sequence-aware user representation.** Right now a user is a single ID embedding. A real system would feed the user's recent interaction history into a transformer/GRU.
- **Cold-start handling.** Brand-new users have no embedding. Would need fallback to demographic features.
- **Online evaluation harness.** Offline metrics like Recall@10 don't always correlate with online business metrics.
- **Distillation.** The two-tower model is itself a *retrieval* model — production stacks usually have a heavier reranker on top of the candidates it returns.
