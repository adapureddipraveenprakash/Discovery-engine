import numpy as np
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """
    PyTorch Dataset that yields (user, pos_item, neg_items) triples.

    Negative sampling strategies:
        - "uniform"     : sample uniformly at random from all items
        - "popularity"  : sample weighted by item popularity (hard negatives —
                          popular items the user *didn't* interact with are
                          informative because the model will tend to recommend them).
    """

    def __init__(self, df, num_items: int, num_negatives: int = 4, sampling: str = "uniform"):
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.sampling = sampling

        self.users = df["user_idx"].values
        self.items = df["item_idx"].values

        # user → set of positive items (for negative-sampling rejection)
        self.user_pos_items: dict[int, set] = {}
        for u, i in zip(self.users, self.items):
            self.user_pos_items.setdefault(u, set()).add(i)

        # Popularity distribution (smoothed with 0.75 power, like word2vec)
        if sampling == "popularity":
            counts = np.bincount(self.items, minlength=num_items).astype(np.float64)
            counts = np.power(counts, 0.75)
            self.pop_probs = counts / counts.sum()
        else:
            self.pop_probs = None

        self.interactions = list(zip(self.users, self.items))

    def __len__(self):
        return len(self.interactions)

    def sample_negatives(self, user):
        positives = self.user_pos_items[user]
        neg_items = []
        while len(neg_items) < self.num_negatives:
            if self.pop_probs is not None:
                # Sample a small batch at once — ~3x faster than per-item draws
                candidates = np.random.choice(self.num_items, size=self.num_negatives * 2, p=self.pop_probs)
            else:
                candidates = np.random.randint(0, self.num_items, size=self.num_negatives * 2)
            for c in candidates:
                if c not in positives:
                    neg_items.append(int(c))
                    if len(neg_items) == self.num_negatives:
                        break
        return neg_items

    def __getitem__(self, idx):
        user, pos_item = self.interactions[idx]
        neg_items = self.sample_negatives(user)
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long),
        )
