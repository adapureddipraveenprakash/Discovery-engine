import pandas as pd
import pytest

from src.data.encoder import encode_ids
from src.data.loader import InteractionDataset
from src.data.split import time_based_split


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "user_id":   [10, 10, 20, 20, 30, 30, 40],
        "item_id":   [100, 200, 100, 300, 200, 400, 100],
        "rating":    [5, 4, 5, 3, 4, 5, 5],
        "timestamp": [1, 2, 3, 4, 5, 6, 7],
    })


def test_encode_ids_produces_contiguous_indices(sample_df):
    df, user_map, item_map = encode_ids(sample_df)
    assert set(user_map.values()) == {0, 1, 2, 3}
    assert set(item_map.values()) == {0, 1, 2, 3}
    assert df["user_idx"].max() == 3
    assert df["item_idx"].max() == 3


def test_time_based_split_is_chronological(sample_df):
    train, test = time_based_split(sample_df, test_ratio=0.3)
    assert train["timestamp"].max() <= test["timestamp"].min()


def test_dataset_negative_sampling_excludes_positives(sample_df):
    df, _, _ = encode_ids(sample_df)
    ds = InteractionDataset(df, num_items=4, num_negatives=2, sampling="uniform")
    for i in range(len(ds)):
        user, pos, negs = ds[i]
        positives_for_user = ds.user_pos_items[user.item()]
        for n in negs.tolist():
            assert n not in positives_for_user, "negative must not be a positive"


def test_dataset_popularity_sampling_runs(sample_df):
    df, _, _ = encode_ids(sample_df)
    ds = InteractionDataset(df, num_items=4, num_negatives=2, sampling="popularity")
    user, pos, negs = ds[0]
    assert negs.shape == (2,)
