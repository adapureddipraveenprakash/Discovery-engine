import pandas as pd

def time_based_split(df, test_ratio=0.2):
    """Sort interactions by timestamp and split chronologically to prevent data leakage."""
    df = df.sort_values("timestamp")

    split_index = int(len(df) * (1 - test_ratio))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df