def encode_ids(df):
    """
    Map raw user_id / item_id columns to contiguous integer indices (0-based).

    Args:
        df: DataFrame with columns ["user_id", "item_id"].

    Returns:
        (df_encoded, user_map, item_map) where *_map is {raw_id: encoded_idx}.
    """

    # Unique IDs
    user_ids = df["user_id"].unique()
    item_ids = df["item_id"].unique()

    # Create mappings
    user_map = {u: idx for idx, u in enumerate(user_ids)}
    item_map = {i: idx for idx, i in enumerate(item_ids)}

    # Apply mappings
    df = df.copy()
    df["user_idx"] = df["user_id"].map(user_map)
    df["item_idx"] = df["item_id"].map(item_map)

    return df, user_map, item_map