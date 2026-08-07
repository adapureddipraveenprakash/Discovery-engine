"""
Build FAISS index from trained item embeddings.

Usage: python scripts/build_index.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from index.build_index import build_index_from_embeddings, save_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS = ROOT / "artifacts"


def main() -> None:
    emb_path = ARTIFACTS / "item_embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"{emb_path} not found. Run `make train` or `python training/train_user_tower.py` first."
        )

    embeddings = np.load(emb_path)
    index = build_index_from_embeddings(embeddings)
    save_index(index, ARTIFACTS / "faiss.index")
    logger.info("Index saved to %s", ARTIFACTS / "faiss.index")


if __name__ == "__main__":
    main()
