import logging

import faiss
import torch

logger = logging.getLogger(__name__)


def build_faiss_index(model, num_items: int, device) -> faiss.Index:
    """
    Extract item embeddings through the full item tower and build a FAISS inner-product index.

    Because embeddings are L2-normalized, inner product == cosine similarity, so
    IndexFlatIP gives exact nearest-neighbour search without a separate normalisation step.

    Args:
        model: TwoTowerModel with an encode_items() method.
        num_items: Total number of items in the vocabulary.
        device: torch.device to run embedding extraction on.

    Returns:
        faiss.IndexFlatIP populated with all item embeddings.
    """
    model.eval()
    with torch.no_grad():
        item_ids = torch.arange(num_items).to(device)
        item_embeddings = model.encode_items(item_ids)  # (num_items, D), already L2-normalised

    embeddings_np = item_embeddings.cpu().numpy().astype("float32")
    logger.info("Item embeddings shape: %s", embeddings_np.shape)

    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    logger.info("FAISS index built with %d vectors", index.ntotal)
    return index
