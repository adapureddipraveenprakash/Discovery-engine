import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _bpr_loss(model, users, pos_items, neg_items):
    """Pairwise BPR loss with explicit random negatives. Legacy."""
    user_emb = model.encode_users(users)              # (B, D)
    pos_emb = model.encode_items(pos_items)            # (B, D)

    B, num_neg = neg_items.shape
    neg_emb = model.encode_items(neg_items.view(-1)).view(B, num_neg, -1)  # (B, N, D)

    pos_scores = (user_emb * pos_emb).sum(dim=1)                            # (B,)
    neg_scores = (user_emb.unsqueeze(1) * neg_emb).sum(dim=2)               # (B, N)

    return -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-8).mean()


def _infonce_loss(model, users, pos_items, temperature: float):
    """
    InfoNCE / Sampled-Softmax loss with in-batch negatives.
    This is the loss used by Google's YouTube two-tower paper (Yi et al. 2019).

    Every other user's positive item in the batch is treated as a negative for the
    current user. Cheap (no extra forward pass) and provides B-1 hard negatives for free.

    Note: implemented manually instead of F.cross_entropy because the MPS backend
    on Apple Silicon segfaults on cross_entropy with large class counts.
    """
    user_emb = model.encode_users(users)            # (B, D)
    item_emb = model.encode_items(pos_items)         # (B, D)

    logits = (user_emb @ item_emb.T) / temperature   # (B, B)
    log_probs = F.log_softmax(logits, dim=1)
    B = user_emb.size(0)
    diag = torch.arange(B, device=user_emb.device)
    return -log_probs[diag, diag].mean()


def train_two_tower(model, dataloader, optimizer, device, epochs, loss_type="infonce", temperature=0.07):
    """
    Train the two-tower model.

    Args:
        loss_type: "infonce" (in-batch negatives, recommended) or "bpr" (random negatives).
        temperature: Used by InfoNCE loss. Lower = sharper softmax. 0.05-0.1 is typical.
    """
    model.to(device)
    logger.info("Training with loss=%s temperature=%.3f", loss_type, temperature)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for users, pos_items, neg_items in dataloader:
            users = users.to(device)
            pos_items = pos_items.to(device)

            optimizer.zero_grad()

            if loss_type == "infonce":
                loss = _infonce_loss(model, users, pos_items, temperature)
            elif loss_type == "bpr":
                neg_items = neg_items.to(device)
                loss = _bpr_loss(model, users, pos_items, neg_items)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info("Epoch %d/%d  %s_loss=%.4f", epoch + 1, epochs, loss_type, avg_loss)
