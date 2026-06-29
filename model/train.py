"""Training loop with early stopping for NewsMarketClassifier."""
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.architecture import NewsMarketClassifier, NewsMarketDataset

logger = logging.getLogger(__name__)

CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "best_model.pt"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def get_dataloaders(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 32,
    returns: torch.Tensor | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Stratified 80/10/10 split. Returns (train_loader, val_loader, test_loader)."""
    idx = list(range(len(X)))
    y_int = y.long().tolist()

    idx_tv, idx_test = train_test_split(idx, test_size=0.10, stratify=y_int, random_state=42)
    y_tv = [y_int[i] for i in idx_tv]
    idx_train, idx_val = train_test_split(idx_tv, test_size=1 / 9, stratify=y_tv, random_state=42)

    def make_loader(indices: list[int], shuffle: bool = False) -> DataLoader:
        """Create a DataLoader for the given sample indices."""
        X_s = X[indices]
        y_s = y[indices]
        r_s = returns[indices] if returns is not None else None
        return DataLoader(
            NewsMarketDataset(X_s, y_s, r_s), batch_size=batch_size, shuffle=shuffle
        )

    train_loader = make_loader(idx_train, shuffle=True)
    val_loader = make_loader(idx_val)
    test_loader = make_loader(idx_test)

    logger.info("Split: train=%d | val=%d | test=%d", len(idx_train), len(idx_val), len(idx_test))
    return train_loader, val_loader, test_loader


def _weighted_bce_loss(
    preds: torch.Tensor, targets: torch.Tensor, pos_weight: float
) -> torch.Tensor:
    """BCELoss with per-sample class weighting (mirrors BCEWithLogitsLoss pos_weight)."""
    weights = torch.where(
        targets == 1,
        torch.tensor(pos_weight, device=preds.device),
        torch.tensor(1.0, device=preds.device),
    )
    bce = nn.BCELoss(reduction="none")(preds, targets)
    return (bce * weights).mean()


def train(session, chroma_collection) -> NewsMarketClassifier:
    """Build dataset, train with early stopping, return best model loaded from checkpoint."""
    from features.builder import build_dataset

    logger.info("Using device: %s", device)
    logger.info("Building dataset (FinBERT runs once per article, cached after)...")
    X, y, returns = build_dataset(session, chroma_collection)

    pos = int(y.sum().item())
    neg = len(y) - pos
    pos_weight = neg / pos if pos > 0 else 1.0
    use_weighted = pos / len(y) < 0.4 or pos / len(y) > 0.6
    if use_weighted:
        logger.info("Class imbalance detected: applying pos_weight=%.3f", pos_weight)

    train_loader, val_loader, _ = get_dataloaders(X, y, returns=returns)

    model = NewsMarketClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 10
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, 101):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            X_b, y_b = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            preds = model(X_b)
            loss = (
                _weighted_bce_loss(preds, y_b, pos_weight) if use_weighted
                else criterion(preds, y_b)
            )
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses: list[float] = []
        val_probs: list[float] = []
        val_labels: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                X_b, y_b = batch[0].to(device), batch[1].to(device)
                preds = model(X_b)
                loss = (
                    _weighted_bce_loss(preds, y_b, pos_weight) if use_weighted
                    else criterion(preds, y_b)
                )
                val_losses.append(loss.item())
                val_probs.extend(preds.cpu().tolist())
                val_labels.extend(y_b.cpu().tolist())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except Exception:
            val_auc = 0.5

        logger.info(
            "Epoch %3d | Train Loss: %.4f | Val Loss: %.4f | Val AUC: %.4f",
            epoch, train_loss, val_loss, val_auc,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save(str(CHECKPOINT_PATH))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

    logger.info("Best val loss: %.4f. Loading best checkpoint...", best_val_loss)
    return NewsMarketClassifier.load(str(CHECKPOINT_PATH))


if __name__ == "__main__":
    from config import setup_logging
    from db.session import get_session, init_db
    from rag.retriever import get_or_create_collection

    setup_logging()
    init_db()
    collection = get_or_create_collection()
    with get_session() as session:
        best_model = train(session, collection)
    logger.info("Training complete. Model saved to %s", CHECKPOINT_PATH)
