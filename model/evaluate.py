"""Evaluation metrics and backtest simulation for NewsMarketClassifier."""
import logging
import sys
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.architecture import NewsMarketClassifier

logger = logging.getLogger(__name__)

CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "best_model.pt"
CM_PATH = Path(__file__).resolve().parent / "checkpoints" / "confusion_matrix.png"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _collect_predictions(
    model: NewsMarketClassifier, loader: DataLoader
) -> tuple[list[float], list[float], list[float] | None]:
    """Run inference on a DataLoader and return (probs, labels, returns)."""
    model.eval()
    all_probs, all_labels, all_returns = [], [], []
    with torch.no_grad():
        for batch in loader:
            X_b, y_b = batch[0].to(device), batch[1]
            ret_b = batch[2] if len(batch) == 3 else None
            probs = model(X_b).cpu()
            all_probs.extend(probs.tolist())
            all_labels.extend(y_b.tolist())
            if ret_b is not None:
                all_returns.extend(ret_b.tolist())
    return all_probs, all_labels, all_returns or None


def evaluate(model: NewsMarketClassifier, test_loader: DataLoader) -> dict:
    """Compute classification metrics and save a confusion matrix PNG."""
    probs, labels, _ = _collect_predictions(model, test_loader)
    preds = [1 if p >= 0.5 else 0 for p in probs]

    logger.info("=== Classification Report ===")
    report = classification_report(labels, preds, target_names=["DOWN", "UP"])
    print(report)

    auc = roc_auc_score(labels, probs)
    logger.info("ROC-AUC: %.4f", auc)

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["DOWN", "UP"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Test Set")
    CM_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(CM_PATH), dpi=120)
    plt.close()
    logger.info("Confusion matrix saved → %s", CM_PATH)

    macro_f1 = f1_score(labels, preds, average="macro")
    return {"auc": auc, "f1_macro": macro_f1, "cm": cm}


def backtest(
    model: NewsMarketClassifier, test_loader: DataLoader, threshold: float = 0.6
) -> dict:
    """Simulate buying when predicted_prob > threshold; report trade statistics."""
    probs, labels, returns = _collect_predictions(model, test_loader)

    if returns is None:
        logger.warning("No return data in loader — skipping backtest.")
        return {}

    trade_returns = [returns[i] for i, p in enumerate(probs) if p > threshold]
    n_trades = len(trade_returns)

    if n_trades == 0:
        logger.info("No trades taken above threshold=%.2f.", threshold)
        return {"n_trades": 0}

    mean_ret = sum(trade_returns) / n_trades
    win_rate = sum(1 for r in trade_returns if r > 0) / n_trades
    std_ret = (
        (sum((r - mean_ret) ** 2 for r in trade_returns) / (n_trades - 1)) ** 0.5
        if n_trades > 1 else 0.0
    )
    sharpe = (mean_ret / std_ret * sqrt(252)) if std_ret > 0 else 0.0

    logger.info("=== Backtest (threshold=%.2f) ===", threshold)
    logger.info("Total trades      : %d", n_trades)
    logger.info("Mean return/trade : %+.4f (%+.2f%%)", mean_ret, 100 * mean_ret)
    logger.info("Win rate          : %.1f%%", 100 * win_rate)
    logger.info("Sharpe ratio      : %.3f", sharpe)

    return {
        "n_trades": n_trades,
        "mean_return": mean_ret,
        "win_rate": win_rate,
        "sharpe": sharpe,
    }


if __name__ == "__main__":
    from config import setup_logging
    from db.session import get_session, init_db
    from features.builder import build_dataset
    from model.train import get_dataloaders
    from rag.retriever import get_or_create_collection

    setup_logging()
    init_db()
    collection = get_or_create_collection()

    logger.info("Rebuilding dataset for test split...")
    with get_session() as session:
        X, y, returns = build_dataset(session, collection)

    _, _, test_loader = get_dataloaders(X, y, returns=returns)

    logger.info("Loading model from %s...", CHECKPOINT_PATH)
    model = NewsMarketClassifier.load(str(CHECKPOINT_PATH)).to(device)

    eval_results = evaluate(model, test_loader)
    bt_results = backtest(model, test_loader)

    print("\n" + "=" * 42)
    print("         FINAL RESULTS")
    print("=" * 42)
    print(f"AUC:               {eval_results.get('auc', 0):.4f}")
    print(f"F1 (macro):        {eval_results.get('f1_macro', 0):.4f}")
    print(f"Backtest trades:   {bt_results.get('n_trades', 0)}")
    win = bt_results.get('win_rate', 0)
    print(f"Backtest win rate: {100 * win:.1f}%")
    print(f"Sharpe ratio:      {bt_results.get('sharpe', 0):.3f}")
    print("=" * 42)
