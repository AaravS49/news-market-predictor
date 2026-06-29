"""PyTorch classifier and Dataset wrapper for the news-market-predictor."""
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class NewsMarketClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 13,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "NewsMarketClassifier":
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        return model


class NewsMarketDataset(Dataset):
    """Wraps (X, y) or (X, y, returns) tensors for use with DataLoader."""

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        returns: torch.Tensor | None = None,
    ):
        self.X = X
        self.y = y
        self.returns = returns

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.returns is not None:
            return self.X[idx], self.y[idx], self.returns[idx]
        return self.X[idx], self.y[idx]
