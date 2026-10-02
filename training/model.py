"""MLP regresion PyTorch: 2 -> 32 ReLU -> 16 ReLU -> 1 lineal."""
import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, input_dim=2, h1=32, h2=16, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),  # ReLU(x) = max(0, x)
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, output_dim),  # sin activacion: y continua
        )
        # init Kaiming para ReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
