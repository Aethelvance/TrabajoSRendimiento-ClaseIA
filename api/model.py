"""MLP 2-32-16-1 idéntico a training/model.py.

Copia autocontenida: la API nunca importa training/ (ver diagrama 04).
Si cambias la arquitectura, reentrena y actualiza ambos archivos.
"""
import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, input_dim=2, h1=32, h2=16, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, output_dim),
        )

    def forward(self, x):
        return self.net(x)
