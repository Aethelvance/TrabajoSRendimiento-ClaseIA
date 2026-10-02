"""Carga Rendimiento.csv -> tensores normalizados (solo torch + csv stdlib).

Evita pandas/sklearn porque numpy.random esta bloqueado por AppControl
en este equipo. Usa torch (RNG propio) para shuffle y estadisticas.
"""
import csv

import torch

from .config import DATA_CSV, SEED, TRAIN_SPLIT, VAL_SPLIT


def load_splits():
    xs, ys = [], []
    with open(DATA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append([float(row["deporte"]), float(row["sueno"])])
            ys.append([float(row["rendimiento"])])
    X = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)

    g = torch.Generator().manual_seed(SEED)
    n = len(X)
    perm = torch.randperm(n, generator=g)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)
    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    # StandardScaler manual fit solo en train
    x_mean, x_std = X_train.mean(0), X_train.std(0, unbiased=False)
    y_mean, y_std = y_train.mean(0), y_train.std(0, unbiased=False)
    x_std = torch.clamp(x_std, min=1e-8)
    y_std = torch.clamp(y_std, min=1e-8)

    return {
        "X_train": (X_train - x_mean) / x_std,
        "y_train": (y_train - y_mean) / y_std,
        "X_val": (X_val - x_mean) / x_std,
        "y_val": (y_val - y_mean) / y_std,
        "X_test": (X_test - x_mean) / x_std,
        "y_test": (y_test - y_mean) / y_std,
        "y_test_raw": y_test,
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
    }
