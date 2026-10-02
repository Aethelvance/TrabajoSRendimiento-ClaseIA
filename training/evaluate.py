"""Evalua best_model.pt en test: MAE, RMSE, R2 (solo torch)."""
import torch

from .config import BEST_MODEL, HIDDEN1, HIDDEN2, INPUT_DIM, OUTPUT_DIM
from .dataset import load_splits
from .model import MLPRegressor


def evaluate():
    d = load_splits()
    model = MLPRegressor(INPUT_DIM, HIDDEN1, HIDDEN2, OUTPUT_DIM)
    model.load_state_dict(torch.load(BEST_MODEL, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        pred_norm = model(d["X_test"])
    pred = pred_norm * d["y_std"] + d["y_mean"]
    real = d["y_test_raw"]
    mae = (pred - real).abs().mean().item()
    rmse = float((((pred - real) ** 2).mean()).sqrt().item())
    ss_res = ((real - pred) ** 2).sum()
    ss_tot = ((real - real.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()
    print(f"MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


if __name__ == "__main__":
    evaluate()
