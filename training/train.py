"""Entrena MLP con MSELoss + AdamW, guarda checkpoints y best_model.pt."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    BATCH_SIZE, BEST_MODEL, CHECKPOINTS_DIR, HIDDEN1, HIDDEN2,
    INPUT_DIM, LR, MAX_EPOCHS, OUTPUT_DIM, PATIENCE_EARLY,
    PATIENCE_LR, SCALER_FILE, WEIGHT_DECAY,
)
from .dataset import load_splits
from .model import MLPRegressor, count_params


def train():
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL.parent.mkdir(parents=True, exist_ok=True)

    d = load_splits()
    train_ds = TensorDataset(d["X_train"], d["y_train"])
    val_ds = TensorDataset(d["X_val"], d["y_val"])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = MLPRegressor(INPUT_DIM, HIDDEN1, HIDDEN2, OUTPUT_DIM)
    print(f"params: {count_params(model)}")  # esperado 641

    loss_fn = nn.MSELoss()  # entrena MSE, reporta MAE/RMSE/R2
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=PATIENCE_LR, factor=0.5)

    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        # val
        model.eval()
        with torch.no_grad():
            v = sum(loss_fn(model(xb), yb).item() * len(xb) for xb, yb in val_loader)
            v /= len(val_loader.dataset)
        sched.step(v)

        # checkpoint cada 50 epochs
        if epoch % 50 == 0:
            torch.save(model.state_dict(), CHECKPOINTS_DIR / f"epoch_{epoch:03d}.pt")

        if v < best_val:
            best_val = v
            bad_epochs = 0
            torch.save(model.state_dict(), BEST_MODEL)
            torch.save({"x_mean": d["x_mean"], "x_std": d["x_std"],
                        "y_mean": d["y_mean"], "y_std": d["y_std"]}, SCALER_FILE)
        else:
            bad_epochs += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} train_loss={loss.item():.4f} val_mse={v:.4f} lr={opt.param_groups[0]['lr']:.1e}")

        if bad_epochs >= PATIENCE_EARLY:
            print(f"EarlyStopping en epoch {epoch}, best_val={best_val:.4f}")
            break

    print(f"best -> {BEST_MODEL}")


if __name__ == "__main__":
    train()
