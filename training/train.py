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

    # AdamW manual: torch.optim esta roto en torch 2.14.1+cpu/cp314
    # (torch._dynamo NP_SUPPORTED_MODULES). Mismas cuentas que AdamW lr=1e-3.
    params = [p for p in model.parameters() if p.requires_grad]
    m_buf = [torch.zeros_like(p) for p in params]
    v_buf = [torch.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    lr = LR
    step = 0

    @torch.no_grad()
    def adamw_step():
        nonlocal step
        step += 1
        bc1 = 1.0 - b1 ** step
        bc2 = 1.0 - b2 ** step
        for p, m, v in zip(params, m_buf, v_buf):
            if p.grad is None:
                continue
            g = p.grad
            m.mul_(b1).add_(g, alpha=1.0 - b1)
            v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
            m_hat = m / bc1
            v_hat = v / bc2
            p.mul_(1.0 - lr * WEIGHT_DECAY)  # decoupled weight decay
            p.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)

    best_val = float("inf")
    bad_epochs = 0
    bad_lr = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            model.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            adamw_step()

        # val
        model.eval()
        with torch.no_grad():
            v = sum(loss_fn(model(xb), yb).item() * len(xb) for xb, yb in val_loader)
            v /= len(val_loader.dataset)
        # ReduceLROnPlateau manual: halve lr si no mejora en PATIENCE_LR
        if v < best_val:
            bad_lr = 0
        else:
            bad_lr += 1
            if bad_lr >= PATIENCE_LR:
                lr *= 0.5
                bad_lr = 0

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
            print(f"epoch {epoch:3d} train_loss={loss.item():.4f} val_mse={v:.4f} lr={lr:.1e}")

        if bad_epochs >= PATIENCE_EARLY:
            print(f"EarlyStopping en epoch {epoch}, best_val={best_val:.4f}")
            break

    print(f"best -> {BEST_MODEL}")


if __name__ == "__main__":
    train()
