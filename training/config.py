from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "raw" / "Rendimiento.csv"

MODELS_DIR = ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
BEST_MODEL = MODELS_DIR / "best_model.pt"
SCALER_FILE = MODELS_DIR / "scaler.pt"

# MLP 2-32-16-1 (641 params)
INPUT_DIM = 2
HIDDEN1 = 32
HIDDEN2 = 16
OUTPUT_DIM = 1

BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 500
PATIENCE_EARLY = 50
PATIENCE_LR = 20
SEED = 42
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15  # resto 0.15 test
