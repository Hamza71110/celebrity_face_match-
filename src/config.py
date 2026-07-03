"""Central configuration and paths for the MLOps pipeline.

Everything is derived from the project root so the code runs the same way on
Windows (development) and Linux (Docker / Kubernetes).
"""
import os

# Project root = one level above this file's directory (src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data + artifacts
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
EMBEDDING_PKL = os.path.join(ROOT_DIR, "embedding.pkl")
FILENAMES_PKL = os.path.join(ROOT_DIR, "filenames.pkl")

# Saved model / label artifacts (produced by training/train.py)
V1_MODEL_PATH = os.path.join(MODELS_DIR, "v1_baseline.h5")
V2_MODEL_PATH = os.path.join(MODELS_DIR, "v2_transfer.h5")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")

# Image / model settings (must match the VGGFace pipeline already in the repo)
IMG_SIZE = 224
EMBED_DIM = 2048  # VGGFace ResNet50 avg-pool output size

# MLflow. A SQLite backend (not the plain file store) is used so the Model
# Registry works -> required for the "registered model" deliverable. Artifacts
# still land under ./mlartifacts. Override with the MLFLOW_TRACKING_URI env var.
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "sqlite:///" + os.path.join(ROOT_DIR, "mlflow.db")
)
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "celebrity-face-match")

# The API defaults to the improved model but this can be overridden via env.
DEFAULT_SERVED_MODEL = os.environ.get("SERVED_MODEL", "v2")


def ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)
