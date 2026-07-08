import os


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data + artifacts
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
EMBEDDING_PKL = os.path.join(ROOT_DIR, "embedding.pkl")
FILENAMES_PKL = os.path.join(ROOT_DIR, "filenames.pkl")

# Saved model / label artifacts 
V1_MODEL_PATH = os.path.join(MODELS_DIR, "v1_baseline.h5")
V2_MODEL_PATH = os.path.join(MODELS_DIR, "v2_transfer.h5")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")

# Image / model settings 
IMG_SIZE = 224
EMBED_DIM = 2048  # VGGFace ResNet50  output size


BLUR_THRESHOLD = 30.0          
FACE_AMBIGUITY_RATIO = 0.85    


MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "sqlite:///" + os.path.join(ROOT_DIR, "mlflow.db")
)
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "celebrity-face-match")


DEFAULT_SERVED_MODEL = os.environ.get("SERVED_MODEL", "v2")


def ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)
