
import numpy as np

from src import config, data_loader

_model_cache = {}
_labels = None


def get_labels():
    global _labels
    if _labels is None:
        _labels = data_loader.load_labels()
    return _labels


def load_model(version="v2"):
    """Lazily load and cache a trained Keras model by version ('v1'/'v2')."""
    if version not in _model_cache:
        from tensorflow.keras.models import load_model as keras_load
        path = config.V2_MODEL_PATH if version == "v2" else config.V1_MODEL_PATH
        _model_cache[version] = keras_load(path)
    return _model_cache[version]


def predict_from_embedding(embedding, version="v2", top_k=5):
    """Return top-k [{'celebrity', 'confidence'}] from a 2048-d embedding."""
    model = load_model(version)
    labels = get_labels()
    probs = model.predict(embedding.reshape(1, -1)).flatten()
    idx = np.argsort(probs)[::-1][:top_k]
    return [
        {"celebrity": labels[i].replace("_", " "), "confidence": float(probs[i])}
        for i in idx
    ]
