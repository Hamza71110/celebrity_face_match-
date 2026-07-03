"""Lightweight smoke tests for CI.

These deliberately avoid importing TensorFlow / VGGFace so the CI job stays
fast: the API module and helpers import those heavy libraries lazily (inside
functions), so importing them here is cheap. Prediction inference is only
exercised when a trained model is present.
"""
import os

from starlette.testclient import TestClient

from src import config
from src.data_loader import _class_from_filename
from deployment.api import app

client = TestClient(app)


def test_class_parsing_handles_both_separators():
    assert _class_from_filename("data\\Aamir_Khan\\Aamir.100.jpg") == "Aamir_Khan"
    assert _class_from_filename("data/Alia_Bhatt/x.jpg") == "Alia_Bhatt"


def test_config_paths_are_absolute():
    assert os.path.isabs(config.MODELS_DIR)
    assert config.EMBED_DIM == 2048
    assert config.IMG_SIZE == 224


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "Celebrity Face Match API"
    assert "POST /predict" in body["endpoints"]


def test_health_endpoint_responds():
    resp = client.get("/health")
    # 200 when a trained model exists, 503 otherwise - both are valid states
    assert resp.status_code in (200, 503)
    assert "status" in resp.json()


def test_predict_requires_image_content_type():
    # a non-image upload should be rejected (400) or blocked by missing model (503)
    resp = client.post(
        "/predict",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    assert resp.status_code in (400, 503)
