"""Face detection + VGGFace embedding extraction.

Mirrors exactly the preprocessing used in the original app.py / test.py so the
API produces embeddings compatible with the trained models. Heavy objects
(MTCNN detector, VGGFace backbone) are lazily created and cached.
"""
import numpy as np
from PIL import Image

from src import config

_detector = None
_vggface = None


def get_detector():
    global _detector
    if _detector is None:
        from mtcnn import MTCNN
        _detector = MTCNN()
    return _detector


def get_vggface():
    """Frozen VGGFace ResNet50 feature extractor (2048-d, avg pooling)."""
    global _vggface
    if _vggface is None:
        from keras_vggface.vggface import VGGFace
        _vggface = VGGFace(
            model="resnet50",
            include_top=False,
            input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
            pooling="avg",
        )
    return _vggface


def image_bytes_to_bgr(data):
    """Decode raw uploaded bytes into a BGR numpy array (OpenCV convention)."""
    img = Image.open(_bytes_io(data)).convert("RGB")
    rgb = np.asarray(img)
    return rgb[:, :, ::-1]  # RGB -> BGR


def _bytes_io(data):
    import io
    return io.BytesIO(data)


def detect_and_crop(bgr_image):
    """Return the largest detected face crop, or None if no face is found."""
    detector = get_detector()
    results = detector.detect_faces(bgr_image)
    if not results:
        return None
    # pick the highest-confidence detection
    best = max(results, key=lambda r: r.get("confidence", 0))
    x, y, w, h = best["box"]
    x, y = max(0, x), max(0, y)
    return bgr_image[y:y + h, x:x + w]


def embed_face(face_bgr):
    """Crop -> 224x224 -> VGGFace preprocess -> 2048-d embedding."""
    from keras_vggface.utils import preprocess_input

    image = Image.fromarray(face_bgr)
    image = image.resize((config.IMG_SIZE, config.IMG_SIZE))
    arr = np.asarray(image).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return get_vggface().predict(arr).flatten()


def embedding_from_bytes(data):
    """Full pipeline: uploaded bytes -> 2048-d embedding (or None if no face)."""
    bgr = image_bytes_to_bgr(data)
    face = detect_and_crop(bgr)
    if face is None or face.size == 0:
        return None
    return embed_face(face)
