
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



FACE_ERROR_MESSAGES = {
    "blurry": "Image is too blurry or low quality — please upload a clearer photo.",
    "no_face": "No face detected — please upload a clear photo that contains a face.",
    "multiple_faces": ("Multiple faces detected — please upload a photo where you "
                       "are the main person."),
}


def is_blurry(bgr_image):
    """True if the image is too low-quality for reliable detection."""
    import cv2
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < config.BLUR_THRESHOLD


def detect_main_face(bgr_image):
    
    detector = get_detector()
    results = detector.detect_faces(bgr_image)

    boxes = []
    for r in results:
        x, y, w, h = r["box"]
        x, y = max(0, x), max(0, y)
        if w > 0 and h > 0:
            boxes.append((w * h, x, y, w, h))
    boxes.sort(reverse=True, key=lambda b: b[0])

    if not boxes:
        return {"error": "blurry" if is_blurry(bgr_image) else "no_face"}
    if (len(boxes) >= 2
            and boxes[1][0] / boxes[0][0] >= config.FACE_AMBIGUITY_RATIO):
        return {"error": "multiple_faces"}

    _, x, y, w, h = boxes[0]
    face = bgr_image[y:y + h, x:x + w]
    return {"face": face} if face.size else {"error": "no_face"}


def detect_and_crop(bgr_image):
    """Backward-compatible helper: the largest face crop, or None."""
    return detect_main_face(bgr_image).get("face")


def embed_face(face_bgr):
    """Crop -> 224x224 -> VGGFace preprocess -> 2048-d embedding."""
    from keras_vggface.utils import preprocess_input

    image = Image.fromarray(face_bgr)
    image = image.resize((config.IMG_SIZE, config.IMG_SIZE))
    arr = np.asarray(image).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return get_vggface().predict(arr).flatten()


def analyze_bytes(data):
    """Full pipeline: uploaded bytes -> {'embedding': vec} or {'error': code}."""
    bgr = image_bytes_to_bgr(data)
    result = detect_main_face(bgr)
    if "error" in result:
        return result
    return {"embedding": embed_face(result["face"])}


def embedding_from_bytes(data):
    """Backward-compatible: full pipeline -> 2048-d embedding, or None."""
    return analyze_bytes(data).get("embedding")
