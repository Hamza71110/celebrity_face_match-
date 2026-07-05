"""FastAPI serving layer for the Celebrity Face Match model.

Endpoints required by the assignment:
    GET  /         - service info / metadata
    GET  /health   - liveness + readiness (is the model loaded?)
    POST /predict  - upload a face image, return the predicted celebrity

Run locally:
    uvicorn deployment.api:app --host 0.0.0.0 --port 8000

Interactive docs are auto-generated at /docs (Swagger) and /redoc.
"""
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src import config

app = FastAPI(
    title="Celebrity Face Match API",
    description="Upload a face photo and get the most similar Bollywood "
                "celebrity, powered by a VGGFace transfer-learning model.",
    version="2.0.0",
)

MODEL_VERSION = config.DEFAULT_SERVED_MODEL  # 'v2' by default


def _model_file():
    return config.V2_MODEL_PATH if MODEL_VERSION == "v2" else config.V1_MODEL_PATH


def _model_available():
    return os.path.exists(_model_file()) and os.path.exists(config.LABELS_PATH)


@app.get("/")
def root():
    """Service metadata and available routes."""
    return {
        "service": "Celebrity Face Match API",
        "version": app.version,
        "served_model": MODEL_VERSION,
        "endpoints": {
            "GET /": "this info",
            "GET /health": "health check",
            "POST /predict": "multipart image upload -> celebrity prediction",
            "GET /docs": "interactive Swagger UI",
        },
    }


@app.get("/health")
def health():
    """Return 200 when the process is up and the model artifacts are present."""
    ready = _model_available()
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "healthy" if ready else "model_not_loaded",
            "model_version": MODEL_VERSION,
            "model_ready": ready,
        },
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 5):
    """Detect the face in the uploaded image and classify the celebrity."""
    if not _model_available():
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run `python -m training.train "
                   "--model v2` to produce models/v2_transfer.h5.",
        )
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image.")

    data = await file.read()

    # imported lazily so the module loads fast and /health stays cheap
    from starlette.concurrency import run_in_threadpool

    from src import face_utils, inference

    # Run the heavy CPU work (MTCNN + VGGFace + classifier) in a threadpool so
    # the event loop stays free to answer the /health liveness probe. Otherwise
    # a single prediction blocks /health and Kubernetes restarts the pod.
    result = await run_in_threadpool(face_utils.analyze_bytes, data)
    if "error" in result:
        raise HTTPException(
            status_code=422,
            detail=face_utils.FACE_ERROR_MESSAGES[result["error"]],
        )

    predictions = await run_in_threadpool(
        inference.predict_from_embedding, result["embedding"], MODEL_VERSION, top_k
    )
    return {
        "filename": file.filename,
        "model_version": MODEL_VERSION,
        "prediction": predictions[0],
        "top_k": predictions,
    }
