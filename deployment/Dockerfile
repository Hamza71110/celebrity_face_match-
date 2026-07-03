# ============================================================
#  Celebrity Face Match  -  FastAPI model-serving container
# ============================================================
# Separate from the root Dockerfile (which serves the Streamlit app). This one
# serves the trained model through a REST API and is what Kubernetes deploys.
#
# Build from the PROJECT ROOT so the build context contains src/ and models/:
#   docker build -f deployment/Dockerfile -t celebrity-face-api:v2 .
#   docker run -p 8000:8000 celebrity-face-api:v2
FROM python:3.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVED_MODEL=v2

# OpenCV runtime libs (cv2 -> libGL, libglib)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install base + API deps (cached across code changes)
COPY requirements.txt requirements-api.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt -r requirements-api.txt

# Application code + shared package + trained model artifacts
COPY src/ ./src/
COPY deployment/ ./deployment/
COPY models/ ./models/

EXPOSE 8000

# Uvicorn ASGI server, reachable from outside the container
CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
