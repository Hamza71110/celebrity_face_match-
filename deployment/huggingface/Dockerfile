# Hugging Face Docker Space for the Celebrity Face Match API.
# HF Spaces serve on port 7860. This image clones the GitHub repo (which
# contains the code + the committed models/v2_transfer.h5) and runs uvicorn.
FROM python:3.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVED_MODEL=v2

# git (to fetch the repo) + OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the project (public repo). Pin to a branch for reproducibility.
RUN git clone --depth 1 --branch main \
    https://github.com/Hamza71110/celebrity_face_match-.git repo

WORKDIR /app/repo

RUN pip install --upgrade pip && \
    pip install -r requirements.txt -r requirements-api.txt

EXPOSE 7860

CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "7860"]
