# ============================================================
#  Celebrity Face Matcher  -  Streamlit + TensorFlow/VGGFace
# ============================================================
# Python 3.7 is required by this ML stack (TensorFlow 2.3.1,
# keras-vggface 0.6, numpy 1.18.5). Do not bump the base image.
FROM python:3.7-slim

# Cleaner Python/pip behaviour inside containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libraries needed by OpenCV (cv2 -> libGL, libglib)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached across code changes
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the app code, model artifacts (embedding.pkl / filenames.pkl)
# and the celebrity image data used to display the match.
COPY . .

# Streamlit saves uploaded images here at runtime
RUN mkdir -p uploads

# Port the app listens on (matches the assignment example: -p 8000:8000)
EXPOSE 8000

# Start the Streamlit server, reachable from outside the container
CMD ["streamlit", "run", "app.py", \
     "--server.port=8000", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
