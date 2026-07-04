# 🎬 Celebrity Face Matcher

Upload a photo and discover which Bollywood celebrity you resemble. The app detects
the face in your image, extracts a deep facial embedding with **VGGFace (ResNet-50)**,
and finds the closest match among **8,600+** celebrity images using cosine similarity.

> ### 🔗 Live Demos (Hugging Face Spaces)
>
> - **Streamlit app** (face match UI): https://Hamzurna-celebrity-face-match.hf.space
> - **REST API** (FastAPI, V2 model): https://hamzurna-celebrity-face-match-3b11cfc.hf.space — interactive docs at [`/docs`](https://hamzurna-celebrity-face-match-3b11cfc.hf.space/docs)

---

## ✨ Features

- 📤 **Upload** a face photo (`jpg`, `jpeg`, `png`) **or** 📸 **capture** one live from your camera
- 🧠 **Robust face detection** with **MTCNN** — works on full-body / cluttered photos,
  auto-selects the **main (largest) face**, with clear messages for blurry / no-face /
  ambiguous multi-face images
- 🧬 Deep feature extraction with **VGGFace / ResNet-50**
- 🏆 **Top-3** look-alikes with a visual **confidence bar** in a unified "You vs Celebrity" card
- 💭 Optional "who do *you* think you look like?" prompt
- 📱 **Mobile-responsive** UI + a **Contact Me** page
- 🎨 Clean, modern Streamlit UI
- 🐳 Fully containerized with Docker

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| UI | Streamlit |
| Face detection | MTCNN |
| Feature extraction | keras-vggface (VGGFace, ResNet-50) |
| Deep learning | TensorFlow 2.3.1 / Keras 2.4.3 |
| Similarity | scikit-learn (cosine similarity) |
| Image processing | OpenCV, Pillow |
| Containerization | Docker |

---

## 🚀 Run Locally

> **Note:** The precomputed model artifacts `embedding.pkl` and `filenames.pkl` are
> **not** included in this repository (large / generated files). Generate them once
> from the included `data/` folder using `feature_extract.py`, then run the app.

```bash
# 1. Install dependencies (Python 3.7 recommended)
pip install -r requirements.txt

# 2. Generate embeddings (creates filenames.pkl + embedding.pkl)
python feature_extract.py

# 3. Launch the app
streamlit run app.py
```

Then open <http://localhost:8501>.

---

## 🐳 Run with Docker

```bash
# Build the image
docker build -t final-project:v1 .

# Run the container
docker run -p 8000:8000 final-project:v1
```

Then open <http://localhost:8000>.

See **[DOCKER_GUIDE.md](DOCKER_GUIDE.md)** for a full walkthrough and troubleshooting.

---

## 📁 Project Structure

```
celebrity_face_match/
├── app.py                     # Streamlit application (face-match demo)
├── feature_extract.py         # Builds embedding.pkl from the data/ images
├── test.py                    # Standalone matching test script
├── data/                      # Celebrity image dataset (100 actors, 8.6k imgs)
├── src/                       # Shared package (config, data, models, inference)
│   ├── config.py              # paths + MLflow settings
│   ├── data_loader.py         # embedding / image dataset loaders
│   ├── models.py              # V1 baseline CNN + V2 transfer head
│   ├── face_utils.py          # MTCNN detect + VGGFace embed
│   └── inference.py           # top-k prediction for the API
├── training/                  # Model training + comparison
│   ├── train.py               # trains V1/V2, logs to MLflow
│   ├── compare.py             # writes comparison_report.md
│   └── comparison_report.md   # V1 vs V2 results
├── deployment/                # Model serving
│   ├── api.py                 # FastAPI: / , /health , /predict
│   └── Dockerfile             # API serving image
├── kubernetes/                # Minikube deployment
│   ├── deployment.yaml
│   └── service.yaml
├── tests/                     # pytest smoke tests (used by CI)
├── models/                    # trained artifacts (v2_transfer.h5, labels.json)
├── .github/workflows/ci-cd.yml# GitHub Actions CI/CD pipeline
├── requirements.txt           # base deps (Streamlit app)
├── requirements-train.txt     # + MLflow (training)
├── requirements-api.txt       # + FastAPI (serving)
├── Dockerfile                 # Streamlit app image
├── DOCKER_GUIDE.md            # Docker reference / instructions
└── README.md
```

---

## 🧠 MLOps Pipeline (Deep-Learning Project Deliverables)

This repository is both a **Streamlit demo** (above) and a full **MLOps pipeline**
built around a trainable celebrity classifier. The sections below map to the
project guidelines.

### 1. Problem statement & justification

| Item | Description |
|------|-------------|
| **Problem** | Given a face photo, identify which of 100 Bollywood celebrities it most resembles (image classification / face recognition). |
| **Real-world significance** | Face-embedding + classification pipelines power photo tagging, identity verification, content personalization and media search. |
| **Dataset** | Bollywood celebrity face dataset — **100 classes, 8,631 images** in `data/`. VGGFace (ResNet-50) embeddings are pre-computed in `embedding.pkl`. |
| **Expected outcome** | A model that classifies a face with high top-1/top-5 accuracy, served via a REST API and deployed to Kubernetes. |

### 2. Two model versions (train + compare)

| | V1 — Baseline | V2 — Improved |
|--|---------------|----------------|
| Approach | 3-block CNN trained from scratch on raw pixels | Transfer learning: frozen VGGFace ResNet-50 embeddings + regularized dense head (dropout, BatchNorm, L2) |
| Val accuracy (top-1) | ~4.8% (short demo run) | **97.1%** |
| Val top-5 accuracy | n/a | **99.1%** |

```bash
pip install -r requirements.txt -r requirements-train.txt

python -m training.train --model v2 --epochs 25      # improved model
python -m training.train --model v1 --epochs 30       # full baseline
python -m training.compare                            # -> training/comparison_report.md
```

### 3. Experiment tracking with MLflow

Training logs parameters, metrics, artifacts (training curves) and **registers
the model** in the MLflow Model Registry (SQLite backend).

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db      # open http://localhost:5000
```

### 4. Model serving — FastAPI

```bash
pip install -r requirements.txt -r requirements-api.txt
uvicorn deployment.api:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | service info |
| `/health` | GET | health / readiness check |
| `/predict` | POST | upload image → predicted celebrity + top-5 |
| `/docs` | GET | interactive Swagger UI |

```bash
curl -X POST http://localhost:8000/predict -F "file=@sample/amir_1.jpg"
```

### 5. Docker (serving image)

```bash
docker build -f deployment/Dockerfile -t celebrity-face-api:v2 .
docker run -p 8000:8000 celebrity-face-api:v2
```

### 6. CI/CD — GitHub Actions

`.github/workflows/ci-cd.yml` runs on every push/PR: **lint + pytest**, then
**builds and health-checks the serving Docker image** on pushes to `main`.

### 7. Kubernetes (Minikube)

```bash
minikube start
eval $(minikube docker-env)                                   # Linux/macOS
docker build -f deployment/Dockerfile -t celebrity-face-api:v2 .
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl get pods
kubectl get services
minikube service celebrity-face-api --url                     # get the URL
```

---

## ⚙️ How It Works

1. **Detect** – MTCNN locates the face and returns a bounding box.
2. **Embed** – the cropped face (224×224) is passed through VGGFace (ResNet-50,
   `include_top=False`, average pooling) to produce a feature vector.
3. **Match** – cosine similarity is computed against every celebrity embedding; the
   highest-scoring image is returned as your look-alike, along with the score.

---



🚀 Powered by TensorFlow • VGGFace • Streamlit
