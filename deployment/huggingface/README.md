---
title: Celebrity Face Match API
emoji: 🎬
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Celebrity Face Match API (FastAPI)

Public deployment of the `celebrity-face-api:v2` container — the same FastAPI
service and V2 transfer-learning model that runs on Kubernetes/Minikube.

Endpoints: `GET /` · `GET /health` · `POST /predict` · interactive docs at `/docs`.

The image is built by cloning the project's GitHub repository, so the model
(`models/v2_transfer.h5`) and code stay in sync with the main project.
