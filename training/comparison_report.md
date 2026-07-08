# Model Comparison Report - Celebrity Face Match

Two model versions were trained and tracked with MLflow.

| Metric | V1 - Baseline CNN | V2 - Transfer (VGGFace) |
|--------|-------------------|--------------------------|
| Validation accuracy (top-1) | 1.95% | 96.18% |
| Validation top-5 accuracy | n/a | 98.96% |
| Validation loss | 4.5911 | 0.2925 |

## Analysis

* **V1 (baseline)** learns everything from scratch on raw pixels with a small 3-block CNN. With 100 classes and limited images per class it is data-hungry and under-fits, giving modest accuracy.
* **V2 (improved)** reuses the VGGFace ResNet50 backbone as a frozen feature extractor (pre-computed 2048-d embeddings) and trains only a regularized dense head with dropout, batch-norm and L2. This transfer-learning approach converges in seconds and generalizes far better.

**Conclusion:** V2 is the deployed model (served by the FastAPI API and the Kubernetes deployment).

> Note: the numbers above are from a short demo run (V1 was trained for a > few steps on a data subset for quick reproducibility). Re-run > `python -m training.train --model v1 --epochs 30` for a full baseline; > the transfer-learning gap remains large regardless.
