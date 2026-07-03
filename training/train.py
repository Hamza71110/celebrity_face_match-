"""Train the two required model versions with MLflow tracking.

Usage (run from the project root):

    python -m training.train --model v2 --epochs 30
    python -m training.train --model v1 --epochs 10

    # quick smoke/demo run (subset of data, few epochs):
    python -m training.train --model v2 --epochs 5 --demo

Both runs are logged to MLflow: parameters, metrics (accuracy / top-5 / loss),
the training-curve artifact and the registered model. Compare them with
``mlflow ui`` or ``python -m training.compare``.
"""
import argparse
import json
import os

import numpy as np

from src import config, data_loader, models


def _sk_top_k_accuracy(y_true, probs, k=5):
    topk = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))


def train_v2(args, mlflow):
    """V2 - transfer-learning head on frozen VGGFace embeddings."""
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam

    X, y, classes = data_loader.load_embedding_dataset()

    if args.demo:
        # keep a stratified-ish subset so a run finishes fast for screenshots
        rng = np.random.RandomState(42)
        keep = rng.choice(len(X), size=min(len(X), args.demo_size), replace=False)
        X, y = X[keep], y[keep]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if not args.demo else None
    )

    num_classes = len(classes)
    model = models.build_transfer_head(num_classes)
    model.compile(
        optimizer=Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    mlflow.log_params({
        "model_version": "v2_transfer",
        "backbone": "VGGFace-ResNet50 (frozen)",
        "input": "precomputed 2048-d embeddings",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_classes": num_classes,
        "train_samples": len(X_tr),
        "val_samples": len(X_val),
        "demo": args.demo,
    })

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=2,
    )

    val_probs = model.predict(X_val)
    top1 = float(np.mean(np.argmax(val_probs, axis=1) == y_val))
    top5 = _sk_top_k_accuracy(y_val, val_probs, k=5)
    val_loss = float(history.history["val_loss"][-1])

    _log_results(mlflow, model, classes, history, config.V2_MODEL_PATH,
                 "v2_transfer", top1, top5, val_loss, args)
    return {"version": "v2_transfer", "val_accuracy": top1,
            "val_top5_accuracy": top5, "val_loss": val_loss}


def train_v1(args, mlflow):
    """V1 - baseline CNN trained from raw pixels."""
    from tensorflow.keras.optimizers import Adam

    train_gen, val_gen, classes = data_loader.build_image_generators(
        batch_size=args.batch_size
    )
    num_classes = len(classes)
    model = models.build_baseline_cnn(num_classes)
    model.compile(
        optimizer=Adam(args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    steps = args.demo_steps if args.demo else None

    mlflow.log_params({
        "model_version": "v1_baseline",
        "architecture": "3-block CNN from scratch",
        "input": "raw 224x224 images",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_classes": num_classes,
        "demo": args.demo,
    })

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        steps_per_epoch=steps,
        validation_steps=(args.demo_steps if args.demo else None),
        verbose=2,
    )

    top1 = float(history.history["val_accuracy"][-1])
    val_loss = float(history.history["val_loss"][-1])
    _log_results(mlflow, model, classes, history, config.V1_MODEL_PATH,
                 "v1_baseline", top1, None, val_loss, args)
    return {"version": "v1_baseline", "val_accuracy": top1,
            "val_top5_accuracy": None, "val_loss": val_loss}


def _log_results(mlflow, model, classes, history, model_path, name,
                 top1, top5, val_loss, args):
    config.ensure_models_dir()
    data_loader.save_labels(classes)
    model.save(model_path)

    mlflow.log_metric("val_accuracy", top1)
    if top5 is not None:
        mlflow.log_metric("val_top5_accuracy", top5)
    mlflow.log_metric("val_loss", val_loss)

    # training-curve artifact (headless-safe)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(history.history.get("accuracy", []), label="train")
        ax[0].plot(history.history.get("val_accuracy", []), label="val")
        ax[0].set_title(f"{name} accuracy"); ax[0].legend()
        ax[1].plot(history.history.get("loss", []), label="train")
        ax[1].plot(history.history.get("val_loss", []), label="val")
        ax[1].set_title(f"{name} loss"); ax[1].legend()
        curve_path = os.path.join(config.MODELS_DIR, f"{name}_curves.png")
        fig.tight_layout(); fig.savefig(curve_path); plt.close(fig)
        mlflow.log_artifact(curve_path)
    except Exception as exc:  # plotting must never fail the run
        print("Skipping curve artifact:", exc)

    mlflow.log_artifact(config.LABELS_PATH)
    mlflow.log_artifact(model_path)

    # Register the model in the MLflow Model Registry
    try:
        import mlflow.keras
        mlflow.keras.log_model(
            model, artifact_path="model",
            registered_model_name=f"celebrity-face-{name}",
        )
    except Exception as exc:
        print("Model registry logging skipped:", exc)

    # also persist a plain metrics json next to the model
    with open(os.path.join(config.MODELS_DIR, f"{name}_metrics.json"), "w") as f:
        json.dump({"val_accuracy": top1, "val_top5_accuracy": top5,
                   "val_loss": val_loss}, f, indent=2)


def main():
    p = argparse.ArgumentParser(description="Train celebrity face-match models")
    p.add_argument("--model", choices=["v1", "v2"], required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--demo", action="store_true",
                   help="fast run on a data subset for screenshots")
    p.add_argument("--demo_size", type=int, default=2000)
    p.add_argument("--demo_steps", type=int, default=20)
    args = p.parse_args()

    import mlflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    run_name = f"{args.model}{'-demo' if args.demo else ''}"
    with mlflow.start_run(run_name=run_name):
        if args.model == "v2":
            result = train_v2(args, mlflow)
        else:
            result = train_v1(args, mlflow)
    print("RESULT:", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
