"""
Tasks 2.4, 3.3, 3.4, 3.5: MLflow-Instrumented Training Pipeline
Author: 23L-0994
Project: 23L-0994_A2_Ops
Trains models with fixed seeds, callbacks, MLflow logging (manual & autolog),
saves training curves with best epoch marked, registers champion model.
"""
import os
import sys
import argparse
import subprocess
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
from dotenv import load_dotenv

from dataset import create_dataset
from models import build_baseline_cnn, build_transfer_model, unfreeze_transfer_model
from evaluate import evaluate_model_comprehensive

# Load environment variables (.env)
load_dotenv()

def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "git-unavailable"

def get_dvc_version():
    try:
        if os.path.exists(".dvc"):
            return "dvc-v1.0"
        return "local-splits"
    except Exception:
        return "unknown"

class MLflowMetricLogger(tf.keras.callbacks.Callback):
    """Custom Keras callback for manual per-epoch MLflow metric logging (Task 3.3)."""
    def on_epoch_end(self, epoch, logs=None):
        if logs and mlflow.active_run():
            for k, v in logs.items():
                mlflow.log_metric(k, float(v), step=epoch)

def plot_and_save_training_curves(history, output_path="reports/training_curves.png", best_epoch=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(14, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'ro-', label='Validation Loss')
    if best_epoch:
        plt.axvline(x=best_epoch, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.title('Training and Validation Loss', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'ro-', label='Validation Accuracy')
    if best_epoch:
        plt.axvline(x=best_epoch, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.title('Training and Validation Accuracy', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.suptitle("Task 2.4: Training Discipline - Convergence Curves", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Training curves saved to {output_path}")

def train_pipeline(
    model_type="baseline",
    epochs=10,
    batch_size=32,
    lr=1e-3,
    augment=True,
    logging_mode="manual",
    run_name="Run_Baseline",
    experiment_name="VisionGuard_23L0994",
    unfreeze_layers=0,
    seed=42
):
    set_seeds(seed)

    # Setup MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print("=" * 70)
    print(f"VISIONGUARD TRAINING PIPELINE: {run_name}")
    print(f"Model: {model_type} | LR: {lr} | Augment: {augment} | Logging: {logging_mode}")
    print("=" * 70)

    # Datasets
    train_ds = create_dataset("data/splits/train.txt", is_training=True, batch_size=batch_size, augment=augment)
    val_ds = create_dataset("data/splits/val.txt", is_training=False, batch_size=batch_size)
    test_ds = create_dataset("data/splits/test.txt", is_training=False, batch_size=batch_size)

    # Model instantiation
    if model_type == "baseline":
        model = build_baseline_cnn(learning_rate=lr)
    elif model_type == "mobilenet":
        model = build_transfer_model(learning_rate=lr)
        if unfreeze_layers > 0:
            unfreeze_transfer_model(model, unfreeze_layers=unfreeze_layers, fine_tune_lr=lr)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Callbacks (Task 2.4)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/best_checkpoint.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    if logging_mode == "manual":
        callbacks.append(MLflowMetricLogger())

    with mlflow.start_run(run_name=run_name):
        # Tags (Task 3.3)
        git_commit = get_git_commit()
        dvc_version = get_dvc_version()
        mlflow.set_tags({
            "author": "23L-0994",
            "project": "23L-0994_A2_Ops",
            "git_commit": git_commit,
            "dvc_version": dvc_version,
            "seed": str(seed),
            "logging_strategy": logging_mode
        })

        # Hyperparameters
        params = {
            "model_type": model_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "augmentation": augment,
            "unfreeze_layers": unfreeze_layers,
            "optimizer": "Adam",
            "loss_function": "BinaryCrossentropy"
        }
        mlflow.log_params(params)

        if logging_mode == "autolog":
            mlflow.tensorflow.autolog(log_models=True)

        print("[*] Starting Model Fit...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        # Identify best epoch
        val_losses = history.history['val_loss']
        best_epoch = int(np.argmin(val_losses) + 1)
        best_val_loss = float(np.min(val_losses))
        print(f"\n[+] Training Complete! Best Epoch: {best_epoch} with val_loss: {best_val_loss:.4f}")
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("min_val_loss", best_val_loss)

        # Plot curves
        curve_path = f"reports/curves_{run_name}.png"
        plot_and_save_training_curves(history, output_path=curve_path, best_epoch=best_epoch)
        mlflow.log_artifact(curve_path)

        # Final evaluation on test set (Task 2.5)
        print("\n[*] Evaluating on Test Set...")
        eval_metrics = evaluate_model_comprehensive(model, test_ds, val_ds, run_name=run_name)
        for k, v in eval_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"test_{k}", float(v))

        # Save model
        saved_model_path = f"models/{run_name}.keras"
        model.save(saved_model_path)
        print(f"[+] Model saved to {saved_model_path}")

        # Task 3.4: Log Artifacts
        mlflow.log_artifact(saved_model_path)
        if os.path.exists("data/splits/train.txt"):
            mlflow.log_artifact("data/splits/train.txt", artifact_path="splits")
            mlflow.log_artifact("data/splits/val.txt", artifact_path="splits")
            mlflow.log_artifact("data/splits/test.txt", artifact_path="splits")
        if os.path.exists("requirements.txt"):
            mlflow.log_artifact("requirements.txt", artifact_path="environment")
        if os.path.exists(f"reports/confusion_matrix_{run_name}.png"):
            mlflow.log_artifact(f"reports/confusion_matrix_{run_name}.png", artifact_path="evaluation")
        if os.path.exists(f"reports/roc_curve_{run_name}.png"):
            mlflow.log_artifact(f"reports/roc_curve_{run_name}.png", artifact_path="evaluation")

        print("[+] All artifacts successfully logged to MLflow run.")

    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "mobilenet"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--mode", type=str, default="manual", choices=["manual", "autolog"])
    parser.add_argument("--name", type=str, default="Baseline_Run1")
    parser.add_argument("--unfreeze", type=int, default=0)
    args = parser.parse_args()

    train_pipeline(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        augment=not args.no_aug,
        logging_mode=args.mode,
        run_name=args.name,
        unfreeze_layers=args.unfreeze
    )
