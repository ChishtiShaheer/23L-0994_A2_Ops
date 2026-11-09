"""
Tasks 3.1 & 3.2: MLflow Local and DagsHub Verification Script
Author: 23L-0994
Project: 23L-0994_A2_Ops
Executes two comparison runs on a toy dataset, logging parameters, metrics per epoch,
plot artifacts, and trained models locally or directly to DagsHub.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import mlflow
import mlflow.keras
from dotenv import load_dotenv

load_dotenv()

def export_definitions():
    defs = """
Task 3.1 MLOps Definitions:
1. Experiment: A logical container that groups together related model training runs and iterations aimed at solving a specific ML objective.
2. Run: A single, distinct execution of machine learning training code within an experiment, recording specific hyperparameters, outputs, and artifacts.
3. Parameter: A fixed configuration value set prior to training (e.g., learning rate, batch size, optimizer choice) that governs model architecture and training dynamics.
4. Metric: A quantifiable, dynamic measurement evaluated during or after training (e.g., loss, accuracy, precision) that tracks model convergence and performance over time.
5. Artifact: Any binary, file, or tangible asset produced by a run (e.g., serialized weights, confusion matrix plots, environment configurations, ROC curves).
6. Tag: A key-value metadata label applied to runs or experiments for annotation, filtering, and governance (e.g., git commit hash, author, deployment status).
"""
    os.makedirs("reports", exist_ok=True)
    with open("reports/mlflow_definitions.txt", "w") as f:
        f.write(defs.strip())
    print(defs)

def train_toy_run(learning_rate, run_name, exp_name):
    # Built-in MNIST / Fashion-MNIST toy dataset
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
    x_train, x_val = x_train[:1000] / 255.0, x_val[:200] / 255.0
    y_train, y_val = y_train[:1000], y_val[:200]

    with mlflow.start_run(run_name=run_name):
        # 1. Log tags
        mlflow.set_tags({
            "author": "23L-0994",
            "project": "23L-0994_A2_Ops",
            "task": "3.1_3.2_MLflow_Lab",
            "dataset": "MNIST_Toy"
        })

        # 2. Log parameters
        epochs = 3
        batch_size = 64
        mlflow.log_params({
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "Adam",
            "architecture": "Flatten-Dense64-Dense10"
        })

        # 3. Model
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # 4. Train with custom per-epoch metric logging
        for epoch in range(1, epochs + 1):
            history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size,
                                validation_data=(x_val, y_val), verbose=0)
            train_loss = float(history.history['loss'][0])
            train_acc = float(history.history['accuracy'][0])
            val_loss = float(history.history['val_loss'][0])
            val_acc = float(history.history['val_accuracy'][0])

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            print(f"[{run_name}] Epoch {epoch}: Val Acc = {val_acc:.4f}, Val Loss = {val_loss:.4f}")

        # 5. Plot artifact
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, epochs + 1), [train_loss] * epochs, label='Loss')
        plt.title(f'Run {run_name} Convergence')
        plt.legend()
        plt.tight_layout()
        plot_path = f"reports/plot_{run_name}.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)

        # 6. Log model
        mlflow.keras.log_model(model, artifact_path="model")
        print(f"[+] Run {run_name} finished and logged successfully.")

def run_mlflow_lab(target="local"):
    export_definitions()

    if target == "dagshub":
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri or "dagshub" not in tracking_uri:
            print("Error: MLFLOW_TRACKING_URI not set or does not point to DagsHub in .env.")
            print("Please configure .env with your DagsHub MLflow tracking URL.")
            sys.exit(1)
        mlflow.set_tracking_uri(tracking_uri)
        exp_name = "23L-0994_DagsHub_Toy"
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        exp_name = "23L-0994_Local_Toy"

    mlflow.set_experiment(exp_name)
    print(f"\n[*] Starting MLflow Experiments under target: {target.upper()} ({mlflow.get_tracking_uri()})")

    # Run 1: High Learning Rate
    train_toy_run(learning_rate=0.01, run_name="Toy_Run_LR_0.01", exp_name=exp_name)

    # Run 2: Low Learning Rate
    train_toy_run(learning_rate=0.001, run_name="Toy_Run_LR_0.001", exp_name=exp_name)

    print("\n[+] Both runs logged! Open your MLflow UI to view comparison:")
    if target == "local":
        print("    Command: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    else:
        print(f"    Visit your DagsHub MLflow tab: {os.getenv('MLFLOW_TRACKING_URI')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="local", choices=["local", "dagshub"])
    args = parser.parse_args()
    run_mlflow_lab(target=args.target)
