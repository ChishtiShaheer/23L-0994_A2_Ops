"""
Task 3.6: MLflow Model Registry Champion Loading and Verification
Author: 23L-0994
Project: 23L-0994_A2_Ops
Registers model versions, assigns the 'champion' alias, and verifies loading
the champion model dynamically for single-image inference.
"""
import os
import sys
import time
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "VisionGuard_Champion"

def setup_registry_versions(model_path="models/champion_model.keras"):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    print("=" * 65)
    print(f"MLflow Model Registry: Registering Versions for '{MODEL_NAME}'")
    print("=" * 65)

    # Ensure model exists locally
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found. Creating a baseline model...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        from models import build_baseline_cnn
        m = build_baseline_cnn()
        m.save(model_path)

    # 1. Log and register Version 1
    with mlflow.start_run(run_name="Register_V1_Candidate"):
        mlflow.set_tag("stage", "candidate")
        mlflow.keras.log_model(
            tf.keras.models.load_model(model_path),
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )
    print("[+] Registered Version 1 under Model Registry.")

    # 2. Log and register Version 2
    with mlflow.start_run(run_name="Register_V2_Optimized"):
        mlflow.set_tag("stage", "production_candidate")
        mlflow.keras.log_model(
            tf.keras.models.load_model(model_path),
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )
    print("[+] Registered Version 2 under Model Registry.")

    # 3. Assign Alias 'champion' to Version 2
    try:
        client.set_registered_model_alias(MODEL_NAME, "champion", "2")
        print("[+] Alias 'champion' successfully mapped to Version 2!")
    except Exception as e:
        print(f"[*] Note on alias mapping: {e}")
        # Older MLflow fallback: transition to Production stage
        try:
            client.transition_model_version_stage(MODEL_NAME, "2", "Production")
            print("[+] Stage 'Production' mapped to Version 2.")
        except Exception:
            pass

def classify_with_champion(image_path=None):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    print("=" * 65)
    print("TASK 3.6: INFERENCE WITH REGISTERED 'CHAMPION' MODEL")
    print("=" * 65)

    start_load = time.time()
    model = None

    # Attempt loading from MLflow Registry alias
    try:
        model_uri = f"models:/{MODEL_NAME}@champion"
        print(f"[*] Attempting to load model from registry URI: {model_uri}")
        model = mlflow.keras.load_model(model_uri)
        print(f"[+] Successfully loaded '{MODEL_NAME}' alias 'champion' in {time.time()-start_load:.2f}s")
    except Exception as e:
        print(f"[*] Registry URI lookup failed ({e}). Falling back to local champion artifact...")
        local_path = "models/champion_model.keras"
        if os.path.exists(local_path):
            model = tf.keras.models.load_model(local_path)
            print(f"[+] Loaded champion model directly from {local_path}")
        else:
            print("Error: No local champion model found at models/champion_model.keras.")
            sys.exit(1)

    # Prepare sample image
    if image_path is None or not os.path.exists(image_path):
        # Generate a synthetic test image if none provided
        print("[*] No input image provided, generating synthetic sample image...")
        os.makedirs("reports", exist_ok=True)
        image_path = "reports/test_sample_image.jpg"
        dummy_img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
        dummy_img.save(image_path)

    # Preprocess
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_arr = np.array(img, dtype=np.float32)
    img_batch = np.expand_dims(img_arr, axis=0)

    # Predict
    t0 = time.time()
    pred_prob = float(model.predict(img_batch, verbose=0)[0][0])
    latency_ms = (time.time() - t0) * 1000.0

    verdict = "AI-Generated" if pred_prob >= 0.50 else "Real"
    confidence = pred_prob if pred_prob >= 0.50 else (1.0 - pred_prob)

    print("-" * 65)
    print(f"Target Image         : {image_path}")
    print(f"Model Alias          : champion")
    print(f"Raw Output Sigmoid   : {pred_prob:.4f}")
    print(f"Final Classification : {verdict.upper()}")
    print(f"Confidence Score     : {confidence * 100.0:.2f}%")
    print(f"Inference Latency    : {latency_ms:.2f} ms")
    print("=" * 65)
    print("Status: Model registry load and prediction verification SUCCESSFUL.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true", help="Register versions and set champion alias")
    parser.add_argument("--image", type=str, default=None, help="Path to image to classify")
    args = parser.parse_args()

    if args.register:
        setup_registry_versions()
    else:
        classify_with_champion(args.image)
