"""
Tasks 4.2, 4.3, 4.4, Task 5, Task 8: Flask Production Web Application
Author: 23L-0994
Project: 23L-0994_A2_Ops
Zero-disk-storage image classifier with input sanitization, database audit trail,
external cloud API integration, and health check endpoints.
"""
import os
import io
import time
import requests
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from inference import VisionGuardPredictor
from database import init_db, log_prediction, get_all_logs

# Load environment configuration
load_dotenv()

app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "visionguard_secret_23L0994")

# Strict size limit: 5MB (Task 4.2)
MAX_FILE_SIZE_MB = 5
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Configuration from Environment Variables (Task 4.4)
MODEL_PATH = os.getenv("MODEL_PATH", "models/champion_model.keras")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.50"))
USE_CLOUD_API = os.getenv("USE_CLOUD_API", "false").lower() == "true"
CLOUD_API_URL = os.getenv("CLOUD_API_URL", "https://shaheer-visionguard-api.hf.space/predict")

# Initialize database
init_db()

# Initialize predictor singleton (loads champion model ONCE at startup)
predictor = VisionGuardPredictor.get_instance(model_path=MODEL_PATH, threshold=DECISION_THRESHOLD)

def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 1. Check if file is in request
        if 'image' not in request.files:
            return render_template("error.html", code=400, title="Missing File",
                                   message="No file part provided in the upload request."), 400

        file = request.files['image']

        # 2. Check empty filename
        if file.filename == '':
            return render_template("error.html", code=400, title="No File Selected",
                                   message="Please select an image file before clicking Analyze."), 400

        # 3. Sanitize filename (Task 4.2)
        safe_filename = secure_filename(file.filename)

        # 4. Validate allowed file extension (Task 4.2 - Failure Case 1)
        if not is_allowed_file(safe_filename):
            return render_template("error.html", code=400, title="Invalid File Extension",
                                   message=f"File '{safe_filename}' is not supported. Allowed formats: PNG, JPG, JPEG, WEBP."), 400

        # Read into memory (Task 4.2: Do not keep uploaded image on disk)
        file_bytes = file.read()

        # Check for empty file content
        if len(file_bytes) == 0:
            return render_template("error.html", code=400, title="Empty File",
                                   message="Uploaded file contains 0 bytes."), 400

        # 5. Process prediction (Local inference or Cloud API integration - Task 8)
        try:
            if USE_CLOUD_API:
                print(f"[*] Dispatching inference request to Cloud API: {CLOUD_API_URL}")
                t0 = time.time()
                response = requests.post(
                    CLOUD_API_URL,
                    files={"file": (safe_filename, file_bytes, file.mimetype)},
                    timeout=10
                )
                latency_ms = (time.time() - t0) * 1000.0
                if response.status_code == 200:
                    api_data = response.json()
                    result = {
                        "filename": safe_filename,
                        "verdict": api_data.get("verdict", "Unknown"),
                        "confidence": api_data.get("confidence", 0.0),
                        "raw_score": api_data.get("raw_score", 0.5),
                        "latency_ms": round(latency_ms, 2),
                        "threshold": DECISION_THRESHOLD,
                        "model_name": f"Cloud API ({api_data.get('model_name', 'Hosted')})",
                        "caveat": api_data.get("caveat", "Evaluated via cloud inference service.")
                    }
                else:
                    raise RuntimeError(f"Cloud API returned error status {response.status_code}")
            else:
                # Local inference module (Task 4.1)
                pred = predictor.predict(file_bytes)
                result = {
                    "filename": safe_filename,
                    "verdict": pred["verdict"],
                    "confidence": pred["confidence"],
                    "raw_score": pred["raw_score"],
                    "latency_ms": pred["latency_ms"],
                    "threshold": pred["threshold"],
                    "model_name": pred["model_name"],
                    "caveat": pred["caveat"]
                }

            # 6. Database Trail Logging (Task 5)
            log_prediction(
                filename=result["filename"],
                verdict=result["verdict"],
                confidence=result["confidence"],
                raw_score=result["raw_score"],
                model_name=result["model_name"],
                latency_ms=result["latency_ms"]
            )

            return render_template("index.html", result=result)

        except ValueError as ve:
            # Task 4.2 - Failure Case 3: Looks like image but cannot be decoded
            return render_template("error.html", code=400, title="Corrupted Image File",
                                   message=f"The uploaded file could not be decoded as an image. Details: {str(ve)}"), 400
        except Exception as e:
            return render_template("error.html", code=500, title="Inference Pipeline Failure",
                                   message=f"Internal error processing model inference: {str(e)}"), 500

    return render_template("index.html", result=None)

@app.route("/history")
def history():
    """Task 5 & 4.2: Audit history table view."""
    logs = get_all_logs()
    return render_template("history.html", logs=logs)

@app.route("/health")
def health():
    """Task 4.4 & 6.2: Container & service health check endpoint."""
    model_loaded = predictor.model is not None
    db_status = "connected"
    try:
        get_all_logs(limit=1)
    except Exception:
        db_status = "error"

    return jsonify({
        "status": "healthy" if (model_loaded and db_status == "connected") else "degraded",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_loaded": model_loaded,
        "database": db_status,
        "active_model": os.path.basename(MODEL_PATH),
        "cloud_mode": USE_CLOUD_API
    }), 200

# Error Handlers (Task 4.2: Friendly Error Pages)
@app.errorhandler(413)
def request_entity_too_large(error):
    # Task 4.2 - Failure Case 2: Files over size limit
    return render_template("error.html", code=413, title="File Size Exceeded",
                           message=f"The uploaded file exceeds the maximum allowed size limit of {MAX_FILE_SIZE_MB}MB."), 413

@app.errorhandler(404)
def not_found_error(error):
    return render_template("error.html", code=404, title="Page Not Found",
                           message="The requested route does not exist on this server."), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template("error.html", code=500, title="Internal Server Error",
                           message="An unhandled exception occurred in the web server."), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    # Task 4.4: Run in Flask server with debug mode OFF
    print("=" * 65)
    print(f"Starting VisionGuard Web Server on port {port} (Debug: OFF)")
    print("=" * 65)
    app.run(host="0.0.0.0", port=port, debug=False)
