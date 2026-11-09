# VisionGuard (23L-0994_A2_Ops)
![Build Status](https://github.com/23L-0994/23L-0994_A2_Ops/actions/workflows/ci.yml/badge.svg)

**VisionGuard** is an end-to-end MLOps pipeline and web service that classifies images as either **AI-generated** or **Real**.

Developed for **Assignment 2: AI vs Real Image Classification and Deployment**.  
**Author / Roll Number:** `23L-0994`  
**Project Identifier:** `23L-0994_A2_Ops`

---

## 🌟 Key Features
- **Data Engineering & DVC:** Dataset audit (SHA-256 deduplication, dimension stats) and data version control via DVC linked to DagsHub remote storage.
- **Deep Learning Models:**
  - Custom Baseline Convolutional Neural Network (Conv2D, MaxPool, BatchNorm, Dropout).
  - Pretrained Transfer Learning model using **MobileNetV2** (fine-tuned top layers).
- **Hyperparameter Optimization:** Automated tuning via **KerasTuner** (learning rate, dropout rate, unfrozen layers).
- **Experiment Tracking & Model Registry:** Tracked with **MLflow** integrated with **DagsHub**; production model tagged as `champion`.
- **Interpretability:** **Grad-CAM** visual saliency explanations for model predictions.
- **Production Web Application:** Lightweight **Flask** web application with database audit trail (SQLite), strict MIME/dimension validation, sanitized uploads (in-memory processing without writing to disk), and custom error handlers.
- **Containerization & Orchestration:** Production-ready `Dockerfile` (CPU-optimized, non-root user, multi-stage caching) and `docker-compose.yml` with health checks.
- **CI/CD:** Automated testing and Docker build pipeline via **GitHub Actions**.
- **Cloud API Deployment:** Externalized model inference API deployed on cloud with integration to client Flask interface.

---

## 📂 Repository Structure
```
23L-0994_A2_Ops/
├── .github/workflows/ci.yml   # GitHub Actions CI workflow
├── data/                      # Dataset splits and metadata (DVC tracked)
├── src/
│   ├── audit.py               # Dataset audit & SHA-256 duplicate detection
│   ├── eda.py                 # Visual EDA (histograms, sample grids)
│   ├── dataset.py             # tf.data pipeline with augmentation
│   ├── models.py              # Baseline CNN and MobileNetV2 architectures
│   ├── train.py               # MLflow-instrumented training script
│   ├── evaluate.py            # Comprehensive evaluation, ROC, error analysis
│   ├── gradcam.py             # Grad-CAM heatmap generator
│   ├── tune.py                # KerasTuner hyperparameter search
│   ├── toy_mlflow.py          # MLflow local and DagsHub verification run
│   ├── registry_test.py       # MLflow Model Registry champion verification
│   ├── inference.py           # In-memory inference engine
│   ├── database.py            # Database audit trail (SQLite)
│   └── app.py                 # Flask web application
├── templates/                 # HTML templates (index, history, error)
├── static/                    # CSS stylesheet
├── tests/                     # Automated unit and integration tests
├── Dockerfile                 # Slim CPU-only Docker image
├── docker-compose.yml         # Container orchestration configuration
├── .dockerignore
├── .gitignore
├── .env.example
└── requirements.txt
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Local Flask Web App
```bash
cp .env.example .env
python3 src/app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

### 3. Docker Compose Stack
```bash
docker compose up --build
```
Access the application at [http://localhost:5000](http://localhost:5000) and test health via [http://localhost:5000/health](http://localhost:5000/health).
