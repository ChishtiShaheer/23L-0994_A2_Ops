"""
Unit and Integration Tests for VisionGuard Web Application
Author: 23L-0994
Project: 23L-0994_A2_Ops
Tests endpoints, health checks, database logging, and input validation rejection cases.
"""
import io
import sys
import os
import pytest
import numpy as np
from PIL import Image

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from app import app
from database import init_db, get_all_logs

@pytest.fixture
def client():
    app.config['TESTING'] = True
    init_db()
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test Task 4.4 / 6.2 health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
    assert data["model_loaded"] is True
    assert data["database"] == "connected"

def test_index_get(client):
    """Test web index page rendering."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"VisionGuard" in response.data
    assert b"23L-0994" in response.data

def test_history_get(client):
    """Test history page audit log rendering."""
    response = client.get("/history")
    assert response.status_code == 200
    assert b"Database Audit Trail" in response.data

def test_reject_wrong_file_extension(client):
    """Task 4.2 Failure Case 1: Reject disallowed file type (.txt)."""
    data = {
        'image': (io.BytesIO(b"Hello fake content"), 'malicious.txt')
    }
    response = client.post("/", data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert b"Invalid File Extension" in response.data

def test_reject_corrupt_image(client):
    """Task 4.2 Failure Case 3: File named .jpg but cannot be decoded."""
    data = {
        'image': (io.BytesIO(b"Not a real JPEG binary stream"), 'corrupt.jpg')
    }
    response = client.post("/", data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert b"Corrupted Image File" in response.data

def test_successful_image_inference(client):
    """Test valid image upload, inference execution, and database persistence."""
    # Generate valid synthetic JPEG in memory
    img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='JPEG')
    byte_arr.seek(0)

    data = {
        'image': (byte_arr, 'valid_test_sample.jpg')
    }
    response = client.post("/", data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert b"Forensic Analysis Verdict" in response.data

    # Verify recorded in DB
    logs = get_all_logs(limit=5)
    assert any(log['filename'] == 'valid_test_sample.jpg' for log in logs)
