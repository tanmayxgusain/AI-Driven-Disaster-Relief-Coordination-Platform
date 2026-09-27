import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient

from main import app
from backend.services.cv_inference import cv_service
from vision.unet_model import SiameseUNet

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_teardown(tmp_path):
    # Create a synthetic checkpoint for testing
    model = SiameseUNet(in_channels=3, num_classes=4)
    checkpoint_path = str(tmp_path / "synthetic_checkpoint.pt")
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    
    # Set environment variables
    os.environ["CV_CHECKPOINT_PATH"] = checkpoint_path
    os.environ["CV_DEVICE"] = "cpu"
    
    # Reload model for cv_service
    cv_service.load_model()
    
    yield
    
    # Teardown
    os.environ.pop("CV_CHECKPOINT_PATH", None)
    os.environ.pop("CV_DEVICE", None)

def create_dummy_image_bytes(size=(256, 256), color="red"):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_model_loading_success(tmp_path):
    # Already loaded by fixture
    assert cv_service.is_loaded is True
    assert cv_service.device.type == "cpu"

def test_missing_model_behavior(tmp_path):
    os.environ["CV_CHECKPOINT_PATH"] = str(tmp_path / "non_existent.pt")
    cv_service.load_model()
    
    assert cv_service.is_loaded is False
    
    response = client.get("/api/v1/cv/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False
    
    # Inference should fail with 503
    pre_bytes = create_dummy_image_bytes()
    post_bytes = create_dummy_image_bytes()
    response = client.post(
        "/api/v1/cv/infer",
        files={
            "pre_event_image": ("pre.png", pre_bytes, "image/png"),
            "post_event_image": ("post.png", post_bytes, "image/png")
        }
    )
    assert response.status_code == 503
    
    # Restore model for other tests
    os.environ["CV_CHECKPOINT_PATH"] = str(tmp_path / "synthetic_checkpoint.pt")
    torch.save({"model_state_dict": SiameseUNet().state_dict()}, os.environ["CV_CHECKPOINT_PATH"])
    cv_service.load_model()

def test_cuda_fallback_behavior(tmp_path, monkeypatch):
    """Verify that requesting CUDA safely falls back to CPU if CUDA is unavailable."""
    # Force cuda device request
    os.environ["CV_DEVICE"] = "cuda"
    
    # Mock torch.cuda.is_available to return False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    
    # Load model
    success = cv_service.load_model()
    
    assert success is True
    assert cv_service.is_loaded is True
    # It must have fallen back to CPU
    assert cv_service.device.type == "cpu"
    
    # Reset to avoid polluting other tests
    os.environ["CV_DEVICE"] = "cpu"

def test_synthetic_inference():
    pre_img = Image.new("RGB", (512, 512), color="green")
    post_img = Image.new("RGB", (512, 512), color="red")
    
    damage_map, h, w = cv_service.predict_damage(pre_img, post_img)
    
    # Output should be resized by preprocessing to 256x256
    assert h == 256
    assert w == 256
    assert len(damage_map) == 256
    assert len(damage_map[0]) == 256
    
    # Check values are STRICTLY between 0 and 3
    # The model has 4 channels, so argmax can only produce 0, 1, 2, 3.
    # It must NEVER produce the background ignore label 4.
    flat_map = [val for row in damage_map for val in row]
    assert all(isinstance(val, int) for val in flat_map)
    assert min(flat_map) >= 0
    assert max(flat_map) <= 3

def test_mismatched_dimensions_service():
    pre_img = Image.new("RGB", (512, 512))
    post_img = Image.new("RGB", (256, 256))
    
    with pytest.raises(ValueError, match="Image dimension mismatch"):
        cv_service.predict_damage(pre_img, post_img)

def test_api_endpoint_success():
    pre_bytes = create_dummy_image_bytes((512, 512))
    post_bytes = create_dummy_image_bytes((512, 512))
    
    response = client.post(
        "/api/v1/cv/infer",
        files={
            "pre_event_image": ("pre.png", pre_bytes, "image/png"),
            "post_event_image": ("post.png", post_bytes, "image/png")
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["height"] == 256
    assert data["width"] == 256
    assert data["num_classes"] == 4
    assert len(data["damage_map"]) == 256
    assert len(data["damage_map"][0]) == 256

def test_api_invalid_image():
    response = client.post(
        "/api/v1/cv/infer",
        files={
            "pre_event_image": ("pre.png", b"not an image", "image/png"),
            "post_event_image": ("post.png", b"not an image", "image/png")
        }
    )
    assert response.status_code == 400
    assert "valid image" in response.json()["detail"]

def test_api_mismatched_dimensions():
    pre_bytes = create_dummy_image_bytes((512, 512))
    post_bytes = create_dummy_image_bytes((256, 256))
    
    response = client.post(
        "/api/v1/cv/infer",
        files={
            "pre_event_image": ("pre.png", pre_bytes, "image/png"),
            "post_event_image": ("post.png", post_bytes, "image/png")
        }
    )
    
    assert response.status_code == 400
    assert "Mismatched image dimensions" in response.json()["detail"]

def test_health_endpoint():
    response = client.get("/api/v1/cv/health")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "cv"
    assert "model_loaded" in data
    assert "device" in data
