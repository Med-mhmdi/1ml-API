# tests/test_api.py

import os
import sys
import io

from fastapi.testclient import TestClient
from PIL import Image
from image_api.api import app

# ----- ensure project root is on sys.path -----
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

client = TestClient(app)


def create_dummy_image_bytes() -> bytes:
    """Create a small red square image in memory for testing."""
    img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"


def test_predict_image_success():
    img_bytes = create_dummy_image_bytes()

    response = client.post(
        "/predict-image",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()

    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) > 0

    first = data["predictions"][0]
    assert "class_id" in first
    assert "class_name" in first
    assert "score" in first


def test_predict_image_invalid_file():
    fake_bytes = b"not an image"

    response = client.post(
        "/predict-image",
        files={"file": ("fake.txt", fake_bytes, "text/plain")},
    )

    assert response.status_code == 400
