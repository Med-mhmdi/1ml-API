import os
import sys
import io
import pytest

# Skip ML tests when running inside GitHub Actions (no TensorFlow available)
if os.getenv("CI") == "true":
    pytest.skip("Skipping ML API tests in CI (TensorFlow too heavy)", allow_module_level=True)

# Ensure project root is on sys.path (works locally + in CI)
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../1ml-API/tests
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))   # .../1ml-API

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from PIL import Image

from image_api.api import app

client = TestClient(app)


def create_dummy_image_bytes() -> bytes:
    """Create a simple in-memory RGB JPEG image for testing."""
    img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data == {"status": "ok"}


def test_predict_image_success():
    image_bytes = create_dummy_image_bytes()

    response = client.post(
        "/predict-image",
        files={"file": ("dummy.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) >= 1

    first = data["predictions"][0]
    assert "class_id" in first
    assert "class_name" in first
    assert "score" in first
    assert isinstance(first["score"], float)


def test_predict_image_invalid_file():
    fake_bytes = b"not an image"

    response = client.post(
        "/predict-image",
        files={"file": ("fake.txt", fake_bytes, "text/plain")},
    )

    assert response.status_code == 400

    data = response.json()
    assert data["detail"] == "Invalid image file."
