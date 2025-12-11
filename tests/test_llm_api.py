import os
import sys

# Ensure project root is in path
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from llm_api.api import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_llm_health():
    response = client.get("/llm/health")
    assert response.status_code == 200
    assert response.json() == {"status": "llm ok"}


def test_llm_generate():
    response = client.post("/llm/generate", json={"prompt": "Hello AI"})
    assert response.status_code == 200

    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0


def test_llm_empty_prompt():
    response = client.post("/llm/generate", json={"prompt": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Prompt cannot be empty."
