import os
import sys

# Ensure project root is on sys.path
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_agent_health_via_llm():
    """
    Agent should answer a normal text question using LLM.
    """
    response = client.post(
        "/agent/ask",
        json={
            "query": "Explain what machine learning is in one sentence"
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert "answer" in data
    assert "tool_used" in data
    assert data["tool_used"] == "llm"
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0


def test_agent_calculator_tool():
    """
    Agent should detect math expression and use calculator tool.
    """
    response = client.post(
        "/agent/ask",
        json={
            "query": "2 + 3 * 4"
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["tool_used"] == "calculator"
    assert data["answer"] == "14"


def test_agent_invalid_payload():
    """
    Invalid payload should return 422 validation error.
    """
    response = client.post(
        "/agent/ask",
        json={}
    )

    assert response.status_code == 422
