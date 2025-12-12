# ML Microservices Suite 2025
Image Classification + LLM Text Generation APIs

## Overview
This project contains two production-ready machine learning services built with **FastAPI** as part of **Software Engineering – Task Series 2 (Semester 2, 2025/2026)**.

The system demonstrates how classical ML models and modern LLMs can be exposed as REST APIs, tested, and integrated into a single application.

## Services

| Service | Model | Endpoint Prefix | Description |
|-------|-------|----------------|-------------|
| Image Classification | MobileNetV2 (ImageNet) | `/image` | Classify uploaded images |
| LLM Text Generation | Qwen2.5-0.5B / TinyLlama / Phi-2 | `/llm` | Generate text from prompts |

Both services run inside **one FastAPI app** using router mounting.

---

## Live Endpoints (after running)

- Main API docs: http://127.0.0.1:8000/docs  
- Image API docs: http://127.0.0.1:8000/image/docs  
- LLM API health: http://127.0.0.1:8000/llm/health  

---

## Project Structure

```
ml-microservices-2025/
├── image_api/
│   ├── api.py
│   └── __init__.py
├── llm_api/
│   ├── api.py
│   └── __init__.py
├── main.py              # Single entry point
├── tests/
│   ├── test_image_api.py
│   └── test_llm_api.py
├── requirements.txt
├── .github/workflows/ci.yml
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/your-username/ml-microservices-2025.git
cd ml-microservices-2025

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --reload
```

Server will be available at:
```
http://127.0.0.1:8000
```

---

## Image Classification API

### Endpoints

| Method | Endpoint | Description |
|------|---------|-------------|
| GET | `/image/health` | Health check |
| POST | `/image/predict-image` | Upload image and get predictions |

### Example Response

```json
{
  "predictions": [
    {"class_name": "Egyptian Cat", "score": 0.96},
    {"class_name": "Tabby Cat", "score": 0.02},
    {"class_name": "Tiger Cat", "score": 0.01}
  ]
}
```

---

## LLM Text Generation API

### Endpoints

| Method | Endpoint | Description |
|------|---------|-------------|
| GET | `/llm/health` | Health check |
| POST | `/llm/generate` | Generate text from prompt |

### Example Request

```json
{
  "prompt": "Explain quantum computing like I'm 10",
  "max_new_tokens": 100
}
```

### Example Response

```json
{
  "prompt": "Explain quantum computing like I'm 10",
  "response": "Regular computers use bits that are either 0 or 1..."
}
```

---

## Testing

Run all tests:

```bash
pytest -v
```

Tests cover:
- Health checks
- Valid inference
- Invalid inputs
- Schema validation

---

## Continuous Integration

GitHub Actions CI automatically:
- Sets up Python environment
- Installs dependencies
- Runs full test suite
- Fails build on any error

---

## Learning Outcomes

- Deploying ML and LLM models as REST APIs
- Multi-router FastAPI architecture
- Secure file uploads
- Prompt validation and limits
- Automated testing with pytest
- CI/CD with GitHub Actions
- Clean, professional documentation

---

## Status

Software Engineering – Task Series 2  
Parts 1 & 2: **Completed**

Next steps:
- AI Agent implementation
- Hallucination benchmarking
- Observability and monitoring

---

© 2025 – Academic & portfolio project
