# Image Classification API (FastAPI + MobileNetV2)

## Overview
This project implements an image classification API using FastAPI and MobileNetV2 pre-trained on ImageNet.  
It is part of Software Engineering Task Series 2 (Semester 2, 2025/2026) to learn ML deployment, API building, testing, and CI automation.

## API Endpoints

### GET /health
Returns:
{"status": "ok"}

### POST /predict-image
Accepts an uploaded image, preprocesses it, performs inference, and returns the top-3 predicted labels with confidence scores.

Example response:
{
  "predictions": [
    {"class_id": "n02124075", "class_name": "Egyptian_cat", "score": 0.92},
    {"class_id": "n02123045", "class_name": "tabby_cat", "score": 0.04},
    {"class_id": "n02123159", "class_name": "tiger_cat", "score": 0.02}
  ]
}

## Project Structure
image_api/
 ├── api.py
 └── __init__.py
tests/
 └── test_api.py
.github/
 └── workflows/tests.yml
requirements.txt
README.md

## Installation

Create virtual environment:
python -m venv .venv

Activate:
Windows:
.venv\Scripts\activate

Linux/macOS:
source .venv/bin/activate

Install dependencies:
pip install --upgrade pip
pip install -r requirements.txt

## Running the API
uvicorn image_api.api:app --reload

Swagger UI:
http://127.0.0.1:8000/docs

## Testing
Unit tests verify:
- health endpoint works
- valid inference returns predictions
- invalid upload triggers error

Run tests:
pytest -q

## Continuous Integration
GitHub Actions automatically runs:
- dependency installation
- test execution
- commit validation

## Learning Outcome
This project demonstrates:
- Machine learning model deployment via REST APIs
- Test-driven development principles
- Continuous integration automation
- Reproducible development environments

This completes Task Series 2 Part 1 and Part 2.  
Upcoming tasks include LLM API creation, lightweight model deployment, hallucination evaluation, and agent model selection.