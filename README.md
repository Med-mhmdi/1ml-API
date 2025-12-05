# Image Classification API (MobileNetV2)

This project is part of **Software Engineering – Task Series 2**.

It exposes a small REST API built with **FastAPI** that uses a pre-trained
**MobileNetV2** model (ImageNet) to classify images. The API accepts an
uploaded image and returns the top-3 predicted classes.

## Project structure

- `image_api/api.py` – FastAPI application and model inference logic
- `tests/test_api.py` – unit tests for the API
- `requirements.txt` – Python dependencies
- `.github/workflows/tests.yml` – GitHub Actions workflow that runs the tests

## Installation

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
# On Linux / macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
