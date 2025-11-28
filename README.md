# Image Classification API (MobileNetV2)

This project is part of **Software Engineering – Task Series 2**.  
It exposes a REST API for an image classification model using **MobileNetV2** pre‑trained on ImageNet.

## Features

- `/health` – health‑check endpoint
- `/predict-image` – accepts an image file and returns top‑3 predicted classes

Example JSON response:

```json
{
  "predictions": [
    {"class_id": "n02124075", "class_name": "Egyptian_cat", "score": 0.92},
    {"class_id": "n02123045", "class_name": "tabby", "score": 0.04},
    {"class_id": "n02123159", "class_name": "tiger_cat", "score": 0.02}
  ]
}
