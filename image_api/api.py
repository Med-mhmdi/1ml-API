from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from typing import List, Dict

IMAGE_SIZE = (224, 224)

app = FastAPI(
    title="Image Classification API",
    description="API that uses MobileNetV2 (ImageNet) to classify images.",
    version="1.0.0",
)

# ---- Load model once at import time ----
model = tf.keras.applications.MobileNetV2(weights="imagenet")
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions


def read_imagefile(file_bytes: bytes) -> Image.Image:
    """Read raw bytes into a RGB Pillow image or raise HTTP 400."""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return image.convert("RGB")
    except Exception as exc:  # Pillow can raise many types
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc


def classify_image(image: Image.Image) -> List[Dict[str, float]]:
    """Run the model on a Pillow image and return top-3 predictions."""
    # 1. Resize and preprocess
    image = image.resize(IMAGE_SIZE)
    x = np.array(image, dtype=np.float32)
    x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)
    x = preprocess(x)

    # 2. Run model
    preds = model.predict(x)

    # 3. Decode top-3 predictions
    decoded = decode_predictions(preds, top=3)[0]

    results: List[Dict[str, float]] = [
        {
            "class_id": class_id,
            "class_name": class_name,
            "score": float(score),
        }
        for (class_id, class_name, score) in decoded
    ]

    return results


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Simple health endpoint for tests / monitoring."""
    return {"status": "ok"}


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)) -> JSONResponse:
    """Accept an uploaded image and return top-3 ImageNet predictions."""
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    image = read_imagefile(file_bytes)
    predictions = classify_image(image)

    return JSONResponse(content={"predictions": predictions})


# Optional: run directly with `python -m image_api.api`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("image_api.api:app", host="127.0.0.1", port=8000, reload=True)
