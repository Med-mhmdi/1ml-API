# image_api/api.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(
    title="Image Classification API",
    description="API that uses MobileNetV2 (ImageNet) to classify images.",
    version="1.0.0",
)

# ---- Load model once at startup ----

model = tf.keras.applications.MobileNetV2(weights="imagenet")
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions


def read_imagefile(file_bytes: bytes) -> Image.Image:
    """Read raw bytes into a PIL Image, ensure RGB."""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return image.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")


@app.get("/health")
def health_check():
    """Simple health endpoint for tests / monitoring."""
    return {"status": "ok"}


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    """
    Accept an image file and return top‑3 predictions from MobileNetV2.

    Request: multipart/form-data with field 'file'
    Response: JSON with 'predictions': list of {class_id, class_name, score}
    """
    if not file:
        raise HTTPException(status_code=400, detail="File is required.")

    contents = await file.read()

    # 1. Load and preprocess image
    image = read_imagefile(contents)
    image = image.resize((224, 224))
    x = np.array(image, dtype=np.float32)
    x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)
    x = preprocess(x)

    # 2. Run model
    preds = model.predict(x)

    # 3. Decode top‑3 predictions
    decoded = decode_predictions(preds, top=3)[0]

    results = [
        {
            "class_id": class_id,
            "class_name": class_name,
            "score": float(score),
        }
        for (class_id, class_name, score) in decoded
    ]

    return JSONResponse(content={"predictions": results})


# Optional: run directly with `python -m image_api.api`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("image_api.api:app", host="0.0.0.0", port=8000, reload=True)
