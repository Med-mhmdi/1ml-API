import requests
import base64
import io


IMAGE_API_URL = "http://127.0.0.1:8000/image/predict-image"


def classify_image(image_base64: str) -> str:
    """
    Sends image to Image Classification API.
    """
    image_bytes = base64.b64decode(image_base64)

    files = {
        "file": ("image.jpg", io.BytesIO(image_bytes), "image/jpeg")
    }

    response = requests.post(IMAGE_API_URL, files=files)

    if response.status_code != 200:
        return "Image classification failed."

    data = response.json()
    preds = data.get("predictions", [])

    if not preds:
        return "No predictions returned."

    top = preds[0]
    return f"Detected object: {top['class_name']} (confidence {top['score']:.2f})"
