from fastapi import FastAPI
from image_api.api import app as image_app
from llm_api.api import router as llm_router
import uvicorn

app = FastAPI(title="ML + LLM API")

# Mount ML API
app.mount("/image", image_app)

# Add LLM router
app.include_router(llm_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
