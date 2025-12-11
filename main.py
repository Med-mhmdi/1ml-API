from fastapi import FastAPI
from image_api.api import app as image_app
from llm_api.api import router as llm_router

app = FastAPI(title="ML + LLM API")

# Mount ML API
app.mount("/image", image_app)

# Add LLM router
app.include_router(llm_router)

# Run: uvicorn main:app --reload
