from fastapi import FastAPI
import uvicorn

# Import APIs
from image_api.api import app as image_app
from llm_api.api import router as llm_router
from agent_api.router import router as agent_router

app = FastAPI(title="ML + LLM + AI Agent API")

# Mount Image Classification API
app.mount("/image", image_app)

# Include LLM API
app.include_router(llm_router)

# Include Agent API (TS3)
app.include_router(agent_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
